"""End-to-end accepted-count Phase 3A qualification pair generator."""

from __future__ import annotations

import importlib.metadata
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from ..physics.lenstronomy_adapter import LenstronomyAdapter
from ..physics.quantities import Magnification, PrimaryDefinition
from ..physics.solver import apply_mass_sheet_transform
from ..provenance import configuration_hash, derive_seed
from ..schema import (
    DETECTOR_SLOTS,
    ArrayProductRole,
    ArrayReference,
    DetectorNoiseReference,
    DistributionMetadata,
    GWObservation,
    ImageTruth,
    LensTruth,
    PairIndex,
    Provenance,
    SelectionMetadata,
    SourceTruth,
    SplitName,
    V2Record,
)
from .dynamics import galkin_velocity_dispersion, sample_kinematics_nuisance
from .gw import (
    ProductionWaveformEngine,
    StrainProducts,
    select_earliest_detectable_images,
)
from .observations import balanced_em_cell, generate_observations
from .population import sample_population
from .proposal_adapter import ProductionProposalDraw, sample_production_proposal
from .run_control import AttemptRecord


@dataclass(frozen=True)
class GeneratedQualificationPair:
    record: V2Record
    products: StrainProducts
    em_cell: str
    image_multiplicity: int
    timings: Mapping[str, float]


@dataclass(frozen=True)
class AttemptOutcome:
    attempt: AttemptRecord
    generated: Optional[GeneratedQualificationPair]
    timings: Mapping[str, float]


class QualificationGenerator:
    def __init__(
        self,
        config: Mapping[str, Any],
        preregistration: Mapping[str, Any],
        generator_git_commit: str,
        proposal_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.preregistration = preregistration
        self.generator_git_commit = generator_git_commit
        self.config_hash = configuration_hash(config)
        self.root_seed = int(config["root_seed"])
        self.waveforms = ProductionWaveformEngine(config["gw"], self.root_seed)
        self.cells = tuple(preregistration["em_observation_model"]["availability_cells"])
        engineering = config.get("engineering_ab")
        self.engineering_ab = dict(engineering) if engineering is not None else None
        self.proposal_config = proposal_config
        if self.engineering_ab is not None and proposal_config is None:
            raise ValueError("engineering A/B generation requires an exact proposal config")

    def _rejected(
        self,
        attempt_id: int,
        proposal_seed: int,
        family: str,
        em_cell: str,
        reason: str,
        timings: Mapping[str, float],
        proposal: ProductionProposalDraw | None = None,
    ) -> AttemptOutcome:
        namespace = (
            str(self.engineering_ab["id_prefix"])
            if self.engineering_ab is not None
            else "proposal"
        )
        prefix = f"{namespace}-proposal-{attempt_id:08d}"
        return AttemptOutcome(
            AttemptRecord(
                attempt_id,
                proposal_seed,
                family,
                em_cell,
                "rejected",
                reason,
                None,
                f"{prefix}-source",
                f"{prefix}-lens",
                f"{prefix}-system",
                None if proposal is None else proposal.component,
                None if proposal is None else proposal.component_log_densities,
                None if proposal is None else proposal.population.proposal_log_probability,
                None if proposal is None else proposal.population.evaluation_log_probability,
                None
                if proposal is None
                else proposal.population.evaluation_log_probability
                - proposal.population.proposal_log_probability,
            ),
            None,
            timings,
        )

    def generate_attempt(
        self,
        *,
        attempt_id: int,
        accepted_index: int,
        dataset_id: str,
    ) -> AttemptOutcome:
        prefix = (
            str(self.engineering_ab["id_prefix"])
            if self.engineering_ab is not None
            else "qualification"
        )
        pair_id = f"{prefix}-pair-{accepted_index:06d}"
        source_id = f"{prefix}-source-{accepted_index:06d}"
        lens_id = f"{prefix}-lens-{accepted_index:06d}"
        system_id = f"{prefix}-system-{accepted_index:06d}"
        em_cell = balanced_em_cell(accepted_index, system_id, self.cells)
        lens_seed = derive_seed(self.root_seed, "lens", f"attempt-{attempt_id:08d}")
        source_seed = derive_seed(self.root_seed, "source", f"attempt-{attempt_id:08d}")
        proposal: ProductionProposalDraw | None = None
        if self.engineering_ab is None:
            draw = sample_population(
                np.random.default_rng(lens_seed),
                np.random.default_rng(source_seed),
            )
        else:
            proposal_started = time.perf_counter()
            proposal = sample_production_proposal(
                np.random.default_rng(lens_seed),
                mode=str(self.engineering_ab["proposal_mode"]),
                proposal_config=self.proposal_config or {},
            )
            draw = proposal.population
        timings: Dict[str, float] = {}
        if proposal is not None:
            timings["proposal_sample_and_log_density_seconds"] = (
                time.perf_counter() - proposal_started
            )
        started = time.perf_counter()
        parameters = {
            **draw.lens_parameters,
            "numerical_contract": self.preregistration["lens_solver"]["numerical_contract"],
        }
        try:
            baseline = LenstronomyAdapter(
                draw.lens_family, draw.z_lens, draw.z_source
            ).solve(draw.source_position_arcsec, parameters)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            timings["lens_solver_seconds"] = time.perf_counter() - started
            return self._rejected(
                attempt_id,
                lens_seed,
                draw.lens_family.value,
                em_cell,
                f"lens_solver_error:{type(error).__name__}",
                timings,
                proposal,
            )
        timings["lens_solver_seconds"] = time.perf_counter() - started
        if not baseline.valid:
            return self._rejected(
                attempt_id,
                lens_seed,
                draw.lens_family.value,
                em_cell,
                "fewer_than_two_physical_images",
                timings,
                proposal,
            )
        transformed, transformed_source = apply_mass_sheet_transform(
            baseline, draw.source_position_arcsec, draw.external_convergence
        )
        started = time.perf_counter()
        base_time = float(self.config["gps_schedule"]["start_gps"]) + float(
            self.config["gps_schedule"]["stride_seconds"]
        ) * attempt_id
        projections = self.waveforms.project_images(draw, transformed.physical_images, base_time)
        timings["waveform_and_projection_seconds"] = time.perf_counter() - started
        selection = select_earliest_detectable_images(projections)
        ordered_network_snrs = sorted(
            (projection.network_snr for projection in projections), reverse=True
        )
        timings["maximum_network_snr"] = ordered_network_snrs[0]
        timings["second_highest_network_snr"] = (
            ordered_network_snrs[1] if len(ordered_network_snrs) > 1 else 0.0
        )
        timings["passing_image_count"] = float(len(selection.passing_image_ids))
        if selection.selected is None:
            return self._rejected(
                attempt_id,
                lens_seed,
                draw.lens_family.value,
                em_cell,
                str(selection.rejection_reason),
                timings,
                proposal,
            )
        selected = selection.selected
        passing = set(selection.passing_image_ids)
        selected_ids = {projection.image.image_id for projection in selected}
        image_truths = []
        unselected: list[str] = []
        censored: list[str] = []
        for image in transformed.physical_images:
            magnification = Magnification(image.mu_signed)
            is_extra = image.image_id not in selected_ids
            is_censored = is_extra and image.image_id not in passing
            if is_extra:
                (censored if is_censored else unselected).append(image.image_id)
            if image.arrival_time_seconds is None:
                raise ValueError("production image lacks physical arrival time")
            image_truths.append(
                ImageTruth(
                    image.image_id,
                    image.position_arcsec,
                    image.mu_signed,
                    magnification.mu_abs,
                    magnification.amplitude_factor,
                    image.arrival_time_seconds,
                    image.parity,
                    image.morse_class,
                    image.image_id in passing,
                    "below_synthetic_selection_threshold" if is_censored else None,
                )
            )
        image_map = {image.image_id: image for image in transformed.physical_images}
        selected_images = (
            image_map[selected[0].image.image_id],
            image_map[selected[1].image.image_id],
        )
        cell_specification = self.preregistration["em_observation_model"][
            "availability_cells"
        ][em_cell]
        kinematics = None
        kinematics_seed = derive_seed(self.root_seed, "stellar_kinematics", system_id)
        started = time.perf_counter()
        if "velocity_dispersion" in set(cell_specification["modalities"]):
            effective_radius, anisotropy_ratio = sample_kinematics_nuisance(
                np.random.default_rng(kinematics_seed)
            )
            kinematics = galkin_velocity_dispersion(
                lens_family=draw.lens_family,
                z_lens=draw.z_lens,
                z_source=draw.z_source,
                einstein_radius_arcsec=float(draw.lens_parameters["einstein_radius_arcsec"]),
                density_slope=float(draw.lens_parameters["density_slope"]),
                effective_radius_arcsec=effective_radius,
                anisotropy_radius_over_effective_radius=anisotropy_ratio,
                monte_carlo_samples=int(self.config["kinematics"]["samples"]),
                seed=kinematics_seed,
            )
        timings["kinematics_seconds"] = time.perf_counter() - started
        em_seed = derive_seed(self.root_seed, "em_measurement_noise", pair_id)
        observations = generate_observations(
            rng=np.random.default_rng(em_seed),
            accepted_index=accepted_index,
            physical_system_id=system_id,
            draw=draw,
            images=transformed.physical_images,
            selected_images=selected_images,
            kinematics=kinematics,
            em_model=self.preregistration["em_observation_model"],
        )
        started = time.perf_counter()
        products = self.waveforms.strain_products(pair_id, selected)
        timings["noise_and_whitening_seconds"] = time.perf_counter() - started
        references = tuple(
            ArrayReference(
                role,
                (
                    "shards/shard-"
                    f"{accepted_index // int(self.config['pairs_per_shard']):05d}/"
                    f"{role.value}.zarr"
                ),
                role.value,
                (2, 3, int(self.config["gw"]["sample_count"])),
                "float32",
                None,
            )
            for role in ArrayProductRole
        )
        pair = PairIndex(
            pair_id,
            source_id,
            lens_id,
            system_id,
            selected[0].image.image_id,
            selected[1].image.image_id,
            PrimaryDefinition.EARLIEST_ARRIVING,
            SplitName.GENERATOR_QUALIFICATION,
            draw.lens_family,
            str(
                self.engineering_ab["proposal_distribution_id"]
                if self.engineering_ab is not None
                else self.preregistration["proposal_distribution"]["id"]
            ),
            str(
                self.engineering_ab["evaluation_distribution_id"]
                if self.engineering_ab is not None
                else self.preregistration["evaluation_distribution"]["id"]
            ),
            self.root_seed,
            dataset_id,
        )
        noise_references = tuple(
            DetectorNoiseReference(
                projection.image.image_id,
                detector,
                f"synthetic-{pair_id}-{projection.image.image_id}-{detector}",
                True,
                "synthetic_gaussian_curve_conditioned",
            )
            for projection in selected
            for detector in DETECTOR_SLOTS
        )
        selection_metadata = SelectionMetadata(
            {
                projection.image.image_id: {
                    detector: projection.snr_by_detector[index]
                    for index, detector in enumerate(DETECTOR_SLOTS)
                }
                for projection in projections
            },
            {
                projection.image.image_id: projection.network_snr
                for projection in projections
            },
            selection.passing_image_ids,
            "earliest_two_images_passing_detection",
            None,
        )
        waveform_parameters = self.waveforms.source_parameters(draw)
        seed_hierarchy = {
            "lens": lens_seed,
            "source": source_seed,
            "pair_selection": derive_seed(self.root_seed, "pair_selection", pair_id),
            "em_measurement_noise": em_seed,
            "missing_modalities": derive_seed(
                self.root_seed, "missing_modalities", system_id
            ),
            "stellar_kinematics": kinematics_seed,
            "augmentation": derive_seed(self.root_seed, "augmentation", pair_id),
        }
        if self.engineering_ab is not None:
            seed_hierarchy["proposal"] = lens_seed
        seed_hierarchy.update(
            {
                f"detector_noise:{projection.image.image_id}:{detector}": derive_seed(
                    self.root_seed,
                    "detector_noise",
                    pair_id,
                    projection.image.image_id,
                    detector,
                )
                for projection in selected
                for detector in DETECTOR_SLOTS
            }
        )
        record = V2Record(
            str(self.config["schema_version"]),
            pair,
            SourceTruth(
                float(waveform_parameters["luminosity_distance"]),
                {
                    key: float(draw.source_parameters[key])
                    for key in (
                        "mass_1_source",
                        "mass_2_source",
                        "mass_ratio",
                        "a_1",
                        "a_2",
                        "tilt_1",
                        "tilt_2",
                        "phi_12",
                        "phi_jl",
                    )
                },
                {
                    key: float(draw.source_parameters[key])
                    for key in ("ra", "dec", "psi", "theta_jn", "phase")
                }
                | {
                    "source_redshift": float(draw.z_source),
                    "mass_1_detector": float(waveform_parameters["mass_1"]),
                    "mass_2_detector": float(waveform_parameters["mass_2"]),
                },
                str(self.config["gw"]["waveform"]),
                importlib.metadata.version("lalsuite"),
            ),
            LensTruth(
                {
                    **{
                        key: float(value)
                        for key, value in draw.lens_parameters.items()
                        if key != "external_convergence"
                    },
                    "source_beta_x_arcsec": transformed_source[0],
                    "source_beta_y_arcsec": transformed_source[1],
                },
                (
                    float(draw.lens_parameters["shear_gamma1"]),
                    float(draw.lens_parameters["shear_gamma2"]),
                ),
                draw.external_convergence,
                {},
                tuple(image_truths),
                tuple(unselected),
                tuple(censored),
            ),
            GWObservation(
                references,
                DETECTOR_SLOTS,
                products.detector_mask,
                int(self.config["gw"]["sample_rate_hz"]),
                int(self.config["gw"]["sample_count"]),
                (selected[0].segment_start, selected[1].segment_start),
                observations.timing,
                str(self.config["gw"]["preprocessing_version"]),
                "detector_specific_synthetic_gaussian_curve_conditioned",
                None,
                None,
                {
                    detector: f"{specification['file']}:{specification['sha256']}"
                    for detector, specification in self.config["gw"]["psd_curves"].items()
                },
            ),
            observations.em,
            Provenance(
                self.generator_git_commit,
                self.config_hash,
                {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "bilby", "lalsuite", "lenstronomy", "astropy")
                },
                seed_hierarchy,
                transformed.solver_name,
                transformed.solver_version,
                str(self.config["gw"]["waveform"]),
                DETECTOR_SLOTS,
                noise_references,
                None,
                DistributionMetadata(
                    draw.proposal_log_probability,
                    draw.evaluation_log_probability,
                    draw.importance_weight,
                    True,
                    False,
                    None,
                ),
                selection_metadata,
            ),
        )
        record.validate()
        generated = GeneratedQualificationPair(
            record,
            products,
            observations.em_cell,
            len(transformed.physical_images),
            timings,
        )
        return AttemptOutcome(
            AttemptRecord(
                attempt_id,
                lens_seed,
                draw.lens_family.value,
                em_cell,
                "accepted",
                None,
                pair_id,
                source_id,
                lens_id,
                system_id,
                None if proposal is None else proposal.component,
                None if proposal is None else proposal.component_log_densities,
                None if proposal is None else draw.proposal_log_probability,
                None if proposal is None else draw.evaluation_log_probability,
                None
                if proposal is None
                else draw.evaluation_log_probability - draw.proposal_log_probability,
            ),
            generated,
            timings,
        )
