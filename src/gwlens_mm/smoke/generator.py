"""Configuration-driven generation of the 48-pair engineering smoke artifact."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from ..arrays import validate_strain_array_semantics
from ..physics.lenstronomy_adapter import LenstronomyAdapter
from ..physics.quantities import LensFamily, Magnification, MorseClass, PrimaryDefinition
from ..physics.solver import LensSystemSolution, PhysicalImage, SISSolver
from ..provenance import SEED_DOMAINS, configuration_hash, derive_seed
from ..schema import (
    DETECTOR_SLOTS,
    SCHEMA_VERSION,
    ArrayProductRole,
    ArrayReference,
    DetectorNoiseReference,
    DistributionMetadata,
    EMObservation,
    GWObservation,
    ImageAstrometryObservation,
    ImageTruth,
    LensTruth,
    PairIndex,
    Provenance,
    ScalarObservation,
    SourceTruth,
    SplitName,
    TimingObservation,
    V2Record,
)


@dataclass(frozen=True)
class GeneratedPair:
    record: V2Record
    noisy: np.ndarray
    clean: np.ndarray
    noise: np.ndarray
    validation: Mapping[str, float]


class SmokeGenerator:
    """Generate deterministic records; no state is written by this class."""

    def __init__(self, config: Mapping[str, Any], generator_git_commit: str) -> None:
        self.config = config
        self.generator_git_commit = generator_git_commit
        self.config_hash = configuration_hash(config)
        self.root_seed = int(config["root_seed"])
        self.sample_rate = int(config["gw"]["sample_rate_hz"])
        self.sample_count = int(config["gw"]["sample_count"])
        self.duration = float(config["gw"]["duration_seconds"])
        if self.sample_count != round(self.sample_rate * self.duration):
            raise ValueError("sample count must equal duration times sample rate")
        self._bilby = importlib.import_module("bilby")
        self._waveform_generator = self._make_waveform_generator()

    def _make_waveform_generator(self) -> Any:
        source = self._bilby.gw.source.lal_binary_black_hole
        return self._bilby.gw.WaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sample_rate,
            frequency_domain_source_model=source,
            waveform_arguments={
                "waveform_approximant": "IMRPhenomXPHM",
                "reference_frequency": 50.0,
                "minimum_frequency": 20.0,
            },
        )

    def _rng(self, domain: str, *identifiers: str) -> np.random.Generator:
        return np.random.default_rng(derive_seed(self.root_seed, domain, *identifiers))

    @staticmethod
    def _family_for_index(index: int) -> Tuple[LensFamily, int]:
        if index < 24:
            return LensFamily.SIS, index
        if index < 36:
            return LensFamily.SIE_EXTERNAL_SHEAR, index - 24
        if index < 48:
            return LensFamily.EPL_EXTERNAL_SHEAR, index - 36
        raise IndexError("the Phase 1B specification contains exactly 48 pairs")

    def _lens_solution(
        self, family: LensFamily, family_index: int
    ) -> Tuple[LensSystemSolution, Dict[str, float], Tuple[float, float]]:
        if family is LensFamily.SIS:
            y = 0.15 + 0.7 * ((family_index + 0.5) / 24.0)
            parameters = {"einstein_radius_arcsec": 1.0 + 0.01 * family_index}
            source = (y * parameters["einstein_radius_arcsec"], 0.0)
            raw = SISSolver().solve(source, parameters)
            # The analytic solver returns a dimensionless Fermat coordinate.  The smoke
            # adapter supplies a documented one-day scale; no delay occurs inside an array.
            minimum = min(image.arrival_time_dimensionless for image in raw.physical_images)
            images = tuple(
                PhysicalImage(
                    image_id=image.image_id,
                    position_arcsec=image.position_arcsec,
                    mu_signed=image.mu_signed,
                    arrival_time_dimensionless=(image.arrival_time_dimensionless - minimum)
                    * 86400.0,
                    parity=image.parity,
                    morse_class=image.morse_class,
                )
                for image in raw.physical_images
            )
            return (
                LensSystemSolution(
                    family,
                    images,
                    raw.solver_name,
                    raw.solver_version,
                ),
                parameters,
                source,
            )
        quad = family_index % 2 == 0
        source = (0.03, 0.02) if quad else (0.5, 0.1)
        parameters = {
            "einstein_radius_arcsec": 1.1,
            "axis_ratio": 0.72,
            "position_angle_rad": 0.2,
            "shear_gamma1": 0.04,
            "shear_gamma2": -0.02,
        }
        if family is LensFamily.EPL_EXTERNAL_SHEAR:
            parameters.update(axis_ratio=0.75, density_slope=2.1)
        solution = LenstronomyAdapter(family, 0.5, 1.5).solve(source, parameters)
        return solution, parameters, source

    @staticmethod
    def _selected_ids(solution: LensSystemSolution, family_index: int) -> Tuple[str, str]:
        ids = tuple(image.image_id for image in solution.physical_images)
        if len(ids) >= 4 and family_index % 4 == 0:
            return ids[0], ids[2]
        return ids[0], ids[1]

    @staticmethod
    def _mask(index: int) -> Tuple[Tuple[bool, bool, bool], Tuple[bool, bool, bool]]:
        patterns = (
            ((True, True, True), (True, True, True)),
            ((True, True, False), (True, False, True)),
            ((True, False, False), (False, True, True)),
            ((False, True, True), (True, True, False)),
        )
        return patterns[index % len(patterns)]

    def _source_parameters(self, pair_id: str, index: int) -> Dict[str, float]:
        rng = self._rng("source", pair_id)
        return {
            "mass_1": float(55.0 + 8.0 * rng.random()),
            "mass_2": float(32.0 + 7.0 * rng.random()),
            "luminosity_distance": float(900.0 + 200.0 * rng.random()),
            "a_1": 0.2,
            "tilt_1": 0.4,
            "phi_12": 0.3,
            "a_2": 0.1,
            "tilt_2": 0.5,
            "phi_jl": 0.2,
            "theta_jn": 0.7,
            "phase": 0.1 + 0.01 * index,
            "ra": 1.0,
            "dec": 0.3,
            "psi": 0.5,
        }

    @staticmethod
    def _morse_factor(morse: MorseClass) -> complex:
        return complex(np.exp(-1j * math.pi * morse.half_integer_index))

    @staticmethod
    def _safe_norm(values: np.ndarray) -> float:
        norm = float(np.linalg.norm(values))
        return norm if norm > 0.0 else 1e-300

    def _project(
        self,
        polarizations: Mapping[str, np.ndarray],
        parameters: Mapping[str, float],
        detector: str,
    ) -> np.ndarray:
        interferometer = self._bilby.gw.detector.get_empty_interferometer(detector)
        return np.asarray(
            interferometer.get_detector_response(
                polarizations,
                parameters,
                frequencies=self._waveform_generator.frequency_array,
            )
        )

    def _noise(self, detector: str, start: float, seed: int) -> np.ndarray:
        interferometer = self._bilby.gw.detector.get_empty_interferometer(detector)
        # Bilby 2.6 draws PSD noise from bilby.core.utils.random.Generator,
        # not NumPy's legacy global RNG.  Seed that generator explicitly so
        # interrupted staging runs reproduce byte-identical float32 arrays.
        self._bilby.core.utils.random.seed(seed)
        interferometer.set_strain_data_from_power_spectral_density(
            self.sample_rate, self.duration, start_time=start
        )
        return np.asarray(interferometer.strain_data.time_domain_strain, dtype=np.float64)

    def _strain_products(
        self,
        pair_id: str,
        selected: Tuple[ImageTruth, ImageTruth],
        mask: Tuple[Tuple[bool, bool, bool], Tuple[bool, bool, bool]],
        source_parameters: Dict[str, float],
        starts: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        base_parameters = dict(source_parameters)
        polarizations = self._waveform_generator.frequency_domain_strain(base_parameters)
        noisy = np.zeros((2, 3, self.sample_count), dtype=np.float32)
        clean = np.zeros_like(noisy)
        noise = np.zeros_like(noisy)
        relative_errors = []
        morse_errors = []
        for image_index, image in enumerate(selected):
            image_parameters = {**base_parameters, "geocent_time": starts[image_index] + 0.5}
            factor = image.amplitude_factor * self._morse_factor(image.morse_class)
            lensed = {name: values * factor for name, values in polarizations.items()}
            for detector_index, detector in enumerate(DETECTOR_SLOTS):
                if not mask[image_index][detector_index]:
                    continue
                reference_fd = self._project(polarizations, image_parameters, detector)
                lensed_fd = self._project(lensed, image_parameters, detector)
                expected_fd = reference_fd * factor
                denominator = self._safe_norm(expected_fd)
                response_error = float(np.linalg.norm(lensed_fd - expected_fd))
                relative_errors.append(response_error / denominator)
                morse_error = float(
                    np.linalg.norm(
                        lensed_fd / image.amplitude_factor
                        - reference_fd * self._morse_factor(image.morse_class)
                    )
                )
                morse_errors.append(
                    morse_error / self._safe_norm(reference_fd)
                )
                clean_time = np.fft.irfft(lensed_fd, n=self.sample_count)
                noise_time = self._noise(
                    detector,
                    starts[image_index],
                    derive_seed(
                        self.root_seed,
                        "detector_noise",
                        pair_id,
                        image.image_id,
                        detector,
                    ),
                )
                clean[image_index, detector_index] = clean_time.astype(np.float32)
                noise[image_index, detector_index] = noise_time.astype(np.float32)
                noisy[image_index, detector_index] = (
                    clean[image_index, detector_index] + noise[image_index, detector_index]
                )
        validate_strain_array_semantics(noisy, clean, noise, mask)
        return noisy, clean, noise, {
            "matched_response_max_relative_error": max(relative_errors, default=0.0),
            "morse_phase_max_relative_error": max(morse_errors, default=0.0),
            "preprocessing_max_relative_error": max(relative_errors, default=0.0),
        }

    def _em_observation(
        self,
        pair_id: str,
        index: int,
        images: Tuple[ImageTruth, ...],
        lens_parameters: Mapping[str, float],
    ) -> EMObservation:
        rng = self._rng("em_measurement_noise", pair_id)
        astrometry = tuple(
            ImageAstrometryObservation(
                image_id=image.image_id,
                position_arcsec=(
                    image.position_arcsec[0] + float(rng.normal(0.0, 0.01)),
                    image.position_arcsec[1] + float(rng.normal(0.0, 0.01)),
                ),
                covariance_arcsec2=((0.0001, 0.0), (0.0, 0.0001)),
            )
            for image in images
        )
        missing_velocity = index % 3 == 1
        missing_source_redshift = index % 5 == 2
        lens_z = ScalarObservation(0.5 + float(rng.normal(0, 0.01)), 0.02)
        if index == 47:
            source_z = ScalarObservation(0.45, 0.3)
        elif missing_source_redshift:
            source_z = None
        else:
            source_z = ScalarObservation(1.5 + float(rng.normal(0, 0.02)), 0.05)
        velocity = (
            None
            if missing_velocity
            else ScalarObservation(240.0 + float(rng.normal(0, 5)), 10.0)
        )
        lens_center = (float(rng.normal(0, 0.005)), float(rng.normal(0, 0.005)))
        einstein_radius = ScalarObservation(
            float(lens_parameters["einstein_radius_arcsec"] + rng.normal(0, 0.02)), 0.03
        )
        values = {
            "image_astrometry": astrometry,
            "lens_center": lens_center,
            "einstein_radius": einstein_radius,
            "lens_redshift": lens_z,
            "source_redshift": source_z,
            "velocity_dispersion": velocity,
        }
        return EMObservation(
            observed_image_astrometry=astrometry,
            observed_lens_center_arcsec=lens_center,
            lens_center_covariance_arcsec2=((0.000025, 0.0), (0.0, 0.000025)),
            einstein_radius_arcsec=einstein_radius,
            lens_redshift=lens_z,
            source_redshift=source_z,
            velocity_dispersion_km_s=velocity,
            modality_availability_mask={name: value is not None for name, value in values.items()},
            censoring_flags={
                image.image_id: image.censoring_reason is not None for image in images
            },
            aperture_metadata={"radius_arcsec": 1.0, "seeing_fwhm_arcsec": 0.7},
            redshift_ordering_valid=(lens_z.value < source_z.value) if source_z else None,
        )

    def generate(self, index: int, dataset_identifier: str) -> GeneratedPair:
        family, family_index = self._family_for_index(index)
        pair_id = f"smoke-pair-{index:03d}"
        solution, lens_parameters, source_position = self._lens_solution(family, family_index)
        if not solution.valid:
            raise RuntimeError(f"invalid lens solution: {solution.validity_reason}")
        primary_id, secondary_id = self._selected_ids(solution, family_index)
        selected_ids = {primary_id, secondary_id}
        image_truths = []
        unselected: list[str] = []
        censored: list[str] = []
        for image_index, physical in enumerate(solution.physical_images):
            is_extra = physical.image_id not in selected_ids
            is_censored = is_extra and image_index % 2 == 1
            if is_extra:
                (censored if is_censored else unselected).append(physical.image_id)
            magnification = Magnification(physical.mu_signed)
            image_truths.append(
                ImageTruth(
                    image_id=physical.image_id,
                    position_arcsec=physical.position_arcsec,
                    mu_signed=physical.mu_signed,
                    mu_abs=magnification.mu_abs,
                    amplitude_factor=magnification.amplitude_factor,
                    arrival_time_seconds=physical.arrival_time_dimensionless,
                    parity=physical.parity,
                    morse_class=physical.morse_class,
                    physically_detectable=not is_censored,
                    censoring_reason=(
                        "below engineering visibility threshold" if is_censored else None
                    ),
                )
            )
        images = tuple(image_truths)
        image_map = {image.image_id: image for image in images}
        selected = (image_map[primary_id], image_map[secondary_id])
        mask = self._mask(index)
        delay = selected[1].arrival_time_seconds - selected[0].arrival_time_seconds
        base_start = 1126259462.0 + 100000.0 * index
        starts = (base_start, base_start + delay)
        source_parameters = self._source_parameters(pair_id, index)
        noisy, clean, noise, validation = self._strain_products(
            pair_id, selected, mask, source_parameters, starts
        )
        array_uri = f"arrays.zarr#{index}"
        references = tuple(
            ArrayReference(role, array_uri, role.value, (2, 3, self.sample_count), "float32", None)
            for role in ArrayProductRole
        )
        pair = PairIndex(
            pair_id=pair_id,
            source_id=f"smoke-source-{index:03d}",
            lens_id=f"smoke-lens-{index:03d}",
            physical_system_id=f"smoke-system-{index:03d}",
            primary_image_id=primary_id,
            secondary_image_id=secondary_id,
            primary_definition=PrimaryDefinition.EARLIEST_ARRIVING,
            split=SplitName.ENGINEERING_SMOKE,
            lens_family=family,
            proposal_distribution_id="phase1b-engineering-grid-v1",
            evaluation_prior_id="not-applicable-engineering-smoke",
            root_seed=self.root_seed,
            dataset_version=dataset_identifier,
        )
        noise_references = tuple(
            DetectorNoiseReference(
                image_id=image_id,
                detector=detector,
                segment_id=(f"synthetic-{pair_id}-{image_id}-{detector}" if mask[i][j] else None),
                available=mask[i][j],
                noise_source=("synthetic_gaussian_design_psd" if mask[i][j] else "unavailable"),
            )
            for i, image_id in enumerate((primary_id, secondary_id))
            for j, detector in enumerate(DETECTOR_SLOTS)
        )
        seed_hierarchy = {
            domain: derive_seed(self.root_seed, domain, pair_id) for domain in SEED_DOMAINS
        }
        source_truth = SourceTruth(
            physical_luminosity_distance_mpc=source_parameters["luminosity_distance"],
            intrinsic_parameters={
                key: source_parameters[key] for key in ("mass_1", "mass_2", "a_1", "a_2")
            },
            extrinsic_parameters={
                key: source_parameters[key] for key in ("ra", "dec", "psi", "theta_jn")
            },
            waveform_model="IMRPhenomXPHM",
            waveform_model_version=importlib.metadata.version("lalsuite"),
        )
        lens_truth = LensTruth(
            lens_parameters={
                **lens_parameters,
                "source_beta_x_arcsec": source_position[0],
                "source_beta_y_arcsec": source_position[1],
            },
            external_shear=(
                lens_parameters.get("shear_gamma1", 0.0),
                lens_parameters.get("shear_gamma2", 0.0),
            ),
            external_convergence=0.0,
            model_discrepancy_parameters={},
            physical_images=images,
            unselected_image_ids=tuple(unselected),
            censored_image_ids=tuple(censored),
        )
        gw = GWObservation(
            array_products=references,
            detector_slots=DETECTOR_SLOTS,
            detector_availability_mask=mask,
            sample_rate_hz=self.sample_rate,
            sample_count=self.sample_count,
            segment_start_gps=starts,
            observed_time_difference=TimingObservation(
                value_seconds=delay + float(self._rng("pair_selection", pair_id).normal(0, 0.002)),
                standard_deviation_seconds=0.002,
                measurement_method="paired_trigger_geocentric_time",
                reference="synthetic-engineering-observation",
            ),
            preprocessing_version="phase1b-identity-float32-v1",
            psd_reference="bilby-default-design-psd",
            calibration_reference=None,
            data_quality_reference=None,
        )
        record = V2Record(
            schema_version=SCHEMA_VERSION,
            pair=pair,
            source_truth=source_truth,
            lens_truth=lens_truth,
            gw_observation=gw,
            em_observation=self._em_observation(pair_id, index, images, lens_parameters),
            provenance=Provenance(
                generator_git_commit=self.generator_git_commit,
                configuration_hash=self.config_hash,
                package_versions={
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "bilby", "lalsuite", "lenstronomy")
                },
                seed_hierarchy=seed_hierarchy,
                solver_name=solution.solver_name,
                solver_version=solution.solver_version,
                waveform_model="IMRPhenomXPHM",
                detector_labels=DETECTOR_SLOTS,
                detector_noise_references=noise_references,
                source_data_release=None,
                distribution=DistributionMetadata(0.0, 0.0, 1.0, True, False, None),
            ),
        )
        record.validate()
        return GeneratedPair(record, noisy, clean, noise, validation)
