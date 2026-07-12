"""Versioned logical v2 metadata schema.

This module deliberately models metadata and array references only. Strain
arrays live in the storage layer and are never embedded in a record.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .physics.quantities import ImageParity, LensFamily, MorseClass, PrimaryDefinition

SCHEMA_VERSION = "2.0.0-alpha.1"
DETECTOR_SLOTS = ("H1", "L1", "V1")


class SplitName(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    IID_TEST = "iid_test"
    BALANCED_TAIL_DIAGNOSTIC = "balanced_tail_diagnostic"
    LENS_FAMILY_OOD = "lens_family_ood"
    PARAMETER_REGION_OOD = "parameter_region_ood"
    REAL_NOISE_TEST = "real_noise_test"
    WAVEFORM_PSD_MISMATCH_TEST = "waveform_psd_mismatch_test"


class ArrayProductRole(str, Enum):
    NOISY_STRAIN = "noisy_strain"
    CLEAN_INJECTED_SIGNAL = "clean_injected_signal"
    NOISE_REALIZATION = "noise_realization"


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def validate_covariance(matrix: Sequence[Sequence[float]], dimension: int) -> None:
    if len(matrix) != dimension or any(len(row) != dimension for row in matrix):
        raise ValueError(f"covariance must have shape ({dimension}, {dimension})")
    values = [[_finite(value, "covariance entry") for value in row] for row in matrix]
    for i in range(dimension):
        if values[i][i] <= 0:
            raise ValueError("covariance diagonal must be positive")
        for j in range(dimension):
            if not math.isclose(values[i][j], values[j][i], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("covariance must be symmetric")
    # Sylvester checks for the 1x1 and 2x2 covariance blocks used in v2 smoke.
    if dimension == 2 and values[0][0] * values[1][1] - values[0][1] ** 2 <= 0:
        raise ValueError("covariance must be positive definite")


@dataclass(frozen=True)
class PairIndex:
    pair_id: str
    source_id: str
    lens_id: str
    physical_system_id: str
    primary_image_id: str
    secondary_image_id: str
    primary_definition: PrimaryDefinition
    split: SplitName
    lens_family: LensFamily
    proposal_distribution_id: str
    evaluation_prior_id: str
    root_seed: int
    dataset_version: str
    augmentation_parent_id: Optional[str] = None

    def validate(self) -> None:
        required = (
            self.pair_id,
            self.source_id,
            self.lens_id,
            self.physical_system_id,
            self.primary_image_id,
            self.secondary_image_id,
            self.proposal_distribution_id,
            self.evaluation_prior_id,
            self.dataset_version,
        )
        if not all(required):
            raise ValueError("pair-index identifiers must be nonempty")
        if self.primary_image_id == self.secondary_image_id:
            raise ValueError("selected image IDs must differ")
        if self.root_seed < 0:
            raise ValueError("root_seed must be nonnegative")


@dataclass(frozen=True)
class SourceTruth:
    physical_luminosity_distance_mpc: float
    intrinsic_parameters: Mapping[str, float]
    extrinsic_parameters: Mapping[str, float]
    waveform_model: str
    waveform_model_version: str

    def validate(self) -> None:
        _positive(self.physical_luminosity_distance_mpc, "physical_luminosity_distance_mpc")
        if not self.waveform_model or not self.waveform_model_version:
            raise ValueError("waveform model metadata are required")
        for name, value in {
            **self.intrinsic_parameters,
            **self.extrinsic_parameters,
        }.items():
            _finite(value, f"source parameter {name}")


@dataclass(frozen=True)
class ImageTruth:
    image_id: str
    position_arcsec: Tuple[float, float]
    mu_signed: float
    mu_abs: float
    amplitude_factor: float
    arrival_time_seconds: float
    parity: ImageParity
    morse_class: MorseClass
    physically_detectable: bool
    censoring_reason: Optional[str] = None

    def validate(self) -> None:
        if not self.image_id or len(self.position_arcsec) != 2:
            raise ValueError("image ID and two-dimensional position are required")
        for coordinate in self.position_arcsec:
            _finite(coordinate, "image position")
        signed = _finite(self.mu_signed, "mu_signed")
        if signed == 0:
            raise ValueError("mu_signed must be nonzero")
        if not math.isclose(abs(signed), self.mu_abs, rel_tol=1e-10):
            raise ValueError("mu_abs is inconsistent with mu_signed")
        if not math.isclose(math.sqrt(self.mu_abs), self.amplitude_factor, rel_tol=1e-10):
            raise ValueError("amplitude_factor is inconsistent with mu_abs")
        expected_parity = ImageParity.POSITIVE if signed > 0 else ImageParity.NEGATIVE
        if expected_parity is not self.parity:
            raise ValueError("parity is inconsistent with mu_signed")
        _finite(self.arrival_time_seconds, "arrival_time_seconds")


@dataclass(frozen=True)
class LensTruth:
    lens_parameters: Mapping[str, float]
    external_shear: Tuple[float, float]
    external_convergence: float
    model_discrepancy_parameters: Mapping[str, float]
    physical_images: Tuple[ImageTruth, ...]
    unselected_image_ids: Tuple[str, ...]
    censored_image_ids: Tuple[str, ...]

    def validate(self, pair: PairIndex) -> None:
        ids = [image.image_id for image in self.physical_images]
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError("lens truth requires at least two uniquely identified physical images")
        for image in self.physical_images:
            image.validate()
        selected = {pair.primary_image_id, pair.secondary_image_id}
        if not selected <= set(ids):
            raise ValueError("selected pair is not contained in physical images")
        extra = set(self.unselected_image_ids) | set(self.censored_image_ids)
        if not extra <= set(ids) - selected:
            raise ValueError("extra/censored images must be non-selected physical images")
        if set(self.unselected_image_ids) & set(self.censored_image_ids):
            raise ValueError("an image cannot be both unselected and censored")
        if len(self.external_shear) != 2:
            raise ValueError("external shear requires exactly two components")
        for component in self.external_shear:
            _finite(component, "external shear")
        _finite(self.external_convergence, "external_convergence")
        for name, value in {
            **self.lens_parameters,
            **self.model_discrepancy_parameters,
        }.items():
            _finite(value, f"lens parameter {name}")


@dataclass(frozen=True)
class ArrayReference:
    product_role: ArrayProductRole
    uri: str
    dataset_key: str
    shape: Tuple[int, int, int]
    dtype: str
    checksum: Optional[str]

    def validate(self) -> None:
        if not self.uri or not self.dataset_key:
            raise ValueError("array URI and dataset key are required")
        if self.shape[0:2] != (2, 3) or self.shape[2] <= 0:
            raise ValueError("array references must have shape (2 images, 3 detectors, samples)")
        if self.dtype != "float32":
            raise ValueError("v2 smoke array products must be float32")


@dataclass(frozen=True)
class GWObservation:
    array_products: Tuple[ArrayReference, ...]
    detector_slots: Tuple[str, str, str]
    detector_availability_mask: Tuple[Tuple[bool, bool, bool], Tuple[bool, bool, bool]]
    sample_rate_hz: int
    sample_count: int
    segment_start_gps: Tuple[float, float]
    observed_event_time_difference_seconds: float
    preprocessing_version: str
    psd_reference: str
    calibration_reference: Optional[str]
    data_quality_reference: Optional[str]

    def validate(self) -> None:
        if self.detector_slots != DETECTOR_SLOTS:
            raise ValueError(f"detector slots must be {DETECTOR_SLOTS}")
        if len(self.detector_availability_mask) != 2 or any(
            len(row) != 3 for row in self.detector_availability_mask
        ):
            raise ValueError("detector mask must have shape (2, 3)")
        if any(not any(row) for row in self.detector_availability_mask):
            raise ValueError("each selected image requires at least one available detector")
        if self.sample_rate_hz <= 0 or self.sample_count <= 0:
            raise ValueError("sample rate and count must be positive")
        roles = [reference.product_role for reference in self.array_products]
        if set(roles) != set(ArrayProductRole) or len(roles) != len(set(roles)):
            raise ValueError("exactly one noisy, clean, and noise array reference is required")
        for reference in self.array_products:
            reference.validate()
            if reference.shape[2] != self.sample_count:
                raise ValueError("array sample dimension disagrees with GW metadata")
        if not self.preprocessing_version or not self.psd_reference:
            raise ValueError("preprocessing and PSD references are required")
        _finite(self.observed_event_time_difference_seconds, "observed time difference")
        for start in self.segment_start_gps:
            _finite(start, "segment start GPS")


@dataclass(frozen=True)
class ScalarObservation:
    value: float
    standard_deviation: float

    def validate(self) -> None:
        _finite(self.value, "observed value")
        _positive(self.standard_deviation, "observed standard deviation")


@dataclass(frozen=True)
class EMObservation:
    observed_image_positions_arcsec: Optional[Tuple[Tuple[float, float], ...]]
    image_position_covariances_arcsec2: Optional[
        Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]
    ]
    observed_lens_center_arcsec: Optional[Tuple[float, float]]
    lens_center_covariance_arcsec2: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
    einstein_radius_arcsec: Optional[ScalarObservation]
    lens_redshift: Optional[ScalarObservation]
    source_redshift: Optional[ScalarObservation]
    velocity_dispersion_km_s: Optional[ScalarObservation]
    modality_availability_mask: Mapping[str, bool]
    censoring_flags: Mapping[str, bool]
    aperture_metadata: Mapping[str, float]

    def validate(self) -> None:
        values: Dict[str, Any] = {
            "image_positions": self.observed_image_positions_arcsec,
            "lens_center": self.observed_lens_center_arcsec,
            "einstein_radius": self.einstein_radius_arcsec,
            "lens_redshift": self.lens_redshift,
            "source_redshift": self.source_redshift,
            "velocity_dispersion": self.velocity_dispersion_km_s,
        }
        if set(values) != set(self.modality_availability_mask):
            raise ValueError("modality mask must enumerate every EM modality exactly")
        for name, value in values.items():
            if self.modality_availability_mask[name] != (value is not None):
                raise ValueError(f"modality mask is inconsistent for {name}")
        if self.observed_image_positions_arcsec is not None:
            covariance = self.image_position_covariances_arcsec2
            if covariance is None or len(covariance) != len(self.observed_image_positions_arcsec):
                raise ValueError("each observed image position requires one covariance")
            for matrix in covariance:
                validate_covariance(matrix, 2)
            for position in self.observed_image_positions_arcsec:
                if len(position) != 2:
                    raise ValueError("each observed image position must be two-dimensional")
                for coordinate in position:
                    _finite(coordinate, "observed image position")
        elif self.image_position_covariances_arcsec2 is not None:
            raise ValueError("image-position covariance exists without observations")
        if self.observed_lens_center_arcsec is not None:
            if self.lens_center_covariance_arcsec2 is None:
                raise ValueError("observed lens center requires covariance")
            validate_covariance(self.lens_center_covariance_arcsec2, 2)
            for coordinate in self.observed_lens_center_arcsec:
                _finite(coordinate, "observed lens center")
        elif self.lens_center_covariance_arcsec2 is not None:
            raise ValueError("lens-center covariance exists without observation")
        for scalar in (
            self.einstein_radius_arcsec,
            self.lens_redshift,
            self.source_redshift,
            self.velocity_dispersion_km_s,
        ):
            if scalar is not None:
                scalar.validate()


@dataclass(frozen=True)
class DistributionMetadata:
    proposal_log_probability: float
    evaluation_prior_log_probability: float
    importance_weight: float
    weight_valid: bool
    clipping_applied: bool
    clipping_reason: Optional[str]

    def validate(self) -> None:
        _finite(self.proposal_log_probability, "proposal log probability")
        _finite(self.evaluation_prior_log_probability, "evaluation-prior log probability")
        _positive(self.importance_weight, "importance weight")
        if self.clipping_applied and not self.clipping_reason:
            raise ValueError("clipped weights require a reason")


@dataclass(frozen=True)
class Provenance:
    generator_git_commit: str
    configuration_hash: str
    package_versions: Mapping[str, str]
    seed_hierarchy: Mapping[str, int]
    solver_name: str
    solver_version: str
    waveform_model: str
    detector_labels: Tuple[str, str, str]
    noise_segment_ids: Tuple[str, str]
    source_data_release: Optional[str]
    distribution: DistributionMetadata

    def validate(self) -> None:
        if len(self.generator_git_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.generator_git_commit.lower()
        ):
            raise ValueError("generator_git_commit must be a 40-character hexadecimal hash")
        if len(self.configuration_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.configuration_hash.lower()
        ):
            raise ValueError("configuration_hash must be SHA-256 hex")
        if self.detector_labels != DETECTOR_SLOTS:
            raise ValueError("provenance detector labels disagree with schema slots")
        if len(set(self.noise_segment_ids)) != 2:
            raise ValueError("the two selected images require independent noise segment IDs")
        self.distribution.validate()


@dataclass(frozen=True)
class V2Record:
    schema_version: str
    pair: PairIndex
    source_truth: SourceTruth
    lens_truth: LensTruth
    gw_observation: GWObservation
    em_observation: EMObservation
    provenance: Provenance

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version: {self.schema_version}")
        self.pair.validate()
        self.source_truth.validate()
        self.lens_truth.validate(self.pair)
        self.gw_observation.validate()
        self.em_observation.validate()
        self.provenance.validate()

    def to_dict(self) -> Dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, Mapping):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "V2Record":
        pair_data = dict(data["pair"])
        pair_data.update(
            primary_definition=PrimaryDefinition(pair_data["primary_definition"]),
            split=SplitName(pair_data["split"]),
            lens_family=LensFamily(pair_data["lens_family"]),
        )
        lens_data = dict(data["lens_truth"])
        lens_data["physical_images"] = tuple(
            ImageTruth(
                **{
                    **image,
                    "position_arcsec": tuple(image["position_arcsec"]),
                    "parity": ImageParity(image["parity"]),
                    "morse_class": MorseClass(image["morse_class"]),
                }
            )
            for image in lens_data["physical_images"]
        )
        lens_data["external_shear"] = tuple(lens_data["external_shear"])
        lens_data["unselected_image_ids"] = tuple(lens_data["unselected_image_ids"])
        lens_data["censored_image_ids"] = tuple(lens_data["censored_image_ids"])
        gw_data = dict(data["gw_observation"])
        gw_data["array_products"] = tuple(
            ArrayReference(
                **{
                    **reference,
                    "product_role": ArrayProductRole(reference["product_role"]),
                    "shape": tuple(reference["shape"]),
                }
            )
            for reference in gw_data["array_products"]
        )
        gw_data["detector_slots"] = tuple(gw_data["detector_slots"])
        gw_data["detector_availability_mask"] = tuple(
            tuple(row) for row in gw_data["detector_availability_mask"]
        )
        gw_data["segment_start_gps"] = tuple(gw_data["segment_start_gps"])
        em_data = dict(data["em_observation"])
        if em_data["observed_image_positions_arcsec"] is not None:
            em_data["observed_image_positions_arcsec"] = tuple(
                tuple(position) for position in em_data["observed_image_positions_arcsec"]
            )
            em_data["image_position_covariances_arcsec2"] = tuple(
                tuple(tuple(row) for row in matrix)
                for matrix in em_data["image_position_covariances_arcsec2"]
            )
        if em_data["observed_lens_center_arcsec"] is not None:
            em_data["observed_lens_center_arcsec"] = tuple(em_data["observed_lens_center_arcsec"])
            em_data["lens_center_covariance_arcsec2"] = tuple(
                tuple(row) for row in em_data["lens_center_covariance_arcsec2"]
            )
        for field in (
            "einstein_radius_arcsec",
            "lens_redshift",
            "source_redshift",
            "velocity_dispersion_km_s",
        ):
            if em_data[field] is not None:
                em_data[field] = ScalarObservation(**em_data[field])
        provenance_data = dict(data["provenance"])
        provenance_data["detector_labels"] = tuple(provenance_data["detector_labels"])
        provenance_data["noise_segment_ids"] = tuple(provenance_data["noise_segment_ids"])
        provenance_data["distribution"] = DistributionMetadata(**provenance_data["distribution"])
        record = cls(
            schema_version=str(data["schema_version"]),
            pair=PairIndex(**pair_data),
            source_truth=SourceTruth(**data["source_truth"]),
            lens_truth=LensTruth(**lens_data),
            gw_observation=GWObservation(**gw_data),
            em_observation=EMObservation(**em_data),
            provenance=Provenance(**provenance_data),
        )
        record.validate()
        return record

    @classmethod
    def from_json(cls, text: str) -> "V2Record":
        return cls.from_dict(json.loads(text))


def v2_json_schema() -> Dict[str, Any]:
    """Return a lightweight JSON Schema for boundary validation/documentation."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://example.invalid/gwlens-mm/v2/{SCHEMA_VERSION}",
        "title": "GWLens-MM v2 metadata record",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "pair",
            "source_truth",
            "lens_truth",
            "gw_observation",
            "em_observation",
            "provenance",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "pair": {"type": "object"},
            "source_truth": {"type": "object"},
            "lens_truth": {"type": "object"},
            "gw_observation": {"type": "object"},
            "em_observation": {"type": "object"},
            "provenance": {"type": "object"},
        },
    }
