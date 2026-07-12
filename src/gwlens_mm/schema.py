"""Versioned logical v2 metadata schema.

This module deliberately models metadata and array references only. Strain
arrays live in the storage layer and are never embedded in a record.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, cast

from .physics.quantities import ImageParity, LensFamily, MorseClass, PrimaryDefinition

SCHEMA_VERSION = "2.0.0-alpha.2"
DETECTOR_SLOTS = ("H1", "L1", "V1")


class SplitName(str, Enum):
    ENGINEERING_SMOKE = "engineering_smoke"
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


@dataclass(frozen=True)
class TimingObservation:
    value_seconds: float
    standard_deviation_seconds: float
    measurement_method: str
    reference: Optional[str]
    deterministic_control: bool = False

    def validate(self) -> None:
        _finite(self.value_seconds, "observed time difference")
        uncertainty = _finite(
            self.standard_deviation_seconds,
            "observed time-difference standard deviation",
        )
        if uncertainty < 0:
            raise ValueError("timing-observation uncertainty must be nonnegative")
        if uncertainty == 0 and not self.deterministic_control:
            raise ValueError("zero timing uncertainty requires deterministic_control=true")
        if self.deterministic_control and uncertainty != 0:
            raise ValueError("deterministic timing controls must have zero uncertainty")
        if not self.measurement_method.strip():
            raise ValueError("timing observation requires a measurement method")


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
        images_by_id = {image.image_id: image for image in self.physical_images}
        selected = {pair.primary_image_id, pair.secondary_image_id}
        if not selected <= set(ids):
            raise ValueError("selected pair is not contained in physical images")
        unselected = set(self.unselected_image_ids)
        censored = set(self.censored_image_ids)
        if len(unselected) != len(self.unselected_image_ids) or len(censored) != len(
            self.censored_image_ids
        ):
            raise ValueError("non-selected image status lists must not contain duplicates")
        if unselected & censored:
            raise ValueError("an image cannot be both unselected and censored")
        expected_extra = set(ids) - selected
        reported_extra = unselected | censored
        if reported_extra != expected_extra:
            missing = sorted(expected_extra - reported_extra)
            unknown = sorted(reported_extra - expected_extra)
            raise ValueError(
                "every non-selected physical image requires exactly one status; "
                f"missing={missing}, invalid={unknown}"
            )
        for image_id in censored:
            reason = images_by_id[image_id].censoring_reason
            if reason is None or not reason.strip():
                raise ValueError("every censored image requires a censoring reason")
        primary = images_by_id[pair.primary_image_id]
        secondary = images_by_id[pair.secondary_image_id]
        if pair.primary_definition is PrimaryDefinition.EARLIEST_ARRIVING:
            if primary.arrival_time_seconds > secondary.arrival_time_seconds and not math.isclose(
                primary.arrival_time_seconds,
                secondary.arrival_time_seconds,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise ValueError("earliest-arriving primary is later than secondary")
        elif pair.primary_definition is PrimaryDefinition.BRIGHTEST:
            if primary.mu_abs < secondary.mu_abs and not math.isclose(
                primary.mu_abs, secondary.mu_abs, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError("brightest primary is fainter than secondary")
        elif pair.primary_definition is PrimaryDefinition.MINIMUM_IMAGE:
            if primary.morse_class is not MorseClass.MINIMUM:
                raise ValueError("minimum-image primary does not have minimum Morse class")
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
    observed_time_difference: TimingObservation
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
        self.observed_time_difference.validate()
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
class ImageAstrometryObservation:
    image_id: str
    position_arcsec: Tuple[float, float]
    covariance_arcsec2: Tuple[Tuple[float, float], Tuple[float, float]]

    def validate(self) -> None:
        if not self.image_id:
            raise ValueError("astrometry observation requires an image ID")
        if len(self.position_arcsec) != 2:
            raise ValueError("astrometry position must be two-dimensional")
        for coordinate in self.position_arcsec:
            _finite(coordinate, "observed image position")
        validate_covariance(self.covariance_arcsec2, 2)


@dataclass(frozen=True)
class EMObservation:
    observed_image_astrometry: Optional[Tuple[ImageAstrometryObservation, ...]]
    observed_lens_center_arcsec: Optional[Tuple[float, float]]
    lens_center_covariance_arcsec2: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
    einstein_radius_arcsec: Optional[ScalarObservation]
    lens_redshift: Optional[ScalarObservation]
    source_redshift: Optional[ScalarObservation]
    velocity_dispersion_km_s: Optional[ScalarObservation]
    modality_availability_mask: Mapping[str, bool]
    censoring_flags: Mapping[str, bool]
    aperture_metadata: Mapping[str, float]
    redshift_ordering_valid: Optional[bool]

    def validate(self, lens_truth: LensTruth) -> None:
        values: Dict[str, Any] = {
            "image_astrometry": self.observed_image_astrometry,
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
        if self.observed_image_astrometry is not None:
            astrometry_ids = [item.image_id for item in self.observed_image_astrometry]
            if len(astrometry_ids) != len(set(astrometry_ids)):
                raise ValueError("astrometry image IDs must be unique")
            physical_ids = {image.image_id for image in lens_truth.physical_images}
            unknown_ids = set(astrometry_ids) - physical_ids
            if unknown_ids:
                raise ValueError(
                    f"astrometry references unknown physical images: {sorted(unknown_ids)}"
                )
            for item in self.observed_image_astrometry:
                item.validate()
        if self.observed_lens_center_arcsec is not None:
            if self.lens_center_covariance_arcsec2 is None:
                raise ValueError("observed lens center requires covariance")
            validate_covariance(self.lens_center_covariance_arcsec2, 2)
            for coordinate in self.observed_lens_center_arcsec:
                _finite(coordinate, "observed lens center")
        elif self.lens_center_covariance_arcsec2 is not None:
            raise ValueError("lens-center covariance exists without observation")
        if self.einstein_radius_arcsec is not None:
            self.einstein_radius_arcsec.validate()
            _positive(self.einstein_radius_arcsec.value, "observed Einstein radius")
        for redshift_name, redshift in (
            ("lens", self.lens_redshift),
            ("source", self.source_redshift),
        ):
            if redshift is not None:
                redshift.validate()
                if redshift.value < 0:
                    raise ValueError(f"observed {redshift_name} redshift must be nonnegative")
        if self.velocity_dispersion_km_s is not None:
            self.velocity_dispersion_km_s.validate()
            _positive(
                self.velocity_dispersion_km_s.value,
                "observed velocity dispersion",
            )
        if self.lens_redshift is not None and self.source_redshift is not None:
            expected_ordering = self.lens_redshift.value < self.source_redshift.value
            if self.redshift_ordering_valid is not expected_ordering:
                raise ValueError("redshift_ordering_valid disagrees with observed point estimates")
        elif self.redshift_ordering_valid is not None:
            raise ValueError("redshift ordering flag requires both redshift observations")


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
class DetectorNoiseReference:
    image_id: str
    detector: str
    segment_id: Optional[str]
    available: bool
    noise_source: str

    def validate(self) -> None:
        if not self.image_id or self.detector not in DETECTOR_SLOTS:
            raise ValueError("noise reference requires an image ID and known detector")
        if not self.noise_source.strip():
            raise ValueError("noise reference requires an explicit noise source")
        if self.available:
            if self.segment_id is None or not self.segment_id.strip():
                raise ValueError("available detector-noise slot requires a segment ID")
            if self.noise_source == "unavailable":
                raise ValueError("available detector-noise slot cannot use unavailable source")
        else:
            if self.segment_id is not None:
                raise ValueError("unavailable detector-noise slot must not have a segment ID")
            if self.noise_source != "unavailable":
                raise ValueError("unavailable detector-noise slot must use unavailable source")


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
    detector_noise_references: Tuple[DetectorNoiseReference, ...]
    source_data_release: Optional[str]
    distribution: DistributionMetadata

    @property
    def used_noise_segment_ids(self) -> Tuple[str, ...]:
        return tuple(
            reference.segment_id
            for reference in self.detector_noise_references
            if reference.segment_id is not None
        )

    def validate(self, pair: PairIndex, gw_observation: GWObservation) -> None:
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
        expected_keys = tuple(
            (image_id, detector)
            for image_id in (pair.primary_image_id, pair.secondary_image_id)
            for detector in DETECTOR_SLOTS
        )
        actual_keys = tuple(
            (reference.image_id, reference.detector) for reference in self.detector_noise_references
        )
        if actual_keys != expected_keys:
            raise ValueError(
                "detector-noise references must be image-major and cover primary/secondary H1/L1/V1"
            )
        for image_index in range(2):
            for detector_index in range(3):
                reference = self.detector_noise_references[image_index * 3 + detector_index]
                reference.validate()
                if (
                    reference.available
                    != gw_observation.detector_availability_mask[image_index][detector_index]
                ):
                    raise ValueError("detector-noise availability disagrees with GW detector mask")
        used_ids = self.used_noise_segment_ids
        if len(used_ids) != len(set(used_ids)):
            raise ValueError("available image-detector slots require unique noise segment IDs")
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
        self.em_observation.validate(self.lens_truth)
        self.provenance.validate(self.pair, self.gw_observation)

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

    def to_json(self, *, indent: Optional[int] = 2) -> str:
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
        if "observed_event_time_difference_seconds" in gw_data:
            raise ValueError("bare observed time differences are forbidden; use TimingObservation")
        gw_data["observed_time_difference"] = TimingObservation(
            **gw_data["observed_time_difference"]
        )
        em_data = dict(data["em_observation"])
        if (
            "observed_image_positions_arcsec" in em_data
            or "image_position_covariances_arcsec2" in em_data
        ):
            raise ValueError(
                "implicit positional astrometry is forbidden; explicit image IDs are required"
            )
        if em_data["observed_image_astrometry"] is not None:
            try:
                em_data["observed_image_astrometry"] = tuple(
                    ImageAstrometryObservation(
                        image_id=item["image_id"],
                        position_arcsec=tuple(item["position_arcsec"]),
                        covariance_arcsec2=cast(
                            Tuple[Tuple[float, float], Tuple[float, float]],
                            tuple(tuple(row) for row in item["covariance_arcsec2"]),
                        ),
                    )
                    for item in em_data["observed_image_astrometry"]
                )
            except (KeyError, TypeError) as error:
                raise ValueError(
                    "each astrometry observation requires image_id, position, and covariance"
                ) from error
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
        if "noise_segment_ids" in provenance_data:
            raise ValueError(
                "image-only noise IDs are forbidden; use detector-specific noise references"
            )
        provenance_data["detector_noise_references"] = tuple(
            DetectorNoiseReference(**item) for item in provenance_data["detector_noise_references"]
        )
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
        "$defs": {
            "ImageAstrometryObservation": {
                "type": "object",
                "additionalProperties": False,
                "required": ["image_id", "position_arcsec", "covariance_arcsec2"],
                "properties": {
                    "image_id": {"type": "string", "minLength": 1},
                    "position_arcsec": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "number"},
                    },
                    "covariance_arcsec2": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "number"},
                        },
                    },
                },
            },
            "TimingObservation": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "value_seconds",
                    "standard_deviation_seconds",
                    "measurement_method",
                    "reference",
                    "deterministic_control",
                ],
                "properties": {
                    "value_seconds": {"type": "number"},
                    "standard_deviation_seconds": {"type": "number", "minimum": 0},
                    "measurement_method": {"type": "string", "minLength": 1},
                    "reference": {"type": ["string", "null"]},
                    "deterministic_control": {"type": "boolean"},
                },
            },
            "DetectorNoiseReference": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "image_id",
                    "detector",
                    "segment_id",
                    "available",
                    "noise_source",
                ],
                "properties": {
                    "image_id": {"type": "string", "minLength": 1},
                    "detector": {"enum": list(DETECTOR_SLOTS)},
                    "segment_id": {"type": ["string", "null"]},
                    "available": {"type": "boolean"},
                    "noise_source": {"type": "string", "minLength": 1},
                },
            },
        },
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
            "gw_observation": {
                "type": "object",
                "required": ["observed_time_difference", "detector_availability_mask"],
                "not": {"required": ["observed_event_time_difference_seconds"]},
                "properties": {
                    "observed_time_difference": {"$ref": "#/$defs/TimingObservation"},
                    "detector_availability_mask": {"type": "array"},
                },
            },
            "em_observation": {
                "type": "object",
                "required": ["observed_image_astrometry", "modality_availability_mask"],
                "not": {
                    "anyOf": [
                        {"required": ["observed_image_positions_arcsec"]},
                        {"required": ["image_position_covariances_arcsec2"]},
                    ]
                },
                "properties": {
                    "observed_image_astrometry": {
                        "type": ["array", "null"],
                        "items": {"$ref": "#/$defs/ImageAstrometryObservation"},
                    },
                    "modality_availability_mask": {"type": "object"},
                },
            },
            "provenance": {
                "type": "object",
                "required": ["detector_noise_references"],
                "not": {"required": ["noise_segment_ids"]},
                "properties": {
                    "detector_noise_references": {
                        "type": "array",
                        "minItems": 6,
                        "maxItems": 6,
                        "items": {"$ref": "#/$defs/DetectorNoiseReference"},
                    }
                },
            },
        },
    }
