"""Policy-checked feature and target extraction from alpha.3 records."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..policy import InputPolicy
from ..schema import EMObservation, ScalarObservation, V2Record

MODEL_INPUT_FIELDS = (
    "gw_strain_primary",
    "gw_strain_secondary",
    "detector_availability_mask",
    "detector_identity",
    "adopted_lens_family",
    "sample_rate_hz",
    "observed_time_difference",
    "observed_image_astrometry",
    "observed_lens_center_arcsec",
    "observed_lens_center_covariance_arcsec2",
    "observed_einstein_radius_arcsec",
    "observed_einstein_radius_std_arcsec",
    "observed_lens_redshift",
    "observed_lens_redshift_std",
    "observed_source_redshift",
    "observed_source_redshift_std",
    "observed_velocity_dispersion_km_s",
    "observed_velocity_dispersion_std_km_s",
    "observed_kinematics_aperture_inner_radius_arcsec",
    "observed_kinematics_aperture_outer_radius_arcsec",
    "observed_kinematics_seeing_fwhm_arcsec",
    "observed_tracer_effective_radius_arcsec",
    "kinematics_model_reference",
    "observed_external_convergence_mean",
    "observed_external_convergence_std",
    "environment_modality_available",
    "em_modality_availability_mask",
    "em_censoring_flags",
    "psd_reference",
    "preprocessing_version",
)


@dataclass(frozen=True)
class PreparedExample:
    """Numerical arrays plus nondeployable bookkeeping kept outside model inputs."""

    gw_strain: np.ndarray
    detector_mask: np.ndarray
    astrometry_items: np.ndarray
    astrometry_mask: np.ndarray
    scalar_features: np.ndarray
    scalar_mask: np.ndarray
    modality_mask: np.ndarray
    lens_family_condition: np.ndarray
    target: np.ndarray
    physical_system_id: str
    lens_family: str
    em_cell_signature: str
    em_cell: Optional[str] = None


@dataclass(frozen=True)
class InputStandardizer:
    """Training-rung-only scales for heterogeneous continuous observations."""

    scalar_mean: Tuple[float, ...]
    scalar_standard_deviation: Tuple[float, ...]
    astrometry_mean: Tuple[float, float, float, float, float]
    astrometry_standard_deviation: Tuple[float, float, float, float, float]

    @classmethod
    def fit(cls, examples: Sequence[PreparedExample]) -> "InputStandardizer":
        return cls.fit_iterable(examples)

    @classmethod
    def fit_iterable(cls, examples: Iterable[PreparedExample]) -> "InputStandardizer":
        """Fit observed-only statistics without retaining strain examples in memory."""

        scalar_count = np.zeros(22, dtype=np.int64)
        scalar_mean = np.zeros(22, dtype=np.float64)
        scalar_m2 = np.zeros(22, dtype=np.float64)
        astrometry_count = 0
        astrometry_mean = np.zeros(5, dtype=np.float64)
        astrometry_m2 = np.zeros(5, dtype=np.float64)
        example_count = 0
        for example in examples:
            example_count += 1
            values = np.asarray(example.scalar_features, dtype=np.float64)
            observed_mask = np.asarray(example.scalar_mask, dtype=bool)
            if values.shape != (22,) or observed_mask.shape != (22,):
                raise ValueError("streaming standardizer received incompatible scalar features")
            for column in np.flatnonzero(observed_mask):
                scalar_count[column] += 1
                delta = values[column] - scalar_mean[column]
                scalar_mean[column] += delta / scalar_count[column]
                scalar_m2[column] += delta * (values[column] - scalar_mean[column])
            available = np.asarray(example.astrometry_mask, dtype=bool)
            for row in np.asarray(example.astrometry_items[available, :5], dtype=np.float64):
                astrometry_count += 1
                delta = row - astrometry_mean
                astrometry_mean += delta / astrometry_count
                astrometry_m2 += delta * (row - astrometry_mean)
        if example_count < 2:
            raise ValueError("input standardization requires at least two training examples")
        missing = np.flatnonzero(scalar_count == 0)
        if len(missing):
            raise ValueError(f"scalar feature {int(missing[0])} is never observed in training")
        if astrometry_count == 0:
            raise ValueError("no astrometry is observed in the training standardizer fit")
        scalar_scale = np.sqrt(scalar_m2 / scalar_count)
        scalar_scale[scalar_scale <= 0] = 1.0
        astrometry_scale = np.sqrt(astrometry_m2 / astrometry_count)
        astrometry_scale[astrometry_scale <= 0] = 1.0
        return cls(
            tuple(float(value) for value in scalar_mean),
            tuple(float(value) for value in scalar_scale),
            tuple(float(value) for value in astrometry_mean),  # type: ignore[arg-type]
            tuple(float(value) for value in astrometry_scale),  # type: ignore[arg-type]
        )

    def transform(self, example: PreparedExample) -> PreparedExample:
        mean = np.asarray(self.scalar_mean, dtype=np.float32)
        scale = np.asarray(self.scalar_standard_deviation, dtype=np.float32)
        if mean.shape != example.scalar_features.shape or np.any(scale <= 0):
            raise ValueError("scalar standardizer is incompatible with prepared example")
        scalar = ((example.scalar_features - mean) / scale) * example.scalar_mask
        astrometry = example.astrometry_items.copy()
        astrometry_mean = np.asarray(self.astrometry_mean, dtype=np.float32)
        astrometry_scale = np.asarray(
            self.astrometry_standard_deviation, dtype=np.float32
        )
        astrometry[:, :5] = (
            (astrometry[:, :5] - astrometry_mean) / astrometry_scale
        ) * example.astrometry_mask[:, None]
        if np.any(scalar[example.scalar_mask == 0] != 0.0) or np.any(
            astrometry[example.astrometry_mask == 0] != 0.0
        ):
            raise RuntimeError("standardization must preserve exact missing-value zeros")
        return replace(
            example,
            scalar_features=scalar.astype(np.float32),
            astrometry_items=astrometry.astype(np.float32),
        )


def load_input_policy(root: Path) -> InputPolicy:
    policy = InputPolicy.from_files(
        root / "configs/policy/deployable_input_allowlist.json",
        root / "configs/policy/privileged_input_denylist.json",
    )
    policy.validate_model_inputs(MODEL_INPUT_FIELDS)
    return policy


def _append_scalar(
    values: List[float], masks: List[float], value: float, available: bool
) -> None:
    values.append(float(value) if available else 0.0)
    masks.append(1.0 if available else 0.0)


def _append_observation(
    values: List[float], masks: List[float], observation: Optional[ScalarObservation]
) -> None:
    if observation is None:
        _append_scalar(values, masks, 0.0, False)
        _append_scalar(values, masks, 0.0, False)
    else:
        _append_scalar(values, masks, observation.value, True)
        _append_scalar(values, masks, observation.standard_deviation, True)


def _astrometry_features(record: V2Record, maximum_items: int) -> Tuple[np.ndarray, np.ndarray]:
    observed = record.em_observation.observed_image_astrometry or ()
    if len(observed) > maximum_items:
        raise ValueError("observed astrometry exceeds the frozen encoder capacity")
    items = np.zeros((maximum_items, 9), dtype=np.float32)
    mask = np.zeros(maximum_items, dtype=np.float32)
    for index, item in enumerate(observed):
        covariance = item.covariance_arcsec2
        role = (
            (1.0, 0.0, 0.0)
            if item.image_id == record.pair.primary_image_id
            else (
                (0.0, 1.0, 0.0)
                if item.image_id == record.pair.secondary_image_id
                else (0.0, 0.0, 1.0)
            )
        )
        items[index] = np.asarray(
            (
                item.position_arcsec[0],
                item.position_arcsec[1],
                covariance[0][0],
                covariance[0][1],
                covariance[1][1],
                *role,
                float(record.em_observation.censoring_flags.get(item.image_id, False)),
            ),
            dtype=np.float32,
        )
        mask[index] = 1.0
    return items, mask


def _scalar_features(em: EMObservation, record: V2Record) -> Tuple[np.ndarray, np.ndarray]:
    values: List[float] = []
    masks: List[float] = []
    timing = record.gw_observation.observed_time_difference
    _append_scalar(values, masks, timing.value_seconds, True)
    _append_scalar(values, masks, timing.standard_deviation_seconds, True)
    center = em.observed_lens_center_arcsec
    covariance = em.lens_center_covariance_arcsec2
    center_available = center is not None and covariance is not None
    for value in (
        0.0 if center is None else center[0],
        0.0 if center is None else center[1],
        0.0 if covariance is None else covariance[0][0],
        0.0 if covariance is None else covariance[0][1],
        0.0 if covariance is None else covariance[1][1],
    ):
        _append_scalar(values, masks, value, center_available)
    _append_observation(values, masks, em.einstein_radius_arcsec)
    _append_observation(values, masks, em.lens_redshift)
    _append_observation(values, masks, em.source_redshift)
    _append_observation(values, masks, em.velocity_dispersion_km_s)
    for key in (
        "aperture_inner_radius_arcsec",
        "aperture_outer_radius_arcsec",
        "seeing_fwhm_arcsec",
    ):
        available = key in em.aperture_metadata
        _append_scalar(values, masks, em.aperture_metadata.get(key, 0.0), available)
    _append_scalar(
        values,
        masks,
        0.0 if em.tracer_effective_radius_arcsec is None else em.tracer_effective_radius_arcsec,
        em.tracer_effective_radius_arcsec is not None,
    )
    environment = em.external_convergence_observation
    _append_scalar(
        values,
        masks,
        0.0 if environment is None else environment.posterior_mean,
        environment is not None,
    )
    _append_scalar(
        values,
        masks,
        0.0 if environment is None else environment.posterior_standard_deviation,
        environment is not None,
    )
    _append_scalar(
        values,
        masks,
        1.0 if em.dynamics_model_reference is not None else 0.0,
        em.dynamics_model_reference is not None,
    )
    result = np.asarray(values, dtype=np.float32)
    result_mask = np.asarray(masks, dtype=np.float32)
    if result.shape != (22,) or result_mask.shape != (22,):
        raise RuntimeError("scalar feature contract drifted from 22 values")
    return result, result_mask


def _target(record: V2Record) -> np.ndarray:
    images = {image.image_id: image for image in record.lens_truth.physical_images}
    primary = images[record.pair.primary_image_id].mu_abs
    secondary = images[record.pair.secondary_image_id].mu_abs
    if primary <= 0 or secondary <= 0:
        raise ValueError("absolute-magnification targets must be positive")
    target = np.asarray((math.log(primary), math.log(secondary)), dtype=np.float32)
    if not np.all(np.isfinite(target)):
        raise ValueError("log-magnification target is nonfinite")
    return target


def _assemble_example(
    record: V2Record, strain: np.ndarray, maximum_astrometry_items: int
) -> PreparedExample:
    detector_mask = np.asarray(
        record.gw_observation.detector_availability_mask, dtype=np.float32
    )
    astrometry, astrometry_mask = _astrometry_features(record, maximum_astrometry_items)
    scalars, scalar_mask = _scalar_features(record.em_observation, record)
    modality_names = tuple(sorted(record.em_observation.modality_availability_mask))
    modality_mask = np.asarray(
        [record.em_observation.modality_availability_mask[name] for name in modality_names],
        dtype=np.float32,
    )
    signature = ",".join(
        name
        for name in modality_names
        if record.em_observation.modality_availability_mask[name]
    )
    family_values = ("sie_external_shear", "epl_external_shear")
    if record.pair.lens_family.value not in family_values:
        raise ValueError("probe supports only the two preregistered lens-family hypotheses")
    lens_family_condition = np.zeros(2, dtype=np.float32)
    lens_family_condition[family_values.index(record.pair.lens_family.value)] = 1.0
    return PreparedExample(
        gw_strain=strain,
        detector_mask=detector_mask,
        astrometry_items=astrometry,
        astrometry_mask=astrometry_mask,
        scalar_features=scalars,
        scalar_mask=scalar_mask,
        modality_mask=modality_mask,
        lens_family_condition=lens_family_condition,
        target=_target(record),
        physical_system_id=record.pair.physical_system_id,
        lens_family=record.pair.lens_family.value,
        em_cell_signature=signature,
    )


def prepare_example(
    record: V2Record,
    whitened_noisy_strain: np.ndarray,
    *,
    maximum_astrometry_items: int = 5,
) -> PreparedExample:
    """Prepare allowlisted observations; truth enters only the explicit target."""

    record.validate()
    strain = np.asarray(whitened_noisy_strain, dtype=np.float32)
    if strain.shape != (2, 3, record.gw_observation.sample_count):
        raise ValueError("whitened noisy strain has the wrong shape")
    if not np.all(np.isfinite(strain)):
        raise ValueError("whitened noisy strain contains NaN or Inf")
    detector_mask = np.asarray(
        record.gw_observation.detector_availability_mask, dtype=np.float32
    )
    if np.any(strain[detector_mask == 0] != 0.0):
        raise ValueError("unavailable whitened detector slots must be zero")
    return _assemble_example(record, strain, maximum_astrometry_items)


def prepare_metadata_example(
    record: V2Record,
    *,
    maximum_astrometry_items: int = 5,
) -> PreparedExample:
    """Prepare only metadata and targets for bounded-memory standardizer fitting."""

    record.validate()
    return _assemble_example(
        record,
        np.empty((0,), dtype=np.float32),
        maximum_astrometry_items,
    )


def collate_numpy(examples: Sequence[PreparedExample]) -> Mapping[str, np.ndarray]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    array_fields = (
        "gw_strain",
        "detector_mask",
        "astrometry_items",
        "astrometry_mask",
        "scalar_features",
        "scalar_mask",
        "modality_mask",
        "lens_family_condition",
        "target",
    )
    return {
        field: np.stack([getattr(example, field) for example in examples])
        for field in array_fields
    }
