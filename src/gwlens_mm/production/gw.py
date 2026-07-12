"""Process-local Phase 3A waveform, selection, noise, and whitening engine."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ..arrays import validate_strain_array_semantics
from ..physics.solver import PhysicalImage
from ..provenance import derive_seed
from ..schema import DETECTOR_SLOTS
from .population import PopulationDraw


@dataclass(frozen=True)
class ImageProjection:
    image: PhysicalImage
    geocent_time: float
    segment_start: float
    clean_time_by_detector: np.ndarray
    snr_by_detector: Tuple[float, float, float]

    @property
    def network_snr(self) -> float:
        return math.sqrt(sum(value**2 for value in self.snr_by_detector))


@dataclass(frozen=True)
class SelectionResult:
    selected: Tuple[ImageProjection, ImageProjection] | None
    passing_image_ids: Tuple[str, ...]
    rejection_reason: str | None


@dataclass(frozen=True)
class StrainProducts:
    noisy: np.ndarray
    clean: np.ndarray
    noise: np.ndarray
    whitened_noise: np.ndarray
    detector_mask: Tuple[Tuple[bool, bool, bool], Tuple[bool, bool, bool]]


def select_earliest_detectable_images(
    projections: Sequence[ImageProjection],
    *,
    network_threshold: float = 10.0,
    detector_threshold: float = 4.0,
    minimum_detectors: int = 2,
) -> SelectionResult:
    passing = tuple(
        projection
        for projection in projections
        if projection.network_snr >= network_threshold
        and sum(value >= detector_threshold for value in projection.snr_by_detector)
        >= minimum_detectors
    )
    if len(passing) < 2:
        return SelectionResult(
            None,
            tuple(projection.image.image_id for projection in passing),
            "fewer_than_two_images_pass_synthetic_selection",
        )
    return SelectionResult(
        (passing[0], passing[1]),
        tuple(projection.image.image_id for projection in passing),
        None,
    )


class ProductionWaveformEngine:
    """Owned by one worker process; never shared between threads."""

    def __init__(self, config: Mapping[str, Any], root_seed: int) -> None:
        self.config = config
        self.root_seed = root_seed
        self.sample_rate = int(config["sample_rate_hz"])
        self.sample_count = int(config["sample_count"])
        self.duration = float(config["duration_seconds"])
        self.merger_offset = float(config["merger_offset_seconds"])
        if self.sample_count != int(self.sample_rate * self.duration):
            raise ValueError("GW sample count is inconsistent with duration and rate")
        if not 0 < self.merger_offset < self.duration:
            raise ValueError("merger offset must lie inside the segment")
        self._bilby = importlib.import_module("bilby")
        self._waveform_generator = self._bilby.gw.WaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sample_rate,
            frequency_domain_source_model=self._bilby.gw.source.lal_binary_black_hole,
            waveform_arguments={
                "waveform_approximant": str(config["waveform"]),
                "reference_frequency": float(config["reference_frequency_hz"]),
                "minimum_frequency": float(config["minimum_frequency_hz"]),
            },
        )

    def source_parameters(self, draw: PopulationDraw) -> Dict[str, float]:
        planck18 = importlib.import_module("astropy.cosmology").Planck18
        redshift_factor = 1.0 + draw.z_source
        source = draw.source_parameters
        return {
            "mass_1": float(source["mass_1_source"] * redshift_factor),
            "mass_2": float(source["mass_2_source"] * redshift_factor),
            "luminosity_distance": float(planck18.luminosity_distance(draw.z_source).value),
            "a_1": float(source["a_1"]),
            "tilt_1": float(source["tilt_1"]),
            "phi_12": float(source["phi_12"]),
            "a_2": float(source["a_2"]),
            "tilt_2": float(source["tilt_2"]),
            "phi_jl": float(source["phi_jl"]),
            "theta_jn": float(source["theta_jn"]),
            "phase": float(source["phase"]),
            "ra": float(source["ra"]),
            "dec": float(source["dec"]),
            "psi": float(source["psi"]),
        }

    @staticmethod
    def _morse_factor(image: PhysicalImage) -> complex:
        return complex(np.exp(-1j * math.pi * image.morse_class.half_integer_index))

    def project_images(
        self,
        draw: PopulationDraw,
        images: Sequence[PhysicalImage],
        base_geocent_time: float,
    ) -> Tuple[ImageProjection, ...]:
        parameters = self.source_parameters(draw)
        polarizations = self._waveform_generator.frequency_domain_strain(parameters)
        projections = []
        for image in images:
            if image.arrival_time_seconds is None:
                raise ValueError("production images require physical arrival times")
            geocent_time = base_geocent_time + image.arrival_time_seconds
            segment_start = geocent_time - self.merger_offset
            response_parameters = {**parameters, "geocent_time": geocent_time}
            factor = math.sqrt(abs(image.mu_signed)) * self._morse_factor(image)
            lensed = {name: values * factor for name, values in polarizations.items()}
            clean = np.zeros((3, self.sample_count), dtype=np.float64)
            snrs = []
            for detector_index, detector in enumerate(DETECTOR_SLOTS):
                interferometer = self._bilby.gw.detector.get_empty_interferometer(detector)
                interferometer.set_strain_data_from_zero_noise(
                    self.sample_rate,
                    self.duration,
                    start_time=segment_start,
                )
                frequency_domain = np.asarray(
                    interferometer.get_detector_response(
                        lensed,
                        response_parameters,
                        frequencies=self._waveform_generator.frequency_array,
                    )
                )
                snr_squared = complex(interferometer.optimal_snr_squared(frequency_domain))
                snrs.append(math.sqrt(max(float(snr_squared.real), 0.0)))
                clean[detector_index] = np.fft.irfft(
                    frequency_domain, n=self.sample_count
                )
            projections.append(
                ImageProjection(
                    image,
                    geocent_time,
                    segment_start,
                    clean,
                    (snrs[0], snrs[1], snrs[2]),
                )
            )
        return tuple(projections)

    def strain_products(
        self,
        pair_id: str,
        selected: Tuple[ImageProjection, ImageProjection],
    ) -> StrainProducts:
        clean = np.zeros((2, 3, self.sample_count), dtype=np.float32)
        noise = np.zeros_like(clean)
        whitened = np.zeros((2, 3, self.sample_count), dtype=np.float64)
        mask = ((True, True, True), (True, True, True))
        for image_index, projection in enumerate(selected):
            clean[image_index] = projection.clean_time_by_detector.astype(np.float32)
            for detector_index, detector in enumerate(DETECTOR_SLOTS):
                interferometer = self._bilby.gw.detector.get_empty_interferometer(detector)
                seed = derive_seed(
                    self.root_seed,
                    "detector_noise",
                    pair_id,
                    projection.image.image_id,
                    detector,
                )
                self._bilby.core.utils.random.seed(seed)
                interferometer.set_strain_data_from_power_spectral_density(
                    self.sample_rate,
                    self.duration,
                    start_time=projection.segment_start,
                )
                noise[image_index, detector_index] = np.asarray(
                    interferometer.strain_data.time_domain_strain,
                    dtype=np.float32,
                )
                whitened[image_index, detector_index] = np.asarray(
                    interferometer.whitened_time_domain_strain,
                    dtype=np.float64,
                )
        noisy = (clean + noise).astype(np.float32)
        validate_strain_array_semantics(noisy, clean, noise, mask)
        if not np.all(np.isfinite(whitened)):
            raise ValueError("whitened noise contains nonfinite values")
        return StrainProducts(noisy, clean, noise, whitened, mask)
