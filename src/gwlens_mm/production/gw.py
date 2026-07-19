"""Process-local Phase 3A waveform, selection, noise, and whitening engine."""

from __future__ import annotations

import hashlib
import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ..arrays import validate_strain_array_semantics
from ..physics.solver import PhysicalImage
from ..provenance import derive_seed
from ..schema import DETECTOR_SLOTS
from .population import PopulationDraw


class WaveformNumericalPathology(ValueError):
    """A finite but numerically invalid source-polarization spectrum."""


@dataclass(frozen=True)
class SourcePolarizationValidity:
    """Frozen isolated-bin diagnostic for one source waveform."""

    maximum_peak_to_quantile_ratio: float
    peak_to_quantile_ratio_by_polarization: Mapping[str, float]
    positive_in_band_bin_count_by_polarization: Mapping[str, int]


def validate_source_polarization_spectrum(
    polarizations: Mapping[str, np.ndarray],
    frequencies_hz: np.ndarray,
    *,
    minimum_frequency_hz: float,
    positive_amplitude_quantile: float,
    maximum_peak_to_quantile_ratio: float,
) -> SourcePolarizationValidity:
    """Reject isolated IMRPhenomXPHM spectral spikes before lensing or selection.

    The statistic is evaluated independently for plus and cross.  It considers
    strictly positive amplitudes at frequencies at or above the declared lower
    waveform cutoff, computes the frozen high quantile, and compares the peak
    with that quantile.  Zeros beyond physical waveform support therefore do not
    change the diagnostic.
    """

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or not np.all(np.isfinite(frequencies)):
        raise WaveformNumericalPathology("waveform frequency grid is invalid")
    if minimum_frequency_hz <= 0.0:
        raise ValueError("minimum waveform-validation frequency must be positive")
    if not 0.0 < positive_amplitude_quantile < 1.0:
        raise ValueError("waveform-validation quantile must lie strictly inside (0, 1)")
    if maximum_peak_to_quantile_ratio <= 1.0:
        raise ValueError("waveform-validation peak ratio must exceed one")
    in_band = frequencies >= minimum_frequency_hz
    if not np.any(in_band):
        raise WaveformNumericalPathology("waveform frequency grid has no in-band bins")

    ratios: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    if set(polarizations) != {"plus", "cross"}:
        raise WaveformNumericalPathology("waveform polarizations are incomplete")
    for name in ("plus", "cross"):
        values = np.asarray(polarizations[name])
        if values.shape != frequencies.shape or not np.all(np.isfinite(values)):
            raise WaveformNumericalPathology(
                f"{name} source polarization is nonfinite or has the wrong shape"
            )
        amplitudes = np.abs(values[in_band])
        positive = amplitudes[amplitudes > 0.0]
        if positive.size == 0:
            raise WaveformNumericalPathology(
                f"{name} source polarization has no positive in-band support"
            )
        reference = float(np.quantile(positive, positive_amplitude_quantile))
        peak = float(np.max(positive))
        ratio = peak / reference if reference > 0.0 else math.inf
        if not math.isfinite(ratio):
            raise WaveformNumericalPathology(
                f"{name} source-polarization peak ratio is nonfinite"
            )
        ratios[name] = ratio
        counts[name] = int(positive.size)

    maximum = max(ratios.values())
    if maximum > maximum_peak_to_quantile_ratio:
        raise WaveformNumericalPathology(
            "source-polarization isolated-bin ratio "
            f"{maximum:.17g} exceeds {maximum_peak_to_quantile_ratio:.17g}"
        )
    return SourcePolarizationValidity(maximum, ratios, counts)


def psd_file_keyword(filename: str) -> str:
    """Return Bilby's explicit curve-file constructor keyword."""

    name = filename.lower()
    if name.endswith("_asd.txt"):
        return "asd_file"
    if name.endswith("_psd.txt"):
        return "psd_file"
    raise ValueError("PSD curve filename must declare ASD or PSD semantics")


def raised_cosine_guard_window(
    sample_count: int,
    zero_guard_samples: int,
    transition_samples: int,
) -> np.ndarray:
    """Return the exact RC.5 zero-guard and half-cosine conditioning window."""

    if sample_count <= 0 or zero_guard_samples <= 0 or transition_samples <= 0:
        raise ValueError("window sample counts must be positive")
    if 2 * (zero_guard_samples + transition_samples) >= sample_count:
        raise ValueError("edge conditioning leaves no unit-weight interior")
    window = np.ones(sample_count, dtype=np.float64)
    window[:zero_guard_samples] = 0.0
    window[-zero_guard_samples:] = 0.0
    phase = np.pi * (np.arange(transition_samples, dtype=np.float64) + 1.0) / (
        transition_samples + 1.0
    )
    ramp = 0.5 - 0.5 * np.cos(phase)
    leading_start = zero_guard_samples
    leading_stop = leading_start + transition_samples
    window[leading_start:leading_stop] = ramp
    trailing_stop = sample_count - zero_guard_samples
    trailing_start = trailing_stop - transition_samples
    window[trailing_start:trailing_stop] = ramp[::-1]
    return window


def detector_frame_newtonian_chirp_time_seconds(
    mass_1_detector_solar: float,
    mass_2_detector_solar: float,
    minimum_frequency_hz: float,
) -> float:
    """Leading-order detector-frame time from the declared low-frequency edge."""

    if min(mass_1_detector_solar, mass_2_detector_solar, minimum_frequency_hz) <= 0:
        raise ValueError("chirp-time inputs must be positive")
    total = mass_1_detector_solar + mass_2_detector_solar
    eta = mass_1_detector_solar * mass_2_detector_solar / total**2
    chirp_mass = total * eta ** (3.0 / 5.0)
    solar_mass_time_seconds = 4.9254909476412675e-6
    return float(
        5.0
        / 256.0
        * (np.pi * minimum_frequency_hz) ** (-8.0 / 3.0)
        * (solar_mass_time_seconds * chirp_mass) ** (-5.0 / 3.0)
    )


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
        self.construction_duration = float(config["construction_duration_seconds"])
        self.construction_sample_count = int(config["construction_sample_count"])
        self.construction_merger_offset = float(
            config["construction_merger_offset_seconds"]
        )
        self.crop_start = int(config["crop_start_sample"])
        self.crop_stop = int(config["crop_stop_sample_exclusive"])
        self.zero_guard_samples = int(config["zero_guard_samples_per_edge"])
        self.transition_samples = int(
            config["raised_cosine_transition_samples_per_edge"]
        )
        if self.sample_count != int(self.sample_rate * self.duration):
            raise ValueError("GW sample count is inconsistent with duration and rate")
        if not 0 < self.merger_offset < self.duration:
            raise ValueError("merger offset must lie inside the segment")
        if config.get("inverse_transform") != "bilby.core.utils.infft":
            raise ValueError("RC.5 requires Bilby's normalized inverse transform")
        if config.get("selection_snr_signal") != "conditioned_published_clean_strain":
            raise ValueError("selection SNR must use conditioned published clean strain")
        if self.construction_sample_count != int(
            self.sample_rate * self.construction_duration
        ):
            raise ValueError("construction sample count is inconsistent")
        if not 0 < self.construction_merger_offset < self.construction_duration:
            raise ValueError("construction merger offset is invalid")
        if self.crop_stop - self.crop_start != self.sample_count:
            raise ValueError("construction crop does not produce the published sample count")
        if self.crop_stop > self.construction_sample_count:
            raise ValueError("construction crop exceeds its internal array")
        expected_merger_sample = self.crop_start + int(
            self.merger_offset * self.sample_rate
        )
        construction_merger_sample = int(
            self.construction_merger_offset * self.sample_rate
        )
        if expected_merger_sample != construction_merger_sample:
            raise ValueError("published and construction merger placement disagree")
        self.conditioning_window = raised_cosine_guard_window(
            self.sample_count,
            self.zero_guard_samples,
            self.transition_samples,
        )
        self._bilby = importlib.import_module("bilby")
        bilby_file = getattr(self._bilby, "__file__", None)
        if bilby_file is None:
            raise RuntimeError("Bilby package path is unavailable")
        curve_root = (
            Path(str(bilby_file)).resolve().parent / "gw" / "detector" / "noise_curves"
        )
        self._psd_by_detector: Dict[str, Any] = {}
        for detector in DETECTOR_SLOTS:
            specification = config["psd_curves"][detector]
            curve_path = curve_root / str(specification["file"])
            digest = hashlib.sha256(curve_path.read_bytes()).hexdigest()
            if digest != str(specification["sha256"]):
                raise ValueError(f"PSD hash mismatch for {detector}: {curve_path}")
            keyword = psd_file_keyword(curve_path.name)
            self._psd_by_detector[detector] = self._bilby.gw.detector.PowerSpectralDensity(
                **{keyword: str(curve_path)}
            )
        self._waveform_generator = self._bilby.gw.WaveformGenerator(
            duration=self.construction_duration,
            sampling_frequency=self.sample_rate,
            frequency_domain_source_model=self._bilby.gw.source.lal_binary_black_hole,
            waveform_arguments={
                "waveform_approximant": str(config["waveform"]),
                "reference_frequency": float(config["reference_frequency_hz"]),
                "minimum_frequency": float(config["minimum_frequency_hz"]),
            },
        )

    def _interferometer(self, detector: str) -> Any:
        interferometer = self._bilby.gw.detector.get_empty_interferometer(detector)
        interferometer.power_spectral_density = self._psd_by_detector[detector]
        return interferometer

    def _conditioned_selection_snr(
        self,
        detector: str,
        clean_time: np.ndarray,
        segment_start: float,
    ) -> float:
        interferometer = self._interferometer(detector)
        interferometer.set_strain_data_from_zero_noise(
            self.sample_rate,
            self.duration,
            start_time=segment_start,
        )
        frequency_domain, frequencies = self._bilby.core.utils.nfft(
            clean_time,
            self.sample_rate,
        )
        if not np.array_equal(frequencies, interferometer.frequency_array):
            raise ValueError("conditioned clean-signal frequency grid mismatch")
        snr_squared = complex(
            interferometer.optimal_snr_squared(np.asarray(frequency_domain))
        )
        value = math.sqrt(max(float(snr_squared.real), 0.0))
        if not math.isfinite(value):
            raise ValueError("conditioned clean-signal SNR is nonfinite")
        return value

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

    def source_polarizations(
        self, parameters: Mapping[str, float]
    ) -> Mapping[str, np.ndarray]:
        """Generate and, when configured, validate unlensed source polarizations."""

        polarizations = self._waveform_generator.frequency_domain_strain(parameters)
        numerical_validity = self.config.get("source_polarization_numerical_validity")
        if numerical_validity is not None:
            if numerical_validity.get("enabled") is not True:
                raise ValueError("source-polarization numerical validity must be enabled")
            validate_source_polarization_spectrum(
                polarizations,
                np.asarray(self._waveform_generator.frequency_array),
                minimum_frequency_hz=float(
                    numerical_validity["minimum_frequency_hz"]
                ),
                positive_amplitude_quantile=float(
                    numerical_validity["positive_amplitude_quantile"]
                ),
                maximum_peak_to_quantile_ratio=float(
                    numerical_validity["maximum_peak_to_quantile_ratio"]
                ),
            )
        return polarizations

    def project_images(
        self,
        draw: PopulationDraw,
        images: Sequence[PhysicalImage],
        base_geocent_time: float,
    ) -> Tuple[ImageProjection, ...]:
        parameters = self.source_parameters(draw)
        polarizations = self.source_polarizations(parameters)
        projections = []
        for image in images:
            if image.arrival_time_seconds is None:
                raise ValueError("production images require physical arrival times")
            geocent_time = base_geocent_time + image.arrival_time_seconds
            segment_start = geocent_time - self.merger_offset
            construction_start = geocent_time - self.construction_merger_offset
            response_parameters = {**parameters, "geocent_time": geocent_time}
            factor = math.sqrt(abs(image.mu_signed)) * self._morse_factor(image)
            lensed = {name: values * factor for name, values in polarizations.items()}
            clean = np.zeros((3, self.sample_count), dtype=np.float64)
            snrs = []
            for detector_index, detector in enumerate(DETECTOR_SLOTS):
                interferometer = self._interferometer(detector)
                interferometer.set_strain_data_from_zero_noise(
                    self.sample_rate,
                    self.construction_duration,
                    start_time=construction_start,
                )
                frequency_domain = np.asarray(
                    interferometer.get_detector_response(
                        lensed,
                        response_parameters,
                        frequencies=self._waveform_generator.frequency_array,
                    )
                )
                full_time = np.asarray(
                    self._bilby.core.utils.infft(frequency_domain, self.sample_rate),
                    dtype=np.float64,
                )
                if full_time.shape != (self.construction_sample_count,):
                    raise ValueError("normalized inverse transform has wrong shape")
                conditioned = np.asarray(
                    full_time[self.crop_start : self.crop_stop]
                    * self.conditioning_window,
                    dtype=np.float32,
                ).astype(np.float64)
                if not np.all(np.isfinite(conditioned)):
                    raise ValueError("conditioned clean strain is nonfinite")
                clean[detector_index] = conditioned
                snrs.append(
                    self._conditioned_selection_snr(
                        detector,
                        conditioned,
                        segment_start,
                    )
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
                interferometer = self._interferometer(detector)
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
