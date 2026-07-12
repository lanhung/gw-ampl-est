#!/usr/bin/env python3
"""Quality-control reporting for smoke generation outputs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "smoke_data"
FIG_DIR = BASE_DIR / "figures" / "smoke_qc"
CHECK_DIR = BASE_DIR / "checks"
SUMMARY_PATH = CHECK_DIR / "smoke_qc_summary.csv"

DATASETS = {
    "ET_PM": {"directory": "ET_PM_data", "prefix": "PM", "ligo": False, "lensed": True},
    "ET_SIS": {"directory": "ET_SIS_data", "prefix": "SIS", "ligo": False, "lensed": True},
    "ET_unlensed": {
        "directory": "ET_unlensed_data",
        "prefix": "unlensed",
        "ligo": False,
        "lensed": False,
    },
    "LIGO_PM": {"directory": "LIGO_PM_data", "prefix": "PM", "ligo": True, "lensed": True},
    "LIGO_SIS": {"directory": "LIGO_SIS_data", "prefix": "SIS", "ligo": True, "lensed": True},
    "LIGO_unlensed": {
        "directory": "LIGO_unlensed_data",
        "prefix": "unlensed",
        "ligo": True,
        "lensed": False,
    },
}


def expected_files(info):
    prefix = info["prefix"]
    files = ["source_samples.csv"]
    if info["lensed"]:
        files += [
            "lensed_index.csv",
            "lens_params.csv",
            "lens.csv",
            "lensed_source_samples.csv",
            "test_source_samples_1.csv",
            "test_source_samples_2.csv",
        ]
        output_prefixes = [
            f"{prefix}_1",
            f"{prefix}_2",
            f"test_{prefix}_1",
            f"test_{prefix}_2",
        ]
    else:
        output_prefixes = [prefix]

    for value in output_prefixes:
        if value.endswith("_1") or value.endswith("_2"):
            root, image = value.rsplit("_", 1)
            files += [
                f"{root}_data_strain_{image}.npy",
                f"{root}_h_strain_{image}.npy",
                f"{root}_time_array_{image}.npy",
            ]
            snr_root = f"{root}_optimal_SNR"
            snr_suffix = f"_{image}.npy"
        else:
            files += [
                f"{value}_data_strain.npy",
                f"{value}_h_strain.npy",
                f"{value}_time_array.npy",
            ]
            snr_root = f"{value}_optimal_SNR"
            snr_suffix = ".npy"
        if info["ligo"]:
            files += [f"{snr_root}_single{snr_suffix}", f"{snr_root}_network{snr_suffix}"]
        else:
            files.append(f"{snr_root}{snr_suffix}")
    return files


def display_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def print_array_report(path):
    array = np.load(path, mmap_mode="r")
    has_nan = bool(np.isnan(array).any())
    has_inf = bool(np.isinf(array).any())
    if array.size:
        min_value = display_value(np.nanmin(array))
        max_value = display_value(np.nanmax(array))
    else:
        min_value = "EMPTY"
        max_value = "EMPTY"
    print(
        f"  NPY {path.name}: shape={array.shape}, dtype={array.dtype}, "
        f"min={min_value}, max={max_value}, nan={has_nan}, inf={has_inf}"
    )
    return not has_nan and not has_inf


def print_snr_reports(directory):
    snr_paths = sorted(directory.glob("*optimal_SNR*.npy"))
    for path in snr_paths:
        values = np.asarray(np.load(path, mmap_mode="r")).reshape(-1)
        if values.size:
            print(
                f"  SNR {path.name}: min={np.min(values):.6g}, "
                f"median={np.median(values):.6g}, max={np.max(values):.6g}, "
                f"mean={np.mean(values):.6g}"
            )
    return snr_paths


def print_lens_report(lens_path):
    if not lens_path.exists():
        return None
    lens = pd.read_csv(lens_path)
    columns = {
        "mu_0": lens["mu_0"].to_numpy(),
        "mu_1": lens["mu_1"].to_numpy(),
        "abs(mu_1)": np.abs(lens["mu_1"].to_numpy()),
        "t_d": lens["t_d"].to_numpy(),
    }
    for label, values in columns.items():
        print(
            f"  LENS {label}: min={np.min(values):.6g}, "
            f"median={np.median(values):.6g}, max={np.max(values):.6g}, "
            f"mean={np.mean(values):.6g}"
        )
    return lens


def test_distances_positive(directory):
    paths = [directory / "test_source_samples_1.csv", directory / "test_source_samples_2.csv"]
    if not all(path.exists() for path in paths):
        return False
    for path in paths:
        samples = pd.read_csv(path)
        distances = pd.to_numeric(samples["luminosity_distance"], errors="coerce")
        if distances.isna().any() or not bool((distances > 0).all()):
            return False
    return True


def load_main_strains(directory):
    paths = sorted(path for path in directory.glob("*data_strain*.npy") if not path.name.startswith("test_"))
    arrays = [np.asarray(np.load(path, mmap_mode="r")) for path in paths]
    if not arrays:
        return None
    return np.concatenate(arrays, axis=0)


def plot_strains(name, directory, info):
    strains = load_main_strains(directory)
    if strains is None or strains.shape[0] == 0:
        return
    count = min(3, strains.shape[0])
    if info["ligo"] and strains.ndim == 3 and strains.shape[1] >= 2:
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for detector_index, detector in enumerate(["H1", "L1"]):
            for sample in range(count):
                axes[detector_index].plot(strains[sample, detector_index], label=f"sample {sample}")
            axes[detector_index].set_title(f"{name} whitened strain: {detector}")
            axes[detector_index].legend()
            axes[detector_index].set_ylabel("strain")
        axes[-1].set_xlabel("sample index")
    else:
        fig, ax = plt.subplots(figsize=(11, 4))
        for sample in range(count):
            values = strains[sample] if strains.ndim == 2 else strains[sample, 0]
            ax.plot(values, label=f"sample {sample}")
        ax.set_title(f"{name} whitened strain")
        ax.set_xlabel("sample index")
        ax.set_ylabel("strain")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_whitened_strain.png", dpi=150)
    plt.close(fig)


def plot_snr(name, snr_paths):
    if not snr_paths:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for path in snr_paths:
        values = np.asarray(np.load(path, mmap_mode="r")).reshape(-1)
        if values.size:
            ax.hist(values, bins=min(20, max(5, values.size)), alpha=0.45, label=path.stem)
    ax.set_title(f"{name} SNR")
    ax.set_xlabel("SNR")
    ax.set_ylabel("count")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_snr_hist.png", dpi=150)
    plt.close(fig)


def plot_lens(name, lens):
    if lens is None or lens.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lens["mu_0"], bins=10, alpha=0.55, label="mu_0")
    ax.hist(np.abs(lens["mu_1"]), bins=10, alpha=0.55, label="abs(mu_1)")
    ax.set_title(f"{name} magnification")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_magnification_hist.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lens["t_d"], bins=10)
    ax.set_title(f"{name} time delay")
    ax.set_xlabel("t_d")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}_time_delay_hist.png", dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for name, info in DATASETS.items():
        directory = DATA_DIR / info["directory"]
        print(f"\n{name}: {directory}")
        expected = expected_files(info)
        missing = [filename for filename in expected if not (directory / filename).exists()]
        npy_paths = sorted(directory.glob("*.npy")) if directory.exists() else []
        finite_results = [print_array_report(path) for path in npy_paths]
        arrays_finite = bool(finite_results) and all(finite_results)
        snr_paths = print_snr_reports(directory) if directory.exists() else []
        lens = print_lens_report(directory / "lens.csv") if info["lensed"] else None
        distances_ok = test_distances_positive(directory) if info["lensed"] else True
        print(f"  Missing expected files: {missing or 'none'}")
        if info["lensed"]:
            print(f"  Positive test luminosity distances: {distances_ok}")

        plot_strains(name, directory, info)
        plot_snr(name, snr_paths)
        if info["lensed"]:
            plot_lens(name, lens)

        ok = directory.exists() and not missing and arrays_finite and distances_ok
        summary.append(
            {
                "dataset": name,
                "output_directory": str(directory),
                "directory_exists": directory.exists(),
                "expected_file_count": len(expected),
                "present_expected_file_count": len(expected) - len(missing),
                "missing_files": ";".join(missing),
                "npy_file_count": len(npy_paths),
                "npy_no_nan_or_inf": arrays_finite,
                "test_luminosity_distance_positive": distances_ok if info["lensed"] else "NA",
                "overall_ok": ok,
            }
        )

    pd.DataFrame(summary).to_csv(SUMMARY_PATH, index=False)
    print(f"\nQC summary written to {SUMMARY_PATH}")
    print(f"QC figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
