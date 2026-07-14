"""Bounded-memory readers for atomically published Stage A shards."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..schema import ArrayProductRole, SplitName, V2Record
from .contracts import TrainingGateError
from .features import PreparedExample, prepare_example
from .whitening import ASDCurve, whiten_detector_grid


@dataclass(frozen=True)
class ShardIndexEntry:
    path: Path
    row_index: int
    physical_system_id: str


def _load_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not a mapping: {path}")
    return value


def index_complete_shards(
    dataset_root: Path,
    *,
    expected_pairs_per_shard: int,
    expected_total_pairs: Optional[int] = None,
    require_published: bool = True,
) -> Tuple[ShardIndexEntry, ...]:
    """Index manifest IDs without opening Parquet or Zarr products."""

    if require_published and "published" not in dataset_root.parts:
        raise TrainingGateError("training readers refuse staging or canary roots")
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if require_published and not (dataset_root / "dataset_manifest.json").is_file():
        raise TrainingGateError("atomic dataset publication manifest is missing")
    shards_root = dataset_root / "shards"
    complete = sorted(
        path
        for path in shards_root.glob("shard-[0-9][0-9][0-9][0-9][0-9]")
        if path.is_dir()
    )
    partial = tuple(shards_root.glob("shard-*.partial"))
    if partial:
        raise TrainingGateError("published dataset contains partial shards")
    entries: List[ShardIndexEntry] = []
    for shard in complete:
        if not (shard / "COMPLETE.json").is_file():
            raise TrainingGateError(f"complete marker missing: {shard}")
        manifest = _load_mapping(shard / "shard_manifest.json")
        if manifest.get("status") != "complete":
            raise TrainingGateError(f"shard is not complete: {shard}")
        count = int(manifest["accepted_pair_count"])
        ids = tuple(str(value) for value in manifest["physical_system_ids"])
        if count != expected_pairs_per_shard or len(ids) != count:
            raise TrainingGateError(f"shard count mismatch: {shard}")
        entries.extend(
            ShardIndexEntry(shard, row_index, physical_id)
            for row_index, physical_id in enumerate(ids)
        )
    if len({entry.physical_system_id for entry in entries}) != len(entries):
        raise TrainingGateError("published dataset contains duplicate physical-system IDs")
    if expected_total_pairs is not None and len(entries) != expected_total_pairs:
        raise TrainingGateError(
            f"published dataset contains {len(entries)}, expected {expected_total_pairs}"
        )
    return tuple(entries)


class PublishedStageADataset:
    """Open one row at a time and never expose clean/noise arrays to the model."""

    def __init__(
        self,
        dataset_root: Path,
        *,
        expected_split: SplitName,
        detector_curves: Mapping[str, ASDCurve],
        expected_pairs_per_shard: int = 128,
        expected_total_pairs: Optional[int] = None,
        selected_physical_system_ids: Optional[Sequence[str]] = None,
        require_published: bool = True,
        minimum_frequency_hz: float = 20.0,
        maximum_frequency_hz: Optional[float] = None,
    ) -> None:
        if set(detector_curves) != {"H1", "L1", "V1"}:
            raise ValueError("detector ASD curves must cover H1/L1/V1 exactly")
        self.dataset_root = dataset_root
        self.expected_split = expected_split
        self.detector_curves: Tuple[ASDCurve, ASDCurve, ASDCurve] = (
            detector_curves["H1"],
            detector_curves["L1"],
            detector_curves["V1"],
        )
        self.minimum_frequency_hz = minimum_frequency_hz
        self.maximum_frequency_hz = maximum_frequency_hz
        entries = index_complete_shards(
            dataset_root,
            expected_pairs_per_shard=expected_pairs_per_shard,
            expected_total_pairs=expected_total_pairs,
            require_published=require_published,
        )
        if selected_physical_system_ids is not None:
            selected = frozenset(selected_physical_system_ids)
            if len(selected) != len(tuple(selected_physical_system_ids)):
                raise ValueError("selected physical-system IDs contain duplicates")
            entries = tuple(entry for entry in entries if entry.physical_system_id in selected)
            if len(entries) != len(selected):
                raise TrainingGateError("selected IDs are not all present in publication")
        self.entries = entries
        self._cached_path: Optional[Path] = None
        self._cached_records: Any = None
        self._cached_noisy: Any = None

    def __len__(self) -> int:
        return len(self.entries)

    def _open_shard(self, path: Path) -> Tuple[Any, Any]:
        if self._cached_path != path:
            pandas = importlib.import_module("pandas")
            zarr = importlib.import_module("zarr")
            records = pandas.read_parquet(path / "records.parquet")
            noisy = zarr.open_array(str(path / "noisy.zarr"), mode="r")
            if len(records) != noisy.shape[0]:
                raise TrainingGateError("Parquet/Zarr row counts disagree")
            self._cached_path = path
            self._cached_records = records
            self._cached_noisy = noisy
        return self._cached_records, self._cached_noisy

    def __getitem__(self, index: int) -> PreparedExample:
        entry = self.entries[index]
        records, noisy_store = self._open_shard(entry.path)
        row = records.iloc[entry.row_index]
        record = V2Record.from_json(str(row["record_json"]))
        record.validate()
        if record.pair.physical_system_id != entry.physical_system_id:
            raise TrainingGateError("Parquet row order disagrees with shard manifest")
        if record.pair.split is not self.expected_split:
            raise TrainingGateError("record belongs to the wrong split")
        distribution = record.provenance.distribution
        if (
            distribution.proposal_log_probability
            != distribution.evaluation_prior_log_probability
            or distribution.importance_weight != 1.0
            or not distribution.weight_valid
            or distribution.clipping_applied
        ):
            raise TrainingGateError("Stage A reader requires exact direct-target unit weights")
        roles = {reference.product_role for reference in record.gw_observation.array_products}
        if roles != set(ArrayProductRole):
            raise TrainingGateError("record array-product roles are incomplete")
        noisy = np.asarray(noisy_store[entry.row_index], dtype=np.float32)
        detector_mask = np.asarray(
            record.gw_observation.detector_availability_mask, dtype=bool
        )
        whitened = whiten_detector_grid(
            noisy,
            detector_mask,
            sampling_frequency_hz=record.gw_observation.sample_rate_hz,
            detector_curves=self.detector_curves,
            minimum_frequency_hz=self.minimum_frequency_hz,
            maximum_frequency_hz=self.maximum_frequency_hz,
        )
        return prepare_example(record, whitened)

    def physical_system_ids(self) -> Tuple[str, ...]:
        return tuple(entry.physical_system_id for entry in self.entries)


def torch_collate(examples: Sequence[PreparedExample]) -> Mapping[str, Any]:
    """Convert a prepared batch after optional PyTorch is installed."""

    from .features import collate_numpy

    torch = importlib.import_module("torch")
    arrays = collate_numpy(examples)
    return {name: torch.from_numpy(value) for name, value in arrays.items()}
