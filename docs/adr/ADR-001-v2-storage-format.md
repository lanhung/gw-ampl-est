# ADR-001: V2 storage format

- Status: accepted for Phase 1B smoke
- Date: 2026-07-12
- Decision: Zarr v2 arrays plus Parquet metadata, using staged single-writer
  shard publication

## Context

V2 needs partial pair/image/detector reads, resumable accepted-count
generation, separate noisy/clean/noise products, checksums, and a schema that
can evolve without rewriting large tensors. AutoDL provides local filesystem
storage; raw arrays do not enter Git.

## Options

| Property | Zarr v2 + Parquet | Sharded HDF5 + Parquet | NPY shards + JSONL |
|---|---|---|---|
| Chunked partial reads | native | native within shard | only at file granularity |
| Resume | write new chunks/shards | close/reopen completed shards | simple but many files |
| Concurrent writers | unsafe to same store; safe per staged shard | unsafe to same file; safe per shard | safe per unique file |
| Schema evolution | metadata columns evolve separately | metadata separate, internal layout rigid | weak contracts |
| Checksums | per published shard/chunk manifest | per HDF5 shard | per file |
| Corruption recovery | replace affected staged/published shard | replace affected HDF5 shard | replace individual file |
| Dependency stability | pin Zarr v2; avoid v2/v3 ambiguity | mature `h5py`/HDF5 ABI | minimal dependencies |
| Cloud/object-store path | strong | weaker | weak |
| AutoDL local disk | suitable | suitable | suitable but metadata-heavy |

## Decision

Phase 1B will use Zarr v2 plus Parquet. Arrays use logical layout
`(pair, image, detector, sample)` and smoke chunks `(1, 2, 3, 4096)`. No two
workers write the same store concurrently. Workers create unique temporary
shards; an atomic single-writer publication step records a completed shard in
the manifest. Temporary or incomplete shards are ignored on resume.

Metadata tables use Parquet with the JSON record as the logical contract.
Nested physical-image information may use a separate image table keyed by
physical-system/image ID rather than forcing two image columns. A small JSON
manifest remains the authoritative inventory and checksum index.

Zarr and a Parquet engine are not installed on Vultr and were not installed in
Phase 1A. Phase 1B must pin mutually compatible versions on AutoDL and record
them. If this environment check fails, the fallback is sharded HDF5 plus
Parquet; changing format requires an ADR amendment, not an implicit switch.

## Integrity and recovery

- write into a dataset-ID-specific staging directory;
- fsync/close a shard before hashing;
- SHA-256 each published shard and metadata file;
- update the manifest atomically after validation;
- never overwrite a complete shard with different bytes;
- record rejected attempts separately from accepted rows;
- resume from completed manifest entries and deterministic IDs.

## Raw storage estimates

Assumptions: one second, 4096 samples, float32, two selected images, three
detector slots, and three products (noisy, clean, noise). This excludes Zarr
metadata, Parquet, truth tables, solver output, compression, and staging
overhead.

| Accepted pairs | Raw bytes | MiB | GiB |
|---:|---:|---:|---:|
| 48 | 14,155,776 | 13.50 | 0.0132 |
| 10,000 | 2,949,120,000 | 2,812.50 | 2.7466 |
| 100,000 | 29,491,200,000 | 28,125.00 | 27.4658 |

The 48-pair smoke should remain far below 1 GB including metadata and solver
products. A multi-GB smoke run is a stop condition. Full-duration 24-second
storage multiplies the raw strain figures by 24 and is not authorized here.
