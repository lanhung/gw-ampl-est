# ADR-002: Production generation and storage

- Status: accepted for Phase 3A qualification
- Date: 2026-07-12
- Scope: 4,096 non-scientific qualification pairs only

## Decision

Production generation uses bounded-memory, dataset-specific staging under
`data_v2/production/staging/<dataset_id>`. It never routes through the Phase 1B
all-pair `np.stack` publication path. Exactly 32 independently valid shards hold
128 accepted pairs each. A worker owns one partial shard and one process-local
deterministic RNG context; Bilby's global generator is never shared by threads.

The layout is:

```text
staging/<dataset_id>/
  run_manifest.json
  attempts/attempts-*.jsonl
  shards/shard-00000.partial/
  shards/shard-00000/
    noisy.zarr/
    clean.zarr/
    noise.zarr/
    records.parquet
    shard_manifest.json
    COMPLETE.json
  validation/
  environment/
```

Each Zarr v2 product has logical shape `(128, 2, 3, 16384)` and chunks
`(1, 2, 3, 16384)`. Metadata are Parquet-partitioned by shard; the canonical
typed record remains JSON in a column. Attempts are append-only JSONL partitions
with deterministic attempt IDs, proposal seeds, disposition, rejection reason,
and accepted pair ID where applicable.

## Atomic states and resume

A shard is written only in `shard-N.partial`. All arrays and tables are closed,
validated and SHA-256 inventoried before `COMPLETE.json` is written. The partial
directory is then renamed on the same filesystem to `shard-N`. Existing complete
shards are immutable: resume verifies their manifest and tree hash and never
rewrites them. A partial shard is validated against its checkpoint; inconsistent
or corrupt partial state is retained as failure evidence and publication stops.

After all shards pass, dataset validation checks counts, IDs, groups, policy,
array decomposition and cross-shard duplicates. Only then is the final dataset
manifest created and the complete dataset root atomically renamed into the
publication root. A partial dataset or shard is never discoverable as complete.

## Integrity and recovery

Every ordinary artifact has a byte count and SHA-256. Directory tree hashes bind
relative paths, file hashes and sizes. The run manifest records code/config/
authorization identities, expected artifacts and state transitions. Corruption
of a complete shard is a hard failure; it is not silently regenerated under the
same dataset ID. Recovery requires retaining evidence and beginning a reviewed
new run identity if byte identity cannot be proven.

## Memory and disk bounds

At most one generated pair plus one 128-pair shard writer buffer may be resident
per process. No operation loads all 4,096 records or arrays. Dataset validation
streams shard metadata and arrays. The qualification has a 10 GB output hard
cap, a 105,807,608,883-byte prelaunch free-space gate, and a 100 GB post-run
floor. Failed staging is retained and included in peak disk accounting.

## Non-scientific boundary

Every manifest states `dataset_purpose: generator_qualification` and sets
scientific, training, calibration and test authorization false. Phase 3A cannot
authorize full production or model training.
