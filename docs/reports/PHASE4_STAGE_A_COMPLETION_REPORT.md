# Phase 4 Stage A direct-target publication report

Status: passed and atomically published; scientific model training remains
separately gated.

## Outcome

The frozen RC.4 Stage A run completed at `2026-07-15T17:45:15Z`. It generated,
validated and atomically published exactly 32,768 train systems and 6,144
validation systems. The common parent contains 304 complete shards of 128
accepted systems, has no partial shard, and replaced the complete staging parent
through one same-filesystem atomic rename.

The immutable execution identities are:

- parent run: `phase4-stage-a-2be777e727ef-d3a60034bbd6`;
- generator commit: `2be777e727ef9d8e1a85f89c68966df5d37932b0`;
- RC.4 hash: `5aeaac395463bd073c44ead4ff4c5c729b5a2d4b4f1840c0825a53b30ab1bc98`;
- train dataset: `gwlens-v2-2.0.0-alpha.3-ef4baee395dbe94b-train`;
- validation dataset: `gwlens-v2-2.0.0-alpha.3-33c3e6cb507a03e9-validation`;
- parent manifest SHA-256:
  `4f3e6b3a7ca1a995d7a7643c48410e479fb812e4a01ff66537232b9d64bf3314`;
- publication-tree SHA-256:
  `1c9d95d0e0157e4123ecb27fe31114aae15cb257c34063aca1b3677a7f1e2621`.

## Count and distribution validation

| Split | Accepted | Attempts | Shards | Acceptance |
|---|---:|---:|---:|---:|
| train | 32,768 | 12,234,366 | 256 | 0.26784% |
| validation | 6,144 | 2,320,671 | 48 | 0.26475% |
| total | 38,912 | 14,555,037 | 304 | 0.26734% |

Every train EM-availability cell contains exactly 4,096 systems and every
validation cell exactly 768. Train contains 16,824 SIE+shear and 15,944
EPL+shear systems; validation contains 3,159 SIE+shear and 2,985 EPL+shear
systems. Train and validation grouped physical-system identities are disjoint.

## Scientific contract validation

The parent validator passed the direct-target contract:

- `q_train = p_eval`;
- proposal and evaluation log probabilities agree;
- every importance weight is exactly one;
- all 304 shards have the frozen alpha.3 schema and artifact hashes;
- the frozen train and validation seeds, configuration hashes and generator
  commit agree with the release certificate;
- no Phase 3 engineering ID enters either scientific split;
- no calibration, SBC or final-evaluation system was materialized;
- no GWOSC or GWTC product was accessed.

This publication authorizes use only under a later exact training gate. It is
not itself a model result, calibration result or performance claim.

## Runtime and storage

The run started at `2026-07-14T02:07:18Z` and wrote its passed execution result
after 39.63 elapsed hours. This was 29.8% faster than the conservative 56.48-hour
planning projection. Actual attempts were 5.25% above the planning estimate.

Published artifact bytes were 46,491,822,064, 9.96% above the 42,281,598,311
byte publication estimate but below the 64,534,762,432 byte projected peak
resource envelope. The runner recorded 264,872,910,848 free bytes after
publication, exceeding the 100 GB post-run floor by 164.87 GB.

## Evidence

Committed small evidence consists of:

- `results/phase4/stage_a_dataset_manifest.json` and its SHA-256;
- `results/phase4/stage_a_execution_result.json` and its SHA-256;
- `results/phase4/stage_a_publication_summary.json`;
- this report.

The 46.49 GB Zarr/Parquet publication remains only on AutoDL under the new
project root. No raw data enters Git.

Local closeout verification passed 246 tests with six documented optional
dependency skips, maintained-scope Ruff, mypy over 49 source files, and sdist
and wheel builds. The literal repository-wide Ruff command continues to report
the same 18 pre-existing Phase 0 audit-script findings documented in Phase 3A;
no Stage A or maintained-scope file introduced a new Ruff finding.

## Next gate

Stage A materialization is complete. Before any optimizer may start, a separate
probe-training authorization must bind:

- the parent manifest SHA-256 above;
- finalized evaluation commitment SHA-256
  `c13412eced163bac26abc4b22d054f3a6fa967e7e5a4dd7849ebf54f42df6083`;
- training-code commit
  `71fe4fd2563ef02e5c06cf6d907e1fb39e4d38e2`;
- training wheel SHA-256
  `09313797a17abe2e1850324a20ec70188f310db2777253f19bf72f73e28049dd`;
- model configuration hash
  `4930651be569725748a0025311ea8b479a217ecab8137be9855a7ee6d2c0377c`;
- normalized CUDA environment SHA-256
  `2e45000a8cea6712ae307c87782c593245ad56607a772f27a0cc5af726e37b95`.

That future gate may open only the 16,384/32,768 probe rungs and seeds 0, 1
and 2. Calibration, SBC, final evaluation, Stage B and GWOSC/GWTC remain
closed.
