# Phase 4 final-evaluation generator implementation report

## Outcome

The sealed final-evaluation generator implementation is complete and frozen at
`bc02054c1f95e7f6cd143fb9dc796ae48f0a15ac`. It implements all six RC.3/RC.4
final-pool splits, a fail-closed future materialization runner, streaming
validators and atomic sealed publication.

No final-evaluation pair was generated. No official final-evaluation identity
was created. Materialization, unsealing, model training, calibration, scientific
analysis and GWOSC/GWTC access remain disabled.

## Frozen pool arithmetic

| Split | Accepted systems | Shards | Contexts |
|---|---:|---:|---:|
| IID test | 8,192 | 64 | 1 |
| balanced-tail diagnostic | 4,096 | 32 | 4 |
| cross-family misspecification | 2,048 | 16 | 4 |
| parameter-region OOD | 2,048 | 16 | 4 |
| waveform mismatch | 2,048 | 16 | 1 |
| PSD mismatch | 2,048 | 16 | 1 |
| **Total** | **20,480** | **160** | **15** |

Every shard contains 128 accepted physical systems. Namespace seeds are
derived independently from root seed `2026071203`, the inherited split seed
domain and the diagnostic context. All 15 resulting roots are distinct.
Because balanced-tail membership is evaluated only after the complete lens,
waveform and selection path, the future fail-closed ceiling is conservatively
20,000,000 attempts per shard rather than the ordinary Stage A ceiling. This
does not alter the target or tail definition.

## Implemented scientific contexts

- IID uses exact direct draws from the frozen evaluation target.
- Balanced-tail cases use first-matching priority: maximum selected
  `abs(mu) >= 20`; otherwise selected min/max `<= 0.10`; otherwise secondary
  network SNR in `[10, 12]`; otherwise `abs(kappa_ext) >= 0.10` or EPL slope
  outside `[1.75, 2.40]`. Nonmatching accepted systems remain rejected attempts.
- Cross-family cells freeze SIE/EPL truth and the separately recorded assumed
  family. Family-marginalized cells remain distinct contexts.
- Parameter OOD uses the parent contract exactly: EPL slope `[1.4,1.6)` or
  `(2.5,2.7]`, axis ratio `[0.25,0.40)`, shear `(0.15,0.25]`, and
  `abs(kappa_ext)` in `(0.15,0.25]` with balanced sign.
- Waveform mismatch replaces truth generation only with `SEOBNRv4PHM`; the
  trained-model assumption remains `IMRPhenomXPHM`.
- PSD mismatch uses the frozen Bilby H1/L1 ZeroDetHighPower PSD and Virgo AdV
  PSD files with exact SHA-256 verification. ASD and PSD files use Bilby's
  explicit constructor semantics rather than filename-agnostic loading.

OOD and balanced-tail distributions are diagnostic distributions. Stored log
densities describe their declared pre-selection proposal/evaluation measures;
the balanced-tail accepted set additionally conditions through the recorded
priority rejection rule. These data are not training, calibration or stopping
inputs.

## Storage and validation

The future runner is implemented but cannot run under the present gate. It
requires a finalized commitment, an exact future generator commit, a separate
sealed-materialization authorization and a ready release certificate with 15
distinct dataset identities.

When separately authorized, it will:

1. generate each namespace with bounded-memory atomic 128-system shards;
2. stream every Parquet record and noisy/clean/noise Zarr array;
3. validate split/context identities, exact q=p unit weights, mismatch truth,
   tail/OOD support, generator/config identity, attempts and checksums;
4. reject duplicate pair/source/lens/system/noise IDs within or across all
   namespaces;
5. stream all release-certificate-bound published train, validation,
   calibration-fit and SBC references and reject cross-pool group leakage;
6. validate exact `noisy = clean + noise` array semantics;
7. create a sealed parent manifest only after all 160 shards pass;
8. enforce a 30 GB publication cap and 100 GB post-run free-space gate before
   a same-filesystem atomic rename into the sealed publication root.

Linear scaling from the Phase 3A publication gives 22,253,472,795 bytes for
20,480 systems. The 30 GB hard publication cap and 145 GB prelaunch free-space
gate therefore preserve more than 100 GB after the projected sealed output.

Unsealing and analysis remain separate future gates. The runner cannot derive
official identities on its own.

## Verification

- focused final-evaluation tests: 16 passed;
- full local suite: 241 passed with five optional dependency skips;
- maintained-scope Ruff: passed;
- mypy, including both Phase 4 final-evaluation scripts: passed for 49 source
  files;
- dry plan: 20,480 systems, 160 shards, 15 namespaces, zero generated pairs;
- execution attempt through the preparation command: fail-closed;
- configuration hash:
  `11277a2a4c5d233e6f525b3ab5d6ece90c115d818d752849076f3a136e574d66`.

The finalized generation commitment SHA-256 is
`c13412eced163bac26abc4b22d054f3a6fa967e7e5a4dd7849ebf54f42df6083`.
It records the exact generator commit, all 15 derived namespace seeds, the
512-stride attempt formula, shard-major accepted-rank rule, counts,
distribution/version identities, cross-pool grouping rules and sealed-use
policy. Accepted IDs remain correctly unknown until future materialization.

## Independent Stage A state

At `2026-07-14T13:15:29Z`, the immutable Stage A run had 82 of 256 train
shards complete (10,496 train systems), 16 train partial shards, zero validation
shards and zero error artifacts. The parent remains staging-only. This
implementation did not read any Stage A record or alter the remote checkout.

## Remaining gates

1. pass PR CI and merge;
2. finish, validate and atomically publish Stage A;
3. obtain a separate probe-training authorization before any scientific fit.

Final-evaluation materialization remains post-lock by default and is not on the
current Stage A publication critical path.
