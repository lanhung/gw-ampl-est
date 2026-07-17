# Phase 4 65k training-stack implementation report

## Outcome

The fail-closed software path for the terminal 65,536-system probe rung is
implemented. It resolves an atomic two-parent training reference, lazily
concatenates the immutable 32,768-system Stage A and Stage B components,
preserves the existing 6,144-system validation publication and trains the
unchanged probe architecture from scratch for seeds 0, 1 and 2.

This is an implementation result only. Stage B is still materializing and no
65k authorization exists. The implementation did not read Stage B staging,
construct a 65k membership, start an optimizer, fit calibration or open final
evaluation.

## Publication resolver

The new resolver requires:

- one passed atomic Stage A parent;
- one passed atomic Stage B parent;
- one passed atomic 65k combined-reference manifest;
- exactly 32,768 systems in each training component;
- exactly 65,536 unique combined systems;
- unchanged 6,144-system validation;
- exact `q=p` and unit-weight assertions;
- recorded Stage A/Stage B and Stage B/validation group-disjoint checks;
- hashes for both parent manifests and the combined manifest;
- a manifest that does not self-authorize training.

The future execution gate must additionally bind and verify the immutable
training wheel file and SHA-256, the non-editable-install policy, training
commit, environment lock, model configuration and finalized evaluation
commitment. The terminal decision writer accepts only the authorization-bound
32k input, 65k input and output path and refuses to overwrite a prior decision.

The bounded-memory reader keeps shard-local I/O. It indexes shard manifests,
loads one Parquet/Zarr shard at a time and exposes only noisy strain plus
deployable EM/GW features. Clean strain, noise, density provenance, selection
statistics and truth IDs remain unavailable as model inputs.

## Frozen 65k fit semantics

The 65k path reuses without alteration:

- `configs/models/phase4_probe_nsf.yaml`;
- conditional NSF with 10 transforms and width 256;
- seeds 0, 1 and 2;
- AdamW, learning rate, epoch budget and early stopping;
- physical microbatch 64 × accumulation 4 = effective batch 256;
- exact direct-target unweighted conditional NLP;
- the identical Stage A validation systems;
- 1,024 development posterior draws per case;
- development-only coverage, CRPS, NLP, EM-cell and tail diagnostics.

Each seed starts from scratch. The existing 32k fits are comparison evidence,
not warm starts.

## Terminal learning-curve decision

The same 10,000-replicate paired physical-system bootstrap compares 32k with
65k. If every saturation condition passes, the state is `lock_train_65k`.
Otherwise the only states are:

- `stop_data_limited_and_new_preregistration` when the NLP interval establishes
  meaningful improvement; or
- `stop_inconclusive_and_new_preregistration` when the result is ambiguous.

Both non-lock states explicitly keep extension above 65,536 closed. No result
can automatically generate a larger rung.

## Engineering verification

Tests cover:

- two-parent hash and count binding;
- a positive exact-authorization path plus parent-hash tamper rejection;
- immutable wheel and environment-lock hash enforcement;
- rejection of self-authorized or tampered manifests;
- fail-closed behavior before the future 65k authorization exists;
- exact reuse of Stage A validation;
- terminal lock and data-limited decision exits;
- prohibition on automatic extension above 65k;
- deterministic three-seed launcher and shared rung preprocessing.

The final scientific execution identity, training wheel and authorization will
be resolved only after this implementation passes PR CI and Stage B publishes.
No scientific result is claimed by this report.

Local release verification completed with 262 tests passed and six optional
dependency skips. Maintained-scope Ruff, mypy over 50 source files, sdist and
wheel construction all passed. The authoritative CUDA/dependency suite remains
part of PR CI and the later pre-execution AutoDL gate.
