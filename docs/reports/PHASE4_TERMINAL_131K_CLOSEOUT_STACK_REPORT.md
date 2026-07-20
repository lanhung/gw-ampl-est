# Phase 4 terminal 131k independent-closeout stack report

## Outcome

The terminal materialization runner already generates the exact train
increment, four development-tail namespaces and logical 131k manifest in one
continuous fail-closed execution. A separate read-only closeout command now
validates its result and publications before any later training review.

The closeout command independently binds the generator, configuration,
worker-32 orchestration, official identities, exact counts, direct-target unit
weights and all closed downstream flags. It resolves the corrected-65k subset,
new train parent, four tail datasets and combined manifest through the typed
terminal publication reader. By default it recomputes both large publication
tree hashes and byte counts and repeats the 100 GB free-space gate.

The command creates only one small JSON result. It cannot generate a pair,
open a checkpoint, start training, select an architecture, calibrate, unseal
final data or access GWOSC/GWTC.

## Usage after atomic publication

```bash
python scripts/phase4/closeout_terminal_131k.py \
  --execution-result /root/autodl-tmp/lensing-4/manifests/phase4/terminal_131k/execution_result_worker32.json \
  --output /root/autodl-tmp/lensing-4/manifests/phase4/terminal_131k/independent_closeout_worker32.json
```

The default full-tree recomputation is required for the official closeout.
The skip option exists only for bounded synthetic tests and diagnostic dry
runs; it cannot support publication acceptance.

## Verification

- terminal execution/closeout focused tests: 17 passed;
- full local suite: 399 passed, seven optional dependency skips;
- maintained-scope Ruff: passed;
- mypy: passed for 65 source files;
- package build and script compilation: passed.

## Boundary

Implementation and synthetic tests do not authorize the terminal optimizer or
any downstream scientific execution. Exact probe authorization must bind the
completed publication hashes, an AutoDL-verified immutable training wheel,
model configuration, final-evaluation commitment and CUDA environment.
