# Phase 4 terminal release-evidence portability report

## Outcome

The post-publication terminal-probe release chain now uses committed,
repository-relative evidence paths. This removes a pre-execution failure mode
in which a packet created under the AutoDL checkout embedded a path that could
not be resolved under the authoritative Vultr checkout, or vice versa.

## Contract

- closeout, release packet and delegated review remain below
  `results/phase4/`;
- their stored paths are relative to the explicit repository root;
- absolute paths and parent traversal fail closed;
- every consumed file remains SHA-256 bound;
- the immutable wheel remains an absolute AutoDL artifact path;
- the release packet remains non-authorizing;
- the second delegated-review object remains mandatory.

No active terminal staging path, scientific publication, checkpoint or final
evaluation case was opened. The running 32-worker materialization was not
modified.

## Verification

- portable-path focused tests: 27 passed;
- full suite: 429 passed, 7 optional dependency skips;
- Ruff: passed;
- mypy: passed for 67 source files.

The locally built pre-correction wheel with SHA-256
`e65b08009640cc3e6643db84cdd615bdfb533f647b1f436eaf2c741082b9951f`
is superseded software evidence only. It was never copied to AutoDL, installed,
reviewed or authorized. The exact training wheel will be rebuilt from the
merged portability commit after atomic publication.

## Evidence-only descendant handoff

The independent closeout cannot be committed before its publication exists.
Packet assembly therefore permits a clean descendant of the frozen training
commit, but only when every intervening path is one of the exact registered
closeout/project-state evidence paths. A source, model, configuration or
unregistered-file change fails closed. The packet records the evidence-review
checkout commit separately from the immutable training commit.

Two temporary Git-repository regression tests exercise the actual script
path: one accepts a committed closeout-only descendant and one rejects a
descendant that changes the frozen model source. Together with packet,
authorization and runtime-binding tests, the focused total is 30 passed.
The complete repository suite passes 432 tests with seven optional dependency
skips; Ruff and mypy also pass.
