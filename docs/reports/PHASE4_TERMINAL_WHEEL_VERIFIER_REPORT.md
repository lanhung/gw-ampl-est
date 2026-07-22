# Phase 4 terminal exact-wheel verifier report

## Outcome

A fail-closed verifier now produces the exact AutoDL wheel-test artifact
required by the terminal probe release-review packet. It is implementation
evidence only. It does not read a scientific shard, resolve terminal
membership, create an authorization or start an optimizer.

## Contract

The command requires an explicit runtime Python, wheel and output path. Before
running tests it verifies:

- the wheel exists and its SHA-256 is computed by streaming reads;
- `gwlens_mm` is installed under `site-packages` or `dist-packages`;
- the imported module is not below the repository `src` tree;
- the installation is not editable;
- the PEP 610 archive SHA-256 equals the requested wheel SHA-256;
- Torch reports CUDA available;
- at least three GPUs exist and every name is exactly
  `NVIDIA RTX 5000 Ada Generation`.

Focused and full pytest commands use the runtime interpreter with
`-c /dev/null --noconftest`. The second flag is essential:
`tests/conftest.py` inserts the repository `src` directory independently of
pytest configuration and would otherwise make the test subprocess import the
checkout rather than the installed wheel. `PYTHONPATH` contains the repository
root only so tests may import maintained `scripts/`; the repository `src`
directory is never added. Separate logs and their SHA-256 values are recorded
beside an atomically renamed JSON result.

The terminal release packet was hardened to require these import-provenance
fields in addition to zero focused/full exit codes.

## Verification

- verifier/release focused tests: 13 passed;
- full local suite: 412 passed, 7 optional dependency skips;
- Ruff: passed after one mechanical import-order correction;
- mypy over `src` and all Phase 4 scripts: passed.

The exact post-publication wheel has not yet been built or tested on AutoDL.
That future execution must occur only after the active immutable terminal
materialization finishes and a safe disposable runtime is available.

Before publication, an isolated runtime smoke installation exposed and fixed
two release-evidence details without opening data: the wheel is installed from
a `file:` URL carrying its frozen SHA-256 fragment so PEP 610 records the
archive hash, and the future pytest subprocess is forced to ignore repository
conftest files. A regression test creates a deliberately failing conftest and
proves that the generated command does not load it. The full exact-wheel test
remains post-publication only.

## Safety boundary

This work authorizes no terminal optimizer, architecture fit, calibration,
SBC, final evaluation, extension beyond 131,072, real noise or GWOSC/GWTC
access. The active 32-worker materialization was not restarted, synchronized
or inspected through a scientific data reader.
