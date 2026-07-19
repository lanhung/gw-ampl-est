# Phase 5 corrected architecture-stack report

## Outcome

The post-size-lock architecture stack now treats the immutable five-system
waveform correction as part of the locked training identity.  It cannot fall
back to the original combined 65k publication when reusing the corrected probe
or fitting any of the nine future architecture results.

This is implementation-only work under the existing Phase 5 software gate.
No architecture fit, selection, calibration, SBC or final-evaluation access
occurred.

## Frozen implementation

- source commit:
  `d7e87a84ffc69a0e7825eb448c8cfdabe4e7fd4d`;
- wheel SHA-256:
  `fcc766a43a61ffdda3e0fca83fbefff4a010c5d35d39ab27b637fb34dbf5490a`;
- architecture-grid file SHA-256:
  `0af9933d7e65954af2b5df48f6e53b030c60fa41e79bc002fded3358c4ac6ff2`.

## Fail-closed behavior

The future corrected architecture authorization must bind:

- the terminal `lock_train_65k` decision and SHA-256;
- the original Stage A, Stage B and combined-base publications;
- the correction parent manifest and publication-tree hashes;
- the corrected combined-view SHA-256;
- all three corrected 65k probe run-summary hashes;
- the source commit, exact wheel, model-grid hash and CUDA environment.

The runner resolves the correction publication before opening any training
array.  It uses the corrected combined-view hash as the train-manifest identity
and the correction-parent hash as the shared-preparation identity.  The three
reused probe summaries and all nine new fits must therefore share the same
corrected membership and standardizers.

The scripts now pass an explicit `--correction-publication` path through fit,
grid-launch and selection entry points.  A corrected authorization without
that path fails closed.

## Verification

- full local pytest: 326 passed, 7 optional dependency skips;
- architecture/correction focused pytest: 12 passed;
- maintained-scope Ruff: passed;
- mypy over 58 source files: passed;
- sdist and wheel build: passed.

Tests prove that corrected identities select the corrected train hash and
correction preparation hash, dispatch to the corrected dataset builder and
reject a corrected execution gate lacking an explicit correction publication.

## Remaining gate

The implementation does not authorize execution.  If and only if the running
terminal probe locks 65k, a separate exact authorization must bind the three
completed corrected probe fits and the terminal decision before the nine new
fits may start.
