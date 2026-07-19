# Phase 4 corrected probe-rerun implementation

## Outcome

The training stack can now resolve the immutable waveform-correction parent as
an exact overlay over the original Stage A and Stage B publications. This is a
software release only. It does not authorize data access or an optimizer.

## Corrected publication contract

The resolver binds all of the following before constructing a dataset:

- the original Stage A, Stage B and combined-65k parent manifests;
- correction preregistration `1.1.1-rc.1`;
- correction parent manifest and complete publication-tree hashes;
- exactly two Stage A and three Stage B exclusions;
- exactly two plus three replacement IDs;
- exact corrected counts of 32,768/32,768/65,536 train and 6,144 validation;
- direct-target equality, exact unit weights and closed downstream flags;
- replacement/base and train/validation physical-system disjointness.

The resolver indexes complete-shard manifests only. It does not open Parquet or
Zarr during the gate. The published base components remain immutable. Training
datasets lazily concatenate a filtered base component with its replacement
component; the replacement records are never copied into or used to rewrite a
base shard.

## Rung semantics

The complete corrected Stage A membership is formed before the existing
SHA-256 rank rule resolves the 16,384-system subset. The 16k subset therefore
may include replacement IDs and cannot accidentally retain an excluded system.
The 32k rung contains every corrected Stage A system. The terminal 65k reader
uses corrected Stage A plus corrected Stage B and the unchanged validation
publication.

All run identities bind a derived corrected training-view manifest hash rather
than the historical base namespace hash. The original failed 65k output and
superseded 16k/32k checkpoints cannot be resumed. A future corrected 65k gate
uses the same implementation and only becomes executable if the fresh 16k/32k
decision again requires continuation.

## Verification

- corrected-view fixture tests cover exact exclusion/replacement arithmetic,
  subset selection, 32k/65k uniqueness and tree/manifest drift rejection;
- existing Stage A and 65k contracts remain backward compatible;
- local full suite: 324 passed with seven optional dependency skips;
- focused corrected/training/rung tests: 29 passed with one optional skip;
- maintained-source Ruff passed;
- mypy passed for 58 source files;
- source/script compilation and package build passed.

The real AutoDL metadata-only resolver then reproduced the immutable correction
manifest and tree hashes and derived:

- corrected Stage A 32k training-view SHA-256:
  `b9390a7faad4bb8097abb09041f1f229e13c6419677926f05642ac611dc6ced2`;
- corrected combined 65k training-view SHA-256:
  `da8aaa8d86afb4d93156191976b420bfc7bbc7dfe0fdc6c6f627515d804a7379`.

This resolution used all real published shard manifests while explicitly
recording that Parquet, Zarr, the optimizer, calibration and final evaluation
were not opened.

No Stage A/B Parquet or Zarr file was opened by this implementation work. No
membership, standardizer, checkpoint, metric or scientific decision was
created. The next step is a metadata-only AutoDL resolution followed by an
exact training release that binds the derived view hash, wheel and CUDA
environment.

## Boundaries

Model tuning, architecture selection, calibration, SBC, final evaluation,
extension above 65,536, real noise and GWOSC/GWTC remain closed.
