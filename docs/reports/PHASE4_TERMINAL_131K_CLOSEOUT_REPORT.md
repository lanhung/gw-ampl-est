# Phase 4 terminal 131k publication closeout

## Outcome

The terminal direct-target materialization and independent closeout passed.
The project now has an atomic logical reference containing exactly 131,072
unique train systems and the unchanged 6,144-system validation publication.
The separately published development-tail pool contains exactly 512
development-only cases.

This result does not authorize an optimizer, architecture selection,
calibration, SBC, final evaluation, an extension above 131,072 systems, real
noise or GWOSC/GWTC access.

## Frozen identities

- preregistration: `1.2.0-rc.1`;
- preregistration hash:
  `77ff5b6b45b886657e90023c50ae002afffb077db594c80665166d537fd2346a`;
- data configuration hash:
  `abd07c5b8031a5cc9564531d29d9349f65b0918fafc494767fc912b7e7444ed7`;
- generator commit: `a4e6bac014ccd521d510c97593cb1368e826d5eb`;
- orchestration commit: `adb4c0981fd15a809005212c76dd972a59822489`;
- identity payload SHA-256:
  `30fa02d9ec5b6ec1a14b3a7fddc718151c0ced2fca5d3cca7c0a66b6ba9ea6c7`.

## Published artifacts

The immutable train increment is:

- parent: `phase4-terminal-131k-a4e6bac014cc-abd07c5b8031`;
- dataset:
  `gwlens-v2-2.0.0-alpha.3-e592848e725db2c3-train-increment`;
- accepted systems: 65,536;
- complete shards: 512;
- parent manifest SHA-256:
  `0f2bb99c851b8079672b42350223cdcfe838a7f5214bf69d9de6f851b9e1c534`;
- independently recomputed tree SHA-256:
  `0ab144922ec34d08db1496af517ecc986fba00e5c95c3c3a554348e93bd6a8ed`;
- bytes: 79,373,465,020.

The development-tail publication is:

- parent: `phase4-terminal-tail-micro128-adb4c0981fd1-30fa02d9ec5b`;
- accepted cases: 512;
- namespaces: 4;
- accepted cases per namespace: 128;
- shards: 512;
- accepted cases per shard: 1;
- parent manifest SHA-256:
  `58fcafd58cbcd407ecf6b35dfa98c0bd2bd66f37151e19e6bf530ca2601260c7`;
- independently recomputed tree SHA-256:
  `90ca582f3bd768046f9ceabb4d42689d76945be2c963b0290ac432662ff619c0`;
- bytes: 8,989,124,609.

The atomic combined reference is:

- ID: `phase4-train-131k-adb4c0981fd1-30fa02d9ec5b`;
- train systems: 131,072;
- validation systems: 6,144;
- combined manifest SHA-256:
  `ad26d51d4f9475c6710cdfee4e71409526e1d776e0b8ec14734feff02855cee5`.

## Validation

The execution result has SHA-256
`8d7ee0adb5334ef6d39cdd94db029377ed290397acf146faa7f753f5865393a6`
and reports `status: passed`. A separate read-only closeout recomputed the
train and tail publication trees and reports
`status: terminal_131k_independent_closeout_passed`.

The closeout verified:

- exact 131,072 train, 6,144 validation and 512 development-tail counts;
- exactly four development-tail namespaces and 128 one-case shards per
  namespace;
- direct proposal equals the evaluation target;
- every importance weight is exactly one;
- both earlier failed tail parents were excluded;
- no automatic extension above 131,072;
- no GWOSC/GWTC access;
- 253,429,231,616 remaining free bytes, above the 100 GB floor.

## Execution history

The first one-by-128 tail layout and the fixed 32-by-4 recovery stopped on
prospective resource evidence. Their partial artifacts remain immutable and
were not reused. The successful 128-by-1 layout preserved the four conditional
tail populations and made only the atomic work partition dynamic.

The successful recovery used 32 worker processes on the 32-physical-core host.
The rare extreme-relative-magnification namespace produced genuine long-tail
stragglers. All streams completed within the frozen two-million-attempt and
96-hour per-microshard caps; none was skipped or substituted.

## Next gate

The next action is the exact terminal-probe release review:

1. verify the immutable training wheel in its isolated AutoDL CUDA runtime;
2. assemble a release packet bound to this committed closeout;
3. bind the three retained corrected-65k checkpoints and summaries;
4. record the delegated scientific and engineering review;
5. create and validate the exact 131k three-seed probe authorization.

Only after that gate passes may the terminal membership be opened and seeds
0, 1 and 2 be trained. All downstream phases remain closed.
