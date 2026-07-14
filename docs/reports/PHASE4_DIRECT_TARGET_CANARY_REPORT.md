# Phase 4 direct-target disposable canary report

## Outcome

The separately authorized direct-target canary passed. It generated exactly
eight train-namespace and eight validation-namespace engineering pairs, for 16
pairs total. The artifact is permanently non-scientific and cannot be used for
training, calibration, testing or reported performance.

The frozen identities were:

- generator commit: `2be777e727ef9d8e1a85f89c68966df5d37932b0`;
- generator wheel SHA-256:
  `14104f8aab3aa911fe43e27c311079f118add7ca8ad22178ca158c13d81d0a88`;
- environment-lock SHA-256:
  `792a93f24f6c38c18ec214665d34c8348e042b21beebd177333dae2c30224d8f`;
- RC.4 hash:
  `5aeaac395463bd073c44ead4ff4c5c729b5a2d4b4f1840c0825a53b30ab1bc98`;
- parent run: `phase4-canary-2be777e727ef-718204954753`.

## Execution and resume

The first invocation completed the train namespace and stopped intentionally
at exactly eight accepted pairs. The second invocation used the same commit,
wheel, environment, seeds and parent identity. It preserved the first shard
byte-for-byte and completed the validation namespace at exactly eight pairs.

The first shard SHA-256 was
`b892f51858788dde75687317d69ddf4f2d426fb37cf187626b8f68671551a255`
both before and after resume. The validation shard SHA-256 was
`37a3d29b0e3d32a5ffc5a710ad53557706dbc87c6822edb35decce3f7e5091ae`.

## Validation

Both namespaces passed the real JSON/Parquet/Zarr/schema health path, exact
array decomposition, finite density provenance, q=p equality and unit-weight
checks. Pair, source, lens, physical-system and noise IDs are disjoint across
the two namespaces. Required shard telemetry fields, manifests and independent
tree checksums are present. Total retained data are 15,417,242 bytes.

The canonical manifest SHA-256 is
`c1984616f2f7cea3d9d07b799cf1578f7e5d702174d2f6ba749ffb78d59afb40`.
No throughput or ESS endpoint was inspected.

## Safety boundary

No Stage A output directory or official Stage A identity was created. The
post-canary release gate remains `blocked_preexecution` and returns null
official identities. Scientific materialization, model training, calibration,
SBC, final evaluation, real noise and GWOSC/GWTC remain closed pending a new
human execution authorization.
