# Phase 3C-A proposal-v3 A/B qualification report

## Outcome

Phase 3C-A stopped fail-closed at the first matched-block health gate with
machine state `execution_failed`. This run does not qualify or reject
proposal-v3 on throughput or post-selection ESS.

The frozen generator commit was
`185e68d4346d84edc118a9197ffb8bceeb026ee4`. Local verification passed 191
tests with three optional dependency skips; the exact synchronized AutoDL
checkout passed 198 tests. Mass-sheet, source-plane solver-union, waveform
boundary, whitening, Galkin convergence and privileged-input preflight gates
all passed before pair generation.

## Retained first matched block

The runner atomically completed exactly one 32-pair block in each arm:

- RC.5 control dataset
  `gwlens-v2-2.0.0-alpha.3-79588465a35421e2-control`, block tree SHA-256
  `d6091efd63b454ea528f59290914b5b6dd4a26dcd8c71d198ef34eba9da6e0dc`;
- proposal-v3 candidate dataset
  `gwlens-v2-2.0.0-alpha.3-17bb6106d0192dcd-v3`, block tree SHA-256
  `13ae11619ea278110b4a618cd6c60251bb97a82e0ab75d5170c3407d572c5e91`.

The parent run is `phase3ca-185e68d4346d-0ed6958442da`. Both complete blocks
remain as engineering-only staging evidence on AutoDL. No parent or arm
publication was created. Total completed accepted evidence is 64 pairs, well
below the 1,024-pair hard maximum.

## Health-gate failure

After both blocks completed, the health validator attempted to read
`DistributionMetadata.evaluation_log_probability`. The alpha.3 schema names
this field `evaluation_prior_log_probability`, so the validator raised
`AttributeError` before finishing the health gate.

This is an implementation defect in the new health-check path. It is not
evidence of a nonfinite density, physics failure, support hole, failed
post-selection ESS or failed throughput threshold. The runner correctly wrote
`execution_failed` and stopped before block 1.

The frozen instructions prohibit changing code after official generation
begins under the same artifact identity. Therefore this report does not patch
the field access or resume the run. A reviewed new generator commit and new
dataset identities are required. The retained blocks must not be mixed into a
replacement run.

## Statistical boundary

No interim or final throughput ratio was computed. The 10,000-replicate paired
bootstrap was not run. Post-selection ESS and support gates were not run.
Proposal-v3 therefore remains neither passed nor failed for future scientific
production.

No scientific data, training, tuning, calibration, SBC, IID/OOD/mismatch
evaluation, real noise, GWOSC or GWTC access occurred. Stage A remains closed.
