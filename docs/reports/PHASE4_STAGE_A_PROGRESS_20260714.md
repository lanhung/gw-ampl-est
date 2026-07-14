# Phase 4 Stage A execution progress report — 2026-07-14

## Executive conclusion

Stage A is running correctly but is not complete. Neither of the following
completion items may be checked yet:

- materialize and atomically publish exactly 32,768 train plus 6,144
  validation systems;
- validate the complete publication, commit final evidence/report and review
  the separate probe-training gate.

At the frozen snapshot time `2026-07-14T08:05:43Z`, 42 of 256 train shards
were complete and validation had not started. No final dataset manifest,
execution-result file or published parent existed. This report is an
in-progress audit snapshot, not a Phase 4 completion report.

## Frozen execution identity

- RC.4 version: `1.1.0-rc.4`;
- RC.4 hash:
  `5aeaac395463bd073c44ead4ff4c5c729b5a2d4b4f1840c0825a53b30ab1bc98`;
- generator commit: `2be777e727ef9d8e1a85f89c68966df5d37932b0`;
- generator wheel SHA-256:
  `14104f8aab3aa911fe43e27c311079f118add7ca8ad22178ca158c13d81d0a88`;
- parent run: `phase4-stage-a-2be777e727ef-d3a60034bbd6`;
- train dataset:
  `gwlens-v2-2.0.0-alpha.3-ef4baee395dbe94b-train`;
- validation dataset:
  `gwlens-v2-2.0.0-alpha.3-33c3e6cb507a03e9-validation`.

The ready release certificate had no blockers and bound the exact canary,
dependency lock, wheel, PSD hashes, free-space gate and closed training/GWOSC
flags. Its SHA-256 is
`a019fcfbe18879220a5ade8fdf75f582615d757cfd990ab58d4695c419a7227e`.

## Measured progress

| Quantity | Completed | Target | Formal completion |
| --- | ---: | ---: | ---: |
| Train shards | 42 | 256 | 16.41% |
| Train accepted pairs in complete shards | 5,376 | 32,768 | 16.41% |
| Validation shards | 0 | 48 | 0% |
| Validation accepted pairs | 0 | 6,144 | 0% |
| Total Stage A shards | 42 | 304 | 13.82% |
| Total accepted pairs in complete shards | 5,376 | 38,912 | 13.82% |

Sixteen train shards were partial. Their 903 written chunks are deliberately
excluded from the formal accepted count because partial shards are never
publication evidence.

The run began at `2026-07-14T02:07:18Z`. At the snapshot it had used 21,505
seconds, approximately 5.97 hours. Sixteen workers consumed about 1,597% CPU
and 5.36 GiB aggregate RSS. Stage A staging occupied 7,467,857,940 bytes and
315,543,281,664 bytes remained free. The execution log contained zero matched
error lines.

## Read-only health evidence

The first completed official train shard was independently inspected without
mutating the running dataset. Its complete marker and artifact checks passed,
all 128 Parquet records passed the direct-target validator, proposal and
evaluation log probabilities matched, and every importance weight was exactly
one. This is health evidence only; the runner will repeat full namespace and
cross-split validation after all 304 shards finish.

## Publication status

The official parent exists only under the staging root. The publication root
does not contain the parent identity. The following required final artifacts
do not yet exist:

- final `dataset_manifest.json`;
- `stage_a_execution_result.json`;
- final publication-tree SHA-256;
- final byte count and remaining-space measurement;
- full train/validation grouped-ID leakage result;
- final Phase 4 completion report.

The runner publishes only after all train and validation shards pass, using a
same-filesystem atomic parent rename. It would be scientifically incorrect to
represent the current staging data as published or training-ready.

## GitHub and AutoDL boundary

All source code, configurations, authorization files and completed small
canary evidence through merge commit
`af6a5c1d2579058f3843d8dcada02166a1750c65` are on GitHub and CI passed. This
progress report, release certificate and launch/progress manifests are small
review evidence intended for GitHub.

The growing Zarr, Parquet and attempt-journal artifacts remain exclusively on
AutoDL under `/root/autodl-tmp/lensing-4`. Uploading those multi-gigabyte
artifacts to Git would violate repository policy. Their final manifests and
checksums will be committed after atomic publication.

## Remaining route

1. Finish the remaining train shards.
2. Generate all 48 validation shards.
3. Run complete namespace validation, duplicate/group-leakage checks and exact
   unit-weight validation.
4. Atomically rename the parent from staging to publication.
5. Record final checksums, byte counts and free space.
6. Copy only the final small evidence into `results/phase4/`, update project
   state/decisions/failures/registry and commit the Phase 4 completion report.
7. Only after human/delegated review, create a separate training authorization.

The probe subset must be selected using the frozen SHA-256 rank over the
complete 32k train membership. It must not be replaced with the first 16,384
systems that happen to finish generation. Training, calibration, SBC, final
evaluation, real noise and GWOSC/GWTC remain closed.

## Time estimate

The completed-shard rate at the snapshot was about 900 accepted pairs per
hour, while partial-shard progress suggested a somewhat higher instantaneous
rate. A reasonable operational estimate was 32–43 additional hours including
final validation and publication. The preregistered conservative projection
remained approximately 51 additional hours. These are projections, not
completion evidence.
