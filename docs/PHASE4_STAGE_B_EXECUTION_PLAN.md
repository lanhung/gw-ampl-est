# Phase 4 Stage B direct-target extension execution plan

## Scope

The completed 16k/32k development comparison requires the one preregistered
conditional extension. Stage B adds exactly 32,768 new direct-target training
physical systems in 256 atomic shards of 128. It does not create new
validation, calibration, SBC or final-evaluation systems.

The existing 32,768 Stage A training systems remain immutable. The 65,536
training rung is a two-component reference over the immutable Stage A train
publication and the new atomic Stage B publication; data are not copied or
rewritten.

## Frozen generation identity

- evaluation/training distribution:
  `balanced_literature_informed_benchmark_v1`;
- exact direct-target weights: `q_train=p_eval`, `log_weight=0`, `weight=1`;
- generator commit: `2be777e727ef9d8e1a85f89c68966df5d37932b0`;
- generator wheel SHA-256:
  `14104f8aab3aa911fe43e27c311079f118add7ca8ad22178ca158c13d81d0a88`;
- root seed: `2026071403`;
- attempt namespace: `phase4-stage-b-train-direct-target-v1`;
- ID prefix: `phase4-stage-b-train`.

The orchestration commit is frozen separately before release. The runner uses
the already qualified immutable generator wheel; its new code is limited to
the exact-count gate, resume control, cross-publication leakage validation and
atomic combined-reference publication.

## Resource and resume plan

Stage A measured 46,491,822,064 bytes for 38,912 systems. Linear projection for
the 32,768-system increment is 39,151,008,054 bytes. With the frozen staging
and reserve model, projected peak use is 61,247,642,662 bytes and prelaunch
free space must be at least 161,247,642,662 bytes. The hard Stage B output cap
is 50 GB and post-publication free space must remain at least 100 GB.

Stage A measured throughput projects about 33.4 active hours. The conservative
execution cap is 60 hours. Before launch the release certificate, run manifest,
log path and exact identities must exist. Re-running the identical command
verifies every completed shard and schedules only missing shard indexes. A
complete shard is never rewritten.

## Publication and safety

After all 256 shards validate, the runner checks pair, source, lens, physical-
system and noise IDs against both Stage A train and validation publications.
It then atomically publishes Stage B and creates an atomic small reference
manifest for the nested 65,536-system training rung.

Stage B publication does not authorize its optimizer. A separate gate must
bind the combined manifest, training code/wheel, model config and CUDA
environment before the three 65k fits begin. Calibration, SBC, final
evaluation, extension above 65,536 and GWOSC/GWTC remain closed.
