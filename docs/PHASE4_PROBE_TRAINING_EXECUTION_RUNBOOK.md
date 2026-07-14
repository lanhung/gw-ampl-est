# Phase 4 probe-training execution runbook

Status: implementation complete; scientific data access and optimization remain
closed pending Stage A atomic publication and a separate exact authorization.

## What the release path now verifies

The training reader resolves the one atomic Stage A parent publication. Stage A
does not write child `dataset_manifest.json` files: the passed train and validation
identities are named by the parent's `dataset_manifest.json`. The release path
checks that parent manifest, both child run manifests, exact 32,768/6,144 counts,
all 304 complete shards, direct-target unit weights, the RC.4 hash, the generator
commit and train/validation physical-system disjointness before opening Parquet or
Zarr.

The 16,384-system membership is computed only after all 32,768 train physical
system IDs exist. It is the lowest SHA-256-ranked half under seed `2026071401` and
domain `adaptive_scientific_split_assignment_v1`; it is never the first half
generated.

Rung preprocessing is a separate bounded-memory pass. It opens Parquet metadata
but no strain arrays, fits observed-only EM statistics and log-magnification target
statistics with streaming moments, hashes them, and writes one immutable
`rung_preparation.json`. The three seed fits reuse that file.

Training epochs randomize both the shard order and rows within each shard from
the frozen seed and epoch. This remains a full permutation of the rung, while
keeping lazy reads shard-local instead of reopening a Parquet/Zarr store for
nearly every randomly ordered example.

## Future authorization shape

After Stage A publication, the reviewed authorization must contain values
equivalent to the following. Placeholders cannot pass the gate.

```yaml
authorization_status: authorized_probe_training_only
authorization:
  stage_a_data_access_authorized: true
  scientific_probe_training_authorized: true
  probe_optimizer_execution_authorized: true
  learning_curve_decision_authorized: true
  model_tuning_authorized: false
  calibration_authorized: false
  sbc_authorized: false
  final_evaluation_authorized: false
  gwosc_gwtc_access_authorized: false

authorized_training_rungs: [16384, 32768]
authorized_training_seeds: [0, 1, 2]
probe_membership_root_seed: 2026071401
maximum_concurrent_fits: 3
data_loader_worker_processes: 4

stage_a_generator_commit: "2be777e727ef9d8e1a85f89c68966df5d37932b0"
stage_a_parent_manifest_sha256: "__FROM_ATOMIC_PUBLICATION__"
final_evaluation_commitment_sha256: "c13412eced163bac26abc4b22d054f3a6fa967e7e5a4dd7849ebf54f42df6083"

immutable_training:
  git_commit: "__PHASE4_TRAINING_RUNNER_MERGE_COMMIT__"
  model_configuration_hash: "__FROZEN_MODEL_CONFIG_HASH__"
  environment_lock_path: configs/environment/phase4-training-freeze.txt
  environment_lock_sha256: "__FROZEN_ENVIRONMENT_HASH__"

training_output_root: /root/autodl-tmp/lensing-4/training/phase4/probe
```

The authorizing commit may change only this authorization and `AGENTS.md` after
the frozen training-code commit. The runtime must exactly match Python 3.10.12,
Torch 2.10.0+cu128, nflows 0.14 and the remaining versions in
`configs/models/phase4_probe_nsf.yaml`. CUDA is mandatory.

The clean AutoDL candidate environment is recorded in
`configs/environment/phase4-training-environment.yaml`; its normalized freeze
SHA-256 is `2e45000a8cea6712ae307c87782c593245ad56607a772f27a0cc5af726e37b95`.
The final training wheel and merge commit remain null until this implementation
is merged, and must be filled by the later execution authorization rather than
rewriting the scientific model configuration.

## Execution sequence

First prepare the 16k rung once, then launch seeds 0, 1 and 2 on three distinct
GPUs through the single launcher command:

```bash
python scripts/phase4/launch_probe_rung.py \
  --root /root/autodl-tmp/lensing-4/repo \
  --authorization configs/execution/phase4_probe_training_authorization.yaml \
  --stage-a-publication __ATOMIC_PARENT_PUBLICATION__ \
  --environment-lock configs/environment/phase4-training-freeze.txt \
  --psd-root __VERIFIED_PSD_DIRECTORY__ \
  --output-root /root/autodl-tmp/lensing-4/training/phase4/probe \
  --training-commit __FROZEN_TRAINING_COMMIT__ \
  --rung 16384 \
  --gpu-indices 0,1,2
```

Repeat with `--rung 32768`. Each fit is initialized from scratch, uses the same
optimizer/epoch/early-stopping contract, writes atomic last/best checkpoints and
can resume only when its complete run identity matches. No best-seed selection is
allowed.

After all six fits complete, run the preregistered comparison:

```bash
python scripts/phase4/compare_probe_learning_curve.py \
  --authorization configs/execution/phase4_probe_training_authorization.yaml \
  --training-output-root /root/autodl-tmp/lensing-4/training/phase4/probe \
  --output /root/autodl-tmp/lensing-4/training/phase4/probe/learning_curve_decision.json \
  --execute
```

The comparison uses identical 6,144 validation IDs, all three seeds and the frozen
10,000-replicate paired physical-system bootstrap. The development output contains
per-case NLP, CRPS, marginal intervals, joint HPD coverage, EM-cell coverage and
validation-internal tail views. Calibration, SBC, IID/OOD/mismatch data and the
sealed final evaluation are never read.

The only outcomes are `lock_train_32k` or `continue_to_train_65k`. The latter does
not itself authorize Stage B generation or a 65k fit.
