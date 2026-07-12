# Remote topology

## Roles

```text
Vultr: /root/work/lensing-4
  authoritative Git source, configs, docs, manifests, lightweight results
                    |
                    | ssh autodl-lensing / rsync without --delete
                    v
AutoDL: /root/autodl-tmp/lensing-4
  repo/ data_v2/ gwosc_cache/ runs/ manifests/ cache/ logs/ tmp/

Read-only legacy evidence on AutoDL
  /root/autodl-tmp/qkzhang
  /root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main
  /root/autodl-tmp/tmp  (PDF baseline assets)
```

## Connectivity and capacity verified on 2026-07-12

- SSH alias: `autodl-lensing`, dedicated Ed25519 identity, passwordless test passed.
- AutoDL user: `root`.
- GPUs: four NVIDIA RTX 5000 Ada, each reporting 32,760 MiB.
- AutoDL storage: 2.8 TB total, 2.4 TB used, 321 GB available, 89% full.
- Candidate sizes: wjx pair project 834 GB; qkzhang root 425 GB; `/root/autodl-tmp/tmp` 143 GB.
- New project root was created empty and uses negligible space.

## Safety boundaries

- `scripts/remote/sync_to_autodl.sh` excludes Git metadata, credentials and large
  data; it deliberately omits `--delete`.
- `scripts/remote/sync_results_back.sh` accepts only lightweight result types.
- Executable repository commands on AutoDL run from
  `/root/autodl-tmp/lensing-4/repo`, not the containing project root.
- `configs/remote/autodl.env` is mode 600 and ignored by Git; the committed file
  is `autodl.env.example`.
- A full hash of the approximately 1 TB relevant legacy waveform corpus was not
  attempted. At a sustained 200–500 MB/s it would require roughly 0.6–1.4 hours
  of ideal sequential I/O, excluding contention and metadata overhead.
- The Phase 1B publication directory is frozen read-only (`0555` directories,
  `0444` ordinary files). Its three closeout evidence hashes match the
  pre-Phase-1B.1 values.

## Resolved path relationships

The legacy roots are neither symlinks nor the same inode tree.

- `qkzhang` holds the original single-channel 0222/0228 arrays.
- The wjx repository is a later Git-managed pair-verification project. Its
  committed independence audit records full hashes for qkzhang 0222/0228 and
  its training configs refer to `${GW_DATA_ROOT}/SIS_data_0222`, etc.
- wjx `final_v3` and `ligo_reduced` contain later two-detector H1/L1 arrays.
  Checked CSVs are numerically identical to qkzhang 0222 to floating-point
  serialization precision; sampled `final_v3` and `ligo_reduced` waveforms are
  identical but stored under different inodes.
- The PDF point-regression baseline is a third lineage under
  `/root/autodl-tmp/tmp`, not a 0222/0228 training run.
