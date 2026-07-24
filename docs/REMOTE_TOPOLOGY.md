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

## Endpoint and capacity audit on 2026-07-24

An owner-provided secondary SSH endpoint was tested as a possible four-GPU
offload host. It is another network entry point to the active AutoDL container,
not an independent machine. The following identities matched exactly through
both entry points:

- hostname and kernel boot ID;
- all four NVIDIA GPU UUIDs;
- the `/root/autodl-tmp/lensing-4` device and inode;
- the active tmux-session set.

No hostname, public address, port, password or private key for the secondary
entry point is committed.

The current container reports:

- four NVIDIA RTX 5000 Ada GPUs with 32,760 MiB each;
- 64 logical CPUs;
- approximately 125.5 GiB RAM;
- approximately 34.2 GiB available on the active 210.2 GB overlay at the audit
  instant.

The terminal 131k probe uses GPUs 0, 1 and 2 through separate
`CUDA_VISIBLE_DEVICES` bindings. GPU 3, and additional capacity on GPUs 0 and
1, were occupied by unrelated owner workloads at the audit instant. Those
processes must not be interrupted or repurposed implicitly.

The active three-seed probe cannot gain capacity from this second endpoint and
must not be migrated mid-run. A future genuinely independent host is eligible
only when it has a distinct boot ID and GPU UUID set and passes an exact
wheel/environment test. Independent future architecture fits may be assigned
across verified hosts; one official fit must remain on one immutable hardware
and software identity from launch through completion.

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
