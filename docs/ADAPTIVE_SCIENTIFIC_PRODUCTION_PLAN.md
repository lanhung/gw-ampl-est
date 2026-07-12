# Adaptive scientific-production plan

Status: design-only preregistration `1.1.0-rc.1`. No data generation, training,
calibration, evaluation or external-data access is authorized.

The machine-readable authority is
`configs/statistics/adaptive_scientific_production_preregistration.yaml`, with
canonical hash
`ba5dae2aa769331b917d3f622bfc967c607700f9908521576301841cb71d804b`.

## Boundary inherited from Phase 3A

Phase 3A qualified generator commit
`fbcd0616611d9cdf915ef0af030e6061c1be7f59` using 4,096 engineering-only
pairs. Dataset `gwlens-v2-2.0.0-alpha.3-7081b2e8be3a84e1` remains permanently
outside every scientific split. Its rejection counts, throughput and storage
are engineering evidence only.

The RC.5 estimand, benchmark population, source-plane measure, observation
models and selection model remain the scientific parent contract. Phase 3B
changes planned allocation and decision rules, not those distributions.

## Nested training ladder

Scientific physical systems receive a deterministic SHA-256 rank before
materialization. Cumulative membership is rank below the rung cutoff:

```text
train_16k (16,384) ⊂ train_32k (32,768) ⊂ train_65k (65,536)
```

Source, lens, physical-system, pair, noise-segment and augmentation-parent
groups cannot cross splits. Every noise augmentation inherits its physical
system's split and never counts as another independent physical system.

There is no automatic rung above 65,536. Evidence that 65k remains
data-limited stops the phase and requires a new preregistration.

## Development pool

The 12,288-system development pool is fixed before materialization:

| Split | Count | Permitted role |
|---|---:|---|
| validation | 6,144 | learning-curve stopping and architecture selection |
| calibration_fit | 4,096 | post-hoc calibration after size/architecture freeze |
| sbc_diagnostic | 2,048 | independent SBC after calibration freeze |

The three splits are group-disjoint. Calibration cases cannot drive scale or
architecture selection. SBC cases never fit a calibration map.

## Final evaluation pool

The 20,480-system final pool is fixed by IDs, seeds, distributions and split
assignments before training, but its default materialization point is after
training size and architecture lock:

| Split | Count |
|---|---:|
| IID test | 8,192 |
| balanced tail | 4,096 |
| cross-family misspecification | 2,048 |
| parameter-region OOD | 2,048 |
| waveform mismatch | 2,048 |
| PSD mismatch | 2,048 |

No final-pool result may affect learning-curve stopping, architecture
selection, calibration fitting or proposal tuning. If final evidence fails,
the current claim is downgraded or a new preregistration is required; the
current model is not retuned against that evidence.

## Total scientific size by stopping point

Phase 3A pairs are excluded from all totals.

| Locked training rung | Training | Development | Final | Total |
|---|---:|---:|---:|---:|
| train_16k | 16,384 | 12,288 | 20,480 | 49,152 |
| train_32k | 32,768 | 12,288 | 20,480 | 65,536 |
| train_65k | 65,536 | 12,288 | 20,480 | 98,304 |

## Model-selection sequence

One fixed probe model—10 flow transforms and conditioner width 256—is trained
from scratch at each rung for seeds 0, 1 and 2 using the identical budget. It
uses validation only and no post-hoc calibration.

After training size is locked, the four existing transform/width combinations
are fit at that size for all three seeds. Architecture is selected by mean
validation NLP across seeds; no best seed is selected. Calibration, SBC and
final evaluation each remain behind later individual execution gates.

## Measured RC.5 resource projection

The projection linearly scales the Phase 3A measurements of 1,455,699 attempts,
21,404.39 active seconds and 4,450,694,559 published bytes per 4,096 accepted
pairs. Conservative peak storage adds 5% retained failure evidence, a 20 GB
run/cache reserve and one active 128-pair shard.

| Total systems | Attempts | Active hours | Published bytes | Peak bytes | Free bytes after peak |
|---:|---:|---:|---:|---:|---:|
| 49,152 | 17,468,388 | 71.35 | 53,408,334,708 | 76,217,835,649 | 254,232,768,383 |
| 65,536 | 23,291,184 | 95.13 | 71,211,112,944 | 94,910,752,797 | 235,539,851,235 |
| 98,304 | 34,936,776 | 142.70 | 106,816,669,416 | 132,296,587,092 | 198,154,016,940 |

A hypothetical exact 2× proposal improvement would halve attempts and active
time while leaving per-pair storage unchanged. That scenario is unmeasured and
cannot be presented as a result. Future production must persist continuous
process-tree/cgroup peak RSS, time-integrated CPU use and peak staging bytes.

## Authorization sequence

Phase 3B authorizes design only. Later human reviews must separately decide:

1. whether to run the 512-pair proposal-efficiency qualification;
2. which scientific rung may be generated;
3. whether probe training may begin;
4. whether calibration and SBC may open after size/architecture lock;
5. whether final IID/OOD/mismatch evaluation may be unsealed;
6. whether a separate real-noise/GWOSC/GWTC protocol may execute.

No item follows automatically from acceptance of this document.
