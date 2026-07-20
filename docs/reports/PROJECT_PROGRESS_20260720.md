# Project progress report — 2026-07-20

## Executive status

The project is on the frozen `1.2.0-rc.1` terminal-scale route. The exact
65,536-system direct-target increment is actively materializing with 32 CPU
workers; downstream terminal-probe and architecture software are already
implemented and execution-disabled so publication can hand off without a new
software build delay.

At the snapshot time, the active parent had 23 of 512 train shards complete:
2,944 of 65,536 new train systems (4.49%). Thirty-two additional shards were in
progress. All 32 worker children were alive, no result/error artifact existed,
approximately 50.6 GiB memory and 215.89 GB disk remained available.

## Completed work

| Workstream | Status | Result |
| --- | --- | --- |
| Scientific/generator foundations | 100% | RC.5 generator, alpha.3 schema, physics/whitening/Galkin/waveform gates qualified |
| Engineering qualification | 100% | 4,096 permanently non-scientific systems, 32 atomic shards, byte-identical resume |
| Direct-target Stage A | 100% | 32,768 train + 6,144 validation atomically published |
| Direct-target Stage B | 100% | 32,768 additional train; logical 65,536 train published |
| Numerical waveform correction | 100% | Five pathologies excluded and replaced; corrected 65,536 view passed independent closeout |
| Corrected probe ladder through 65k | 100% | 16k/32k/65k × three seeds completed; 65k remained decisively data-limited |
| Terminal 131k scientific contract | 100% | RC `1.2.0-rc.1`, exact 131,072 resource cap and independent 4×128 tail pool frozen |
| Terminal materialization software | 100% | Exact generator, atomic publisher, 32-worker scheduler and resource gates frozen |
| Terminal 131k probe software | 100% | Bounded reader, retained-tail evaluation, three-seed launcher and terminal comparison pushed |
| Terminal architecture software | 100% | Three probe reuses + at most nine new fits and validation-only selection pushed |
| Calibration/SBC/final software foundations | approximately 90% | Core materialization, metrics, calibration, inference, reference and ablation stacks exist; terminal-lock binding remains |

## Active and pending work

| Task | Current completion | Exit condition | Conservative remaining time |
| --- | ---: | --- | ---: |
| Terminal train increment | 4.49% (23/512 shards) | 65,536 systems in 512 atomic shards | about 30–36 active hours |
| Development-tail materialization | 0% | four disjoint namespaces ×128 systems | about 1–24 hours; rare-stratum cap dominates uncertainty |
| Logical 131k closeout | 0% | combined manifest, q=p/unit weights, group disjointness, hashes and disk gates pass | 1–3 hours after data generation |
| 131k probe fits | software 100%, execution 0% | seeds 0/1/2 complete from scratch on identical validation; retained/new tail scores complete | roughly 1–3 days on three GPUs |
| 65k→131k terminal decision | software 100%, execution 0% | exactly one frozen lock label; no extension above 131,072 | under 1 hour after fits |
| Architecture grid | software 100%, execution 0% | reuse three probe fits, complete at most nine fits, lock architecture by three-seed validation mean | roughly 2–5 days on three GPUs |
| Calibration + independent SBC | software mostly complete, execution 0% | 4,096 calibration + 2,048 SBC systems and frozen same-seed analyses | roughly 2–4 days |
| Final evaluation and baselines | software mostly complete, execution 0% | 20,480 sealed cases, all frozen diagnostics/baselines, no tuning on final data | roughly 4–8 days |
| Paper/release | not authorized/not started | claims fixed to evidence, figures/tables/model/data cards and reproducibility package | roughly 3–6 weeks after main results |

## Percent complete

Percentages below distinguish software readiness from scientific execution.

- Terminal 131k materialization: **4.46%** by accepted systems
  (2,944 / 66,048 including the future 512-case tail pool).
- Full terminal development-data inventory: **54.18%** materialized
  (existing corrected 65,536 train + 6,144 validation + 2,944 new train,
  divided by 131,072 train + 6,144 validation + 512 tail).
- Terminal probe workflow: **75% by completed seed fits** across the retained
  16k/32k/65k/131k ladder (9 historical corrected fits complete, 3 terminal
  fits pending); the terminal decision itself remains 0% executed.
- Publication-quality scientific pipeline overall: **approximately 55%**.
  This is a milestone-weighted estimate: generation and preregistration are
  mature, but terminal training, architecture choice, calibration/SBC, final
  evaluation and manuscript claims remain scientifically unexecuted.

## Blocking assessment

There is no current correctness blocker. The active critical path is compute
time for direct-target acceptance. The job is making atomic progress with
stable memory/disk and no error markers. Sixty-four workers remain deliberately
disabled because the 64-logical-CPU host would lose orchestration, storage and
operating-system headroom and each worker initializes large scientific library
thread pools.

The main schedule uncertainty is the rare-stratum development-tail acceptance
time, not a scientific ambiguity. Every downstream execution still requires an
exact post-publication authorization binding manifests, code/wheel and the CUDA
environment; the software needed for those gates is being completed before the
data arrive.

## Forecast

If the current host stays available and no hard resource gate fails:

- terminal atomic publication: approximately **1.5–2.5 days**;
- terminal probe result and size lock: approximately **3–6 days** from now;
- architecture lock: approximately **6–11 days** from now;
- calibrated/SBC and final scientific results: approximately **12–23 days**;
- complete submission-quality manuscript package: approximately **5–8 weeks**.

These are planning ranges, not guaranteed scientific outcomes. A failed frozen
gate changes the claim or invokes its preregistered stop; it does not authorize
post-result tuning.
