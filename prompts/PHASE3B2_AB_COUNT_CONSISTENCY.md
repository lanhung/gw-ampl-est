# Phase 3B.2 — Resolve proposal A/B qualification-count ambiguity

Work only on:

`phase3b/adaptive-production-preregistration`

Start from exact clean checkpoint:

`27d40b45be6ede4363ebb31ffba6bd112f51e53e`

This remains design-only.

Do not generate any pair.
Do not implement or run proposal-v2.
Do not train a model.
Do not fit calibration.
Do not run SBC, IID, OOD or mismatch evaluation.
Do not access GWOSC or GWTC.
Do not modify the Phase 3A generator or artifact.
Do not authorize Phase 3C.

Read:

- `AGENTS.md`
- `configs/statistics/adaptive_scientific_production_preregistration.yaml`
- `docs/PROPOSAL_EFFICIENCY_QUALIFICATION_PLAN.md`
- `docs/ADAPTIVE_SCIENTIFIC_PRODUCTION_PLAN.md`
- `docs/reports/PHASE3B1_TARGET_AND_STOPPING_HARDENING_REPORT.md`
- `results/phase3b/final_evaluation_commitment.json`
- `tests/test_phase3b_adaptive_preregistration.py`

The current RC.2 contains a contradiction:

- top-level proposal qualification count is 512;
- the A/B design requires 512 accepted pairs in each of two arms.

Resolve this by freezing a 1,024-total-pair future engineering A/B gate.

======================================================================
1. BUMP THE RELEASE CANDIDATE
======================================================================

Update `1.1.0-rc.2` to `1.1.0-rc.3` and generate a new canonical
configuration hash. Do not silently modify RC.2 while retaining its
version/hash.

======================================================================
2. FREEZE UNAMBIGUOUS A/B COUNTS
======================================================================

Replace the ambiguous proposal gate count with explicit fields equivalent to:

```yaml
proposal_efficiency_future_gate:
  qualification_id: proposal_v2_engineering_ab_qualification_v1
  arm_count: 2
  accepted_pair_count_per_arm: 512
  total_accepted_pair_count: 1024
  control_accepted_pair_count: 512
  candidate_accepted_pair_count: 512
  paired_blocks_per_arm: 16
  accepted_pairs_per_block: 32
```

Require arithmetic:

- 16 × 32 = 512 per arm;
- 2 × 512 = 1,024 total.

Remove or rename every ambiguous statement such as `accepted_pair_count: 512`,
`512-pair A/B run`, or `512-pair qualification` unless it explicitly says
`512 per arm`.

Use the wording: `512 accepted pairs per arm; 1,024 accepted engineering pairs
total`.

======================================================================
3. FREEZE ARM ARTIFACT IDENTITIES
======================================================================

The future control and candidate outputs must not share one dataset identity.
Freeze one parent A/B run ID, one RC.5 control dataset ID, one proposal-v2
candidate dataset ID, one parent comparison manifest, separate arm manifests
and checksums, and identical environment, worker-count and telemetry contracts.
Both arms are engineering-only; scientific/training/calibration/test use is
false. Neither arm may enter a future scientific split.

======================================================================
4. ADD CONSERVATIVE RESOURCE PROJECTION
======================================================================

Use measured Phase 3A values to record a conservative projection for 1,024
accepted pairs. Use the RC.5 measured rate for both arms when setting resource
gates.

Report control projected active time, candidate conservative projected active
time, combined conservative active time, combined projected publication bytes,
temporary/staging reserve and minimum free-space requirement. A hypothetical
candidate 2× speedup may be reported separately but must not lower the
prelaunch resource gate. The future authorization hard maximum is exactly 1,024
accepted pairs across both arms.

======================================================================
5. UPDATE COMMITMENT IDENTITIES
======================================================================

Update the preregistration version/hash referenced in
`results/phase3b/final_evaluation_commitment.json` and recompute
`results/phase3b/final_evaluation_commitment.sha256`. The commitment remains an
unfinalized design template; do not resolve the future scientific generator
placeholder.

======================================================================
6. UPDATE DOCUMENTATION AND TESTS
======================================================================

Update AGENTS.md, adaptive preregistration YAML, proposal-efficiency plan,
adaptive production plan, Phase 3B.1 report or a new Phase 3B.2 report,
project state, decisions, experiment registry and resource projections.

Create `docs/reports/PHASE3B2_AB_COUNT_CONSISTENCY_REPORT.md`.

Add tests proving 512 accepted pairs per arm, exactly two arms, exactly 1,024
total, 16 blocks × 32 pairs per arm, distinct control/candidate dataset
identities, engineering-only arms, the 1,024-total resource projection, all
execution flags false, proposal-v2 unauthorized and Phase 3C unauthorized.

Run pytest, Phase 3B focused tests, maintained-scope Ruff, mypy, package build,
canonical hash, commitment hash, A/B count and resource arithmetic checks.

Commit with:

`docs: resolve proposal A/B qualification count`

Push and stop for human review.
