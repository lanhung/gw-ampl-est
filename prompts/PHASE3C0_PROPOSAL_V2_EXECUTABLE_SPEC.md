# Phase 3C-0 — Executable proposal-v2 specification and implementation

Execute Phase 3C-0 only.

Read first:

- `AGENTS.md`
- `configs/execution/phase3c0_proposal_v2_design_authorization.yaml`
- `configs/statistics/adaptive_scientific_production_preregistration.yaml`
- `docs/PROPOSAL_EFFICIENCY_QUALIFICATION_PLAN.md`
- `docs/ADAPTIVE_SCIENTIFIC_PRODUCTION_PLAN.md`
- `docs/reports/PHASE3B2_AB_COUNT_CONSISTENCY_REPORT.md`
- `configs/statistics/phase2_preregistration.yaml`
- `docs/PHYSICS_CONVENTIONS.md`
- `docs/PROVENANCE_AND_SEEDS.md`
- `docs/PRIVILEGED_INPUT_POLICY.md`

Work only on:

`phase3c0/proposal-v2-executable-spec`

This phase may implement and validate probability distributions and a future
A/B runner skeleton.

Do not generate any waveform pair.
Do not call the full accepted-pair production generator.
Do not run the 1,024-pair A/B qualification.
Do not train or tune a model.
Do not fit calibration.
Do not run SBC, IID, OOD or mismatch evaluation.
Do not access GWOSC or GWTC.
Do not modify the Phase 3A artifact.
Do not authorize Phase 3C-A.

======================================================================
1. PRESERVE THE SCIENTIFIC TARGET
======================================================================

Do not modify adaptive preregistration RC.3 or its canonical hash.

The scientific evaluation target remains the inherited RC.5 target.

Create a separately versioned engineering proposal specification:

- proposal version: `proposal-v2-exact-mixture-1.0.0-rc.1`;
- independent configuration hash;
- no execution authorization.

Create `configs/proposals/proposal_v2_exact_mixture.yaml`.

======================================================================
2. EXACT MIXTURE
======================================================================

Implement the full pre-selection latent proposal:

    q_v2(theta)
      = 0.2 * q_rc5(theta)
      + 0.6 * q_wide(theta)
      + 0.1 * q_narrow(theta)
      + 0.1 * q_low_z(theta)

The weights must sum exactly to one. The 0.2 RC.5 component is the mandatory
broad-support safety component.

All latent factors not explicitly changed below must be identical to the RC.5
broad proposal, including lens-family probability, lens and source parameters,
BBH source masses, spins, sky/orientation, external convergence, EM-cell
assignment and noise/detector provenance rules.

Do not condition or normalize q_v2 on multiple imaging, synthetic detection or
accepted-pair status. Those remain explicit selection events.

======================================================================
3. SOURCE-PLANE COMPONENTS
======================================================================

RC.5 source coordinates are `u_x, u_y in [-2.5, 2.5)`. The RC.5 component
remains uniform on this square.

For q_wide, u_x and u_y are independent zero-mean truncated normals with sigma
1.5. For q_narrow, sigma is 0.8. Both use lower bound -2.5 inclusive and upper
bound 2.5 exclusive.

For one dimension implement:

    f_sigma(u) = phi(u / sigma) /
      [sigma * (Phi(2.5 / sigma) - Phi(-2.5 / sigma))]

inside the half-open interval and zero outside. The 2D density is the product.
Do not describe these components as exact caustic distributions.

======================================================================
4. LOW-REDSHIFT COMPONENT
======================================================================

q_low_z uses the q_wide source-plane density. Define:

    z_min = max(0.5, z_lens + 0.1)
    z_max = 3.0
    t = (z_source - z_min) / (z_max - z_min)
    t ~ Beta(alpha=1, beta=2)
    f(z_source | z_lens) = 2 * (1 - t) / (z_max - z_min)

on the half-open support. All other factors equal RC.5. Invalid ranges fail
closed rather than clipping.

======================================================================
5. EXACT FULL LOG DENSITY
======================================================================

Implement component sampling, component and complete latent-state log
densities, stable log-sum-exp mixture density, exact RC.5 broad density, exact
inherited evaluation-target density, and
`log_w = log_p_eval - log_q_v2`.

Include every conditional normalization term: source-redshift interval,
mass-ratio interval, source-plane Jacobian, truncated normalizations, discrete
lens-family probability and mixture component probability. Sampler/evaluator
half-open semantics must agree.

======================================================================
6. SOFTWARE STRUCTURE
======================================================================

Prefer `src/gwlens_mm/proposals/`. Implement typed immutable specifications
and process-local RNG. Required capabilities: sample one full latent proposal,
evaluate component/mixture/evaluation log density and importance weight,
serialize proposal identity/hash, and replay deterministic seeds. Never use
global NumPy, Bilby or Lenstronomy RNG state.

======================================================================
7. PRIVILEGED-PROVENANCE CONTRACT
======================================================================

Component index, component/full log densities, evaluation log density, log and
normalized importance weights, and proposal RNG seeds are privileged
provenance—not deployable model inputs. Update denylist/tests if necessary. No
training code executes.

======================================================================
8. ANALYTIC AND NUMERICAL NORMALIZATION TESTS
======================================================================

Test analytically/numerically: 1D and 2D truncated-normal normalization,
conditional Beta normalization including boundaries, complete component
normalization, mixture weight sum, safety support positivity, stable log-sum-exp
against high precision, support confinement and half-open upper boundaries.
Use deterministic quadrature and freeze tolerances before tests.

======================================================================
9. LATENT-ONLY IMPORTANCE PREFLIGHT
======================================================================

At least 200,000 deterministic q_v2 latent draws are authorized. Do not call
lens solver, waveform, detector, selection, storage publication or accepted-
pair generation.

Report finite fractions, mean unnormalized weight, overall/family relative ESS,
maximum normalized weight, quantiles, component counts and replay hash.

Require all finite, mean weight in [0.98,1.02], overall ESS >=0.50, each family
ESS >=0.40, no support hole and byte-identical replay. Failure stops without
retuning this frozen version.

======================================================================
10. FUTURE A/B RUNNER SKELETON
======================================================================

Implement config validation and a dry-run-only skeleton freezing two arms, 512
pairs per arm, 1,024 total, sixteen blocks per arm, 32 per block, alternating
order with even/odd reversal, one parent ID, distinct arm IDs/manifests/checksums,
one comparison manifest, identical resources and sequential arm-block execution.
Do not run the generator.

======================================================================
11. FUTURE FAIL-CLOSED EXECUTION CAPS
======================================================================

Freeze maximum attempts per arm 1,000,000; active time per arm 6 hours;
accepted pairs per arm 512; total 1,024. Candidate cap failure retains RC.5;
control cap failure invalidates comparison. Partial/capped blocks cannot be
omitted.

======================================================================
12. TELEMETRY CONTRACT
======================================================================

Future telemetry persists per-block active wall seconds, attempts, accepted
pairs, CPU seconds, process-tree/cgroup peak RSS, integrated CPU utilization,
lens/proposal/density/waveform/kinematics/storage/checksum times, peak staging
bytes and environment identity. Operator pauses are explicitly and identically
excluded.

======================================================================
13. REQUIRED OUTPUTS
======================================================================

Create:

- `configs/proposals/proposal_v2_exact_mixture.yaml`
- `docs/PROPOSAL_V2_EXECUTABLE_SPEC.md`
- `docs/adr/ADR-004-proposal-v2-exact-mixture.md`
- `docs/reports/PHASE3C0_PROPOSAL_V2_IMPLEMENTATION_REPORT.md`
- `results/phase3c0/proposal_v2_latent_validation.json`
- `results/phase3c0/proposal_v2_replay.sha256`
- proposal source code, A/B dry-run validation code, unit/property tests.

Do not commit large Monte Carlo samples.

======================================================================
14. COMPLETION GATE
======================================================================

Run full pytest, proposal-focused tests, maintained-scope Ruff, mypy, package
build, proposal configuration hash and deterministic replay checks.

Pass only when components/densities/support/preflight/privileged fields/A-B
contracts/caps all pass, zero waveform pair is generated, no A/B qualification
runs and all execution flags remain false.

Update AGENTS.md, decisions, failures, project state and experiment registry.

Commit with `feat: implement exact proposal v2 distribution`, push and stop
for human review. Do not create or merge a PR. Do not authorize Phase 3C-A.
