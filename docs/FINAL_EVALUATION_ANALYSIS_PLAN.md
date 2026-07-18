# Final-evaluation analysis plan

## Scope

Preregistration `1.1.0-rc.6` freezes the analysis of the already committed
20,480-system final-evaluation pool. It does not authorize materialization,
unsealing, checkpoint inference, ablation fitting, baseline execution, or any
use of GWOSC/GWTC.

RC.6 is downstream-only. It does not change the estimand, direct evaluation
target, selection model, observation model, split counts, generator commit,
attempt streams, accepted-rank rules, seeds, or the finalized generation
commitment. The immutable generation configuration remains hash
`11277a2a...574d66`; the commitment remains `c13412ec...df6083`.

## Entry gate

Final evaluation can be opened only after all of the following are atomically
published and bound by a later authorization:

1. the locked 32k or 65k training size;
2. one selected architecture based on mean validation NLP across seeds 0, 1,
   and 2;
3. all three retained selected-architecture checkpoints;
4. the 4,096-system calibration-fit publication and one immutable calibration
   map for every retained seed;
5. the independent 1,024-realization SBC result for every retained seed;
6. the sealed 20,480-system final pool and its complete leakage validation.

No final case may affect training-size selection, architecture selection,
calibration fitting, or model tuning.

## Per-seed inference

Every retained seed is evaluated. Each case receives 4,096 posterior draws,
with a future execution gate fixing a bounded draw microbatch no larger than
512. Draws from different model seeds are never pooled to construct a
case-level interval or region. The matching seed's EM-cell conformal map is the
primary calibration map; its global map is secondary. Final data never refit a
map.

Each case records joint NLP per target dimension, CRPS for both log absolute
magnifications, calibrated marginal and joint region membership, and interval
width. Reports retain raw counts and Wilson 95% intervals, every seed, both
lens families, all IID EM cells, all tail strata, and every declared diagnostic
context. Cross-seed summaries are the arithmetic mean and sample standard
deviation; a best seed is never selected.

## Executable cross-family semantics

The frozen generator uses four legacy materialization context IDs. Review
found that one legacy label asked for EPL inference at a fixed density slope
2.08, while the deployed estimator only accepts the lens-family one-hot and
has no slope input. That analysis was therefore not executable.

RC.6 preserves all four materialization namespaces and changes only their
post-materialization interpretation:

| Materialization context | Executable analysis |
|---|---|
| `sie_truth_epl_assumed` | condition on EPL; the training EPL slope prior is marginalized |
| `epl_truth_sie_assumed` | condition on SIE |
| `sie_truth_family_marginalized` | equal-density mixture of SIE- and EPL-conditioned posteriors |
| `epl_truth_family_marginalized` | equal-density mixture of SIE- and EPL-conditioned posteriors |

For the family mixture,

\[
\log p(\mu\mid x)=\operatorname{logaddexp}
  (\log p_{\rm SIE},\log p_{\rm EPL})-\log 2.
\]

Within each model seed, exactly 2,048 draws from each family condition form a
4,096-draw equal mixture. Component calibration maps are not averaged; the
same-seed, matching-EM-cell calibrated threshold is applied to the mixture
region score. These cells are misspecification diagnostics, not claims about a
fixed EPL slope.

## Gates and diagnostics

IID marginal, joint, and per-EM-cell coverage use the frozen absolute-error
tolerances, each defined as the maximum of its floor and three binomial
standard errors. Balanced-tail strata each contain 1,024 systems and use the
frozen 0.06 floor. Every seed and relevant cell/stratum must pass; an aggregate
cannot hide a failure.

Cross-family, parameter-OOD, waveform-mismatch, and PSD-mismatch splits are
diagnostic-only. Their results may narrow or lower claims but may not trigger
retuning.

## Ablations and baselines

GW-only and EM-only are separately trained at the locked size and selected
architecture for all three seeds under the same optimizer and budget. GW-only
retains strain, detector masks, observed timing, and the family condition while
removing EM values/masks/astrometry. EM-only removes strain, detector masks,
and observed GW timing while retaining EM values and the family condition.
No additional architecture search is allowed.

The inherited matched non-neural lens posterior and 256-case joint-likelihood
gold standard still require a separately reviewed executable likelihood
specification before final evaluation can be unsealed. The legacy SIS point
regressor remains an explicitly out-of-domain stress control, not a matched
competitor.

## Fail-closed boundary

The implementation in `gwlens_mm.training.final_evaluation` is pure. It has no
dataset reader, checkpoint loader, optimizer, materializer, or unsealing path.
Its dry plan leaves every official identity null and every execution result
false. A later release gate must bind all data, checkpoint, calibration,
environment, and code hashes before a final case can be opened.
