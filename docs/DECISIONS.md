# Decisions

## D001 — Control plane

Vultr `/root/work/lensing-4` is the only source-code authority. AutoDL `repo/` is
disposable.

## D002 — Legacy immutability

qkzhang, wjx and `/root/autodl-tmp/tmp` baseline assets are immutable inputs.

## D003 — Data lineage

qkzhang is authoritative for original 0222/0228; wjx is a downstream
pair-verification project; `/tmp` is authoritative for the PDF point-regression
baseline.

## D004 — Legacy scientific role

No legacy data enter v2 main training or final testing. They are baseline/smoke
assets only.

## D005 — Implementation reuse

Extract physics and reproducibility concepts only after unit tests. Do not import
legacy monolithic generators as the v2 implementation.

## D006 — Real-noise language

No existing catalog is real detector noise. `realobs` is interpreted only as
simulated observation proxies.

## D007 — Phase gate

The next phase begins with physics/schema/tests. No full generation, training or
GWTC work is authorized by Phase 0 completion.

## D008 — Storage gate

Any AutoDL action expected to create >10 GB requires a prior estimate, manifest,
log path and resume plan. Current free space is 321 GB.

## D009 — Quantity vocabulary

Relative flux is always secondary over an explicitly identified primary. It is
not implicitly faint over bright and may exceed one outside bounded analytic
controls. `mu_rel`, `A21` and numbered-image shorthand are not v2 canonical
fields.

## D010 — Physical systems versus selected pairs

Solvers and lens truth retain all physical images. A separate selected-pair
object identifies the two GW image slots, primary definition, unselected
images, and censored images.

## D011 — Input policy

Deployable inputs are fail-closed: exact allowlist membership is required and
denylisted, suspicious-alias, duplicate, and unknown fields fail validation.
Truth/target/diagnostic permission never implies input permission.

## D012 — Split policy

Model-selection validation, calibration, IID testing, diagnostics, and OOD
tests are distinct. Source, lens/system, pair, noise segment, and augmentation
groups cannot cross splits.

## D013 — Phase 1B storage

Use Zarr v2 plus Parquet with staged unique shards and single-writer manifest
publication. A switch to sharded HDF5 plus Parquet requires an ADR amendment.

## D014 — Phase 1A stop gate

The 48-pair YAML is execution-disabled. Human acceptance of Phase 1A is
required before SIE/EPL integration or any smoke waveform generation.

## D015 — Observation associations

EM astrometry is keyed by physical image ID. Timing is an uncertainty-bearing
observation product. Noise provenance is a six-slot primary/secondary by
H1/L1/V1 grid aligned exactly with the detector mask.

## D016 — Image and primary completeness

Every physical image outside the selected pair has exactly one unselected or
censored status. Earliest, brightest, and minimum primary claims are checked;
catalog anchor imposes no physical ordering.

## D017 — Missing detector convention

Unavailable noisy, clean, and noise slots are float32 zeros and must be ignored
using the mandatory mask. NaN is not a normal missing value. Available slots
must satisfy noisy equals clean plus noise within tolerance.

## D018 — EM domains and redshift ordering

Einstein radius and velocity dispersion are positive; observed redshifts are
nonnegative. Point-estimate `z_l < z_s` is recorded as a quality flag rather
than a hard cut because photometric posteriors may overlap.

## D019 — Alpha.2 before materialization

Schema `2.0.0-alpha.2` supersedes metadata-only alpha.1. No migration is needed
because no v2 dataset was generated. Complete manifests require target count
and every published artifact checksum to be complete.

## D020 — Engineering split and use boundary

Phase 1B records use the explicit `engineering_smoke` split. The dataset
manifest sets `dataset_purpose=engineering_smoke` and
`scientific_use_authorized=false`; these records are excluded from all future
training, calibration, IID, OOD, and reported performance analyses.

## D021 — Separated-image waveform construction

Each selected image has an independently centered one-second window and its
own geocentric detector-response time. Galaxy-scale delay is metadata, never a
multi-day shift inside the tensor. Amplitude is validated against an unlensed
reference at the same response time, never by a raw cross-image detector ratio.

## D022 — Phase 1B publication identity

The published smoke dataset is
`gwlens-v2-2.0.0-alpha.2-ae86beab1c132682`. Its generator commit is
`d7287f8cc800406cc1e727a177fd27d7ca02cddf`; its authorization commit is
`a607ddb6844cb8a9cd6761011cb545633cd4fdf1`. Zarr v2 and Parquet artifacts
remain on AutoDL and are rooted by a complete checksum manifest.

## D023 — Solver time coordinates

`PhysicalImage` keeps `fermat_potential_dimensionless` and
`arrival_time_seconds` as separate optional quantities. A valid image supplies
at least one finite value, and all images in a solution supply one common
ordering coordinate. SIS analytic controls do not pretend their dimensionless
coordinate is seconds; Lenstronomy exposes both raw Fermat potential and an
earliest-normalized physical delay.

## D024 — Frozen dataset versus evolving API

Phase 1B.1 changes solver API and fixture evidence only. The frozen smoke
arrays and metadata remain exactly those generated by commit
`d7287f8cc800406cc1e727a177fd27d7ca02cddf`; they are not regenerated to adopt
a post-publication API cleanup.

## D025 — Phase 1B publication freeze

PR #2 merged after CI at `2a8d8de39d332f90339bd4e7d4c49f66697e6c01`.
The tag `gwlens-v2-2.0.0-alpha.2-ae86beab1c132682` points to the exact
generator commit, not the later solver API cleanup. The corresponding AutoDL
published directory is read-only; the manifest, records, and validation hashes
remain unchanged.

## D026 — Phase 2 is preregistration, not execution

Phase 2 freezes the scientific question and evaluation protocol before any
production simulation or model fitting. Its output is documentation and an
execution-disabled statistical configuration. Phase 3 remains separately
gated.

## D027 — Model-conditional estimand

The primary estimand is the joint posterior for the selected images' absolute
magnifications conditional on the adopted galaxy-lens family, cosmology,
external-convergence/model-discrepancy treatment, selection model, and the
available GW and EM observations. Relative magnification is derived but cannot
substitute for absolute-magnification calibration. No claim may erase the
distance-magnification or mass-sheet dependence.

## D028 — Proposal versus evaluation population

Phase 2 uses a broad proposal for support and a balanced, literature-informed
benchmark evaluation distribution. Neither is labeled as the inferred GWTC or
strong-lens rate population. Proposal densities and importance weights remain
privileged evaluation fields.

## D029 — Detector-specific synthetic noise identity

Every future detector slot records its exact curve file and SHA-256. The
baseline pins Bilby 2.6.0/LALSuite 7.26.1, H1/L1
`aLIGO_O4_high_asd.txt`, V1 `AdV_psd.txt`, IMRPhenomXPHM, and the supported
SEOBNRv4PHM mismatch. The generic frozen smoke label is retained as immutable
history but is documented as imprecise.

## D030 — Fixed calibration and model-selection budget

The Phase 2 configuration freezes 65,536 training, 8,192 validation, 8,192
calibration, and 16,384 IID cases plus four 4,096-case diagnostic/OOD sets.
Model selection is limited to 12 preregistered flow fits. Coverage tolerances,
SBC size and the 256-case likelihood gold subset are fixed before training.

This paragraph records the historical RC.1 allocation. D033 and adaptive
RC.3/RC.4 supersede its nontraining counts with 6,144 validation, 4,096
calibration-fit, 2,048 SBC-diagnostic and the separately sealed 20,480-case
final pool. The twelve-result model-selection ceiling is unchanged.

## D031 — Phase 3A qualification gate

No full production run follows directly from Phase 2. After human acceptance,
Phase 3A may be separately authorized for 4,096 engineering-qualification
pairs, estimated at 4.83 GB raw. Only its measured throughput, solver
acceptance rate, storage and resume evidence can authorize a larger run.

## D032 — Relative magnification is not an observed flux ratio

The derived target is the selected images' relative absolute macro
magnification, with a separately named relative strain amplitude. Phase 2 does
not create a deployable EM flux-ratio product or identify it with the macro
ratio; doing so would require an explicit variability, extinction,
microlensing and measurement model.

## D033 — Independent calibration evidence

The former 8,192-case calibration split is replaced without changing total
size by 6,144 `calibration_fit` cases and 2,048 `sbc_diagnostic` cases. The
former alone may fit post-hoc correction; 1,024 fixed SBC replicates come from
the latter. Their source, lens, physical-system, pair, noise and augmentation
groups are disjoint.

## D034 — Final evidence cannot tune the method

The validation split supplies a 256-case gold development subset that may
trigger revision. A distinct frozen 256-case IID subset is evaluated once
after method and calibration freeze. Its failure downgrades the claim or
requires a new preregistration; it cannot retune the current version.

Architecture selection uses mean validation negative log probability over all
three seeds. Median is diagnostic, and no individual seed is selected as the
winner. All three selected-architecture seeds are reported.

## D035 — External convergence is connected physics and data

Future alpha.3 scientific records apply the declared mass-sheet transformation
with `lambda = 1 - kappa_ext`: source and Fermat/time differences scale by
lambda, image positions remain invariant, and signed magnification scales by
`lambda^-2`. A deployable environment observation supplies posterior mean and
standard deviation or explicit absence; truth remains privileged. Frozen
alpha.2 smoke records remain readable and unchanged.

## D036 — Kinematics uses a declared forward model

Velocity dispersion is generated with Lenstronomy 1.13.6 Galkin using a
circularized spherical power-law mass model, Hernquist light, an
Osipkov–Merritt anisotropy prior, a one-arcsecond aperture, 0.7-arcsecond
Gaussian PSF and luminosity weighting. External line-of-sight convergence is
not treated as bound stellar mass. Direct inversion of Einstein radius is
forbidden.

## D037 — Diagnostic sets are frozen before results

The misleading `lens_family_ood` and combined waveform/PSD split are replaced
by exact cross-family misspecification, parameter-region OOD, waveform
mismatch and PSD mismatch sets. Four balanced-tail strata use fixed physical
boundaries and priority assignment. No boundary may be selected after model
errors are observed.

## D038 — Publication positioning is an intersection, not a “first” claim

Phase 2.1 positions the work against GOLUM, Bayesian pair identification,
Gravelamps, DINGO/DINGO-lensing, micro/millilensing NPE, multimessenger lens
modeling and LVK catalog searches. The preregistered novelty is the evaluated
intersection of separated galaxy images, GW+EM conditioning, model-conditional
absolute magnification and calibration. Journal targeting does not substitute
for validated results.

## D039 — Phase 3A source-plane density hard stop

Phase 3A cannot invent an executable source-plane measure after RC.2 freeze.
The proposal says uniform in an unspecified solver bounding region conditioned
on multiple images; the evaluation population says uniform in the
multiply-imaged cross-section; the execution prompt requires exact normalized
proposal and evaluation log densities. No bounding limits, cross-section area
definition, caustic/pseudo-caustic convention, numerical method or tolerance is
frozen. Choosing any of these in implementation would alter the scientific
distribution. Microbenchmark and qualification generation therefore stop
before materialization pending a reviewed, versioned preregistration amendment.

## D040 — Exact source-plane density precedes selection

RC.3 resolves the RC.2 execution ambiguity by defining `u=beta/theta_E`
uniformly on the half-open square `[-2.5,2.5)^2`. The normalized density with
respect to angular source coordinates is `1/(25 theta_E^2)`. Proposal and
evaluation share this factor. Lens multiplicity and synthetic detection are
explicit selection events and are not hidden inside proposal normalization.

The primary solver is the residual-validated union of Lenstronomy's analytical
EPL/SIE solver and a deterministic grid search, so shallow-EPL demagnified
central images are not discarded. A finer, wider union is the frozen reference.
A failed support audit blocks Phase 3A; implementation may not change these
tolerances after seeing qualification results.

## D041 — Finite source support is a truncated benchmark

The RC.3 boundary probe showed that steep singular EPL models can remain
multiply imaged at finite square boundaries. RC.4 therefore makes no claim that
`[-2.5,2.5)^2` contains the full multiply-imaged cross-section. It is a finite,
normalized benchmark support. The boundary hard gate is agreement between the
primary and finer reference solver unions, including image multiplicity and
positions. This clarification changes no density formula or finite support.

## D042 — Frozen waveform-boundary failure stops Phase 3A

The exact pre-execution generator commit
`a2b8a02b4631e86c39e1b682e4424ecc2f2c5ca9` failed all four predeclared
8-second waveform fixtures against the aligned 32-second reference and
edge-energy limits. Phase 3A therefore stops before the 32-pair microbenchmark.
The observed thresholds may not be relaxed in place. Any duration, placement,
taper, alignment or numerical-acceptance change requires a separately reviewed
versioned configuration, a new clean generator commit, and repetition of every
pre-execution gate.

## D043 — Waveform publication uses a fixed long-grid construction

RC.5 keeps the scientific product at 8 seconds and 2048 Hz but eliminates the
failed comparison between independently constructed 8-second and 32-second
frequency grids. The canonical response is generated on a 64-second grid,
converted with Bilby's normalized inverse transform, cropped to internal
seconds 56--64, and deterministically edge-conditioned. Selection SNR is
computed from the exact conditioned clean product that is stored.

A 128-second construction is the frozen numerical reference. The reviewed
0.005 relative-difference and outside-crop limits, 0.999 energy-retention floor,
exact zero guards and transform-normalization check are disclosed post-RC.4
choices and may not be changed after RC.5 execution begins.

## D044 — Qualification parallelism is shard-deterministic

The 4,096-pair run assigns each of 32 shards an interleaved global attempt-ID
stream `shard_index + 32 * local_attempt`. Sixteen process workers execute these
streams without sharing Bilby or Galkin global RNG state. Accepted indices and
all scientific/observation seeds are fixed by shard, not scheduling. The same
shard can therefore be reproduced independently, while complete shard
directories remain immutable across interruption and resume.

The 32-pair microbenchmark uses eight analogous process streams and is excluded
from the final dataset. Its measured aggregate throughput is projected to the
reviewed 16-worker qualification configuration; no unmeasured GPU acceleration
or physics-fidelity reduction is credited.

## D045 — Phase 3A qualification data are published and permanently non-scientific

The RC.5 generator commit
`fbcd0616611d9cdf915ef0af030e6061c1be7f59` published exactly 4,096 accepted
pairs in 32 shards. The dataset manifest denies scientific, training,
calibration and test use. These denials are permanent: a later authorization
cannot relabel the Phase 3A artifact as scientific data.

The first three shard hashes were byte-identical across the reviewed
interruption and resume sequence. Full production remains a separate human
decision; passing Phase 3A does not open it.

## D046 — Later scientific scale must be preregistered as a stopping rule

The historical 118,784-pair plan is not automatically authorized by generator
qualification. Human review may replace it with a cumulative
16,384/32,768/65,536 training ladder, fixed disjoint evaluation sets and
predeclared learning-curve/calibration stopping rules. Any replacement needs a
new version and canonical hash before implementation.

Physical-system counts and noise augmentations must be reported separately.
Synthetic OOD evaluation and real-noise/GWOSC/GWTC evaluation must not share an
implicit authorization. The latter remains behind its own future gate.

## D047 — Scale decisions use development evidence only (superseded by RC.2 clarification)

Preregistration `1.1.0-rc.1` replaces the single planned training size with
strictly nested 16,384, 32,768 and 65,536 physical-system rungs. The fixed
6,144-system validation split alone may drive learning-curve stopping and
architecture selection. Calibration-fit, SBC and every final IID/OOD/mismatch
split are inaccessible to scale selection.

The 12,288-system development pool and 20,480-system final pool remain fixed
across rungs. Phase 3A's 4,096 engineering pairs are excluded from all counts
and can never be relabeled.

RC.1 incorrectly presented 16k as a possible final lock and claimed concrete
final accepted IDs could be frozen before materialization. D051 and D053
supersede those details without changing the development-only evidence rule.

## D048 — Data-limited evidence at 65k forces a new preregistration

The probe model, three seeds, paired bootstrap and stopping thresholds are
frozen before training. A gray 16k-to-32k result continues to 65k. Meaningful
improvement or uncertainty at 65k stops the workflow; it does not convert
historical evaluation allocations into training data and does not authorize a
larger rung.

Only after size lock does the four-architecture by three-seed selection run at
one size. No best seed is selected.

## D049 — Proposal efficiency preserves support through an RC.5 mixture (hardened by D052)

A future proposal-v2 candidate must retain a positive 0.2 RC.5 broad-support
safety component and evaluate the exact normalized mixture density. This is the
support argument; the finite two-arm qualification cannot prove absence of
holes.

The future engineering gate remains unauthorized in Phase 3B, and failure or
ambiguity retains RC.5. D052 replaces the preliminary acceptance-or-throughput
criterion and freezes how target correction would be used.

## D050 — Catalog counts and external data are versioned future inputs

A proposed 91-event analysis is not frozen fact. Any real-noise or catalog
phase needs a separate release/product freeze, exact event-list hash, data-
quality and PSD rules, ranking statistic, background, multiple-testing
correction and null-result policy. Synthetic OOD permission cannot open
GWOSC/GWTC access.

## D051 — Sixteen thousand systems are probe evidence, not a final stop

Preregistration `1.1.0-rc.2` classifies the first 16,384 ranked training
systems as `train_16k_probe_subset`. The first executable scale decision needs
both 16k and 32k fits, so generation continues to 32k regardless of the 16k
result. Only 32k and 65k may lock, producing completed scientific totals of
65,536 or 98,304.

Materialization proceeds as Stage A (32k train plus validation), conditional
Stage B (32k more train), and post-lock Stage C (calibration, SBC and final
evaluation). A larger-than-65k rung still requires a new preregistration.

## D052 — Efficient training proposals require target-weighted likelihood

When `q_train` differs from `p_eval`, unweighted conditional NPE would learn
the proposal posterior rather than the declared evaluation-target posterior.
RC.2 therefore requires an unclipped importance-weighted conditional NLP using
full-latent `p_eval/q_train` weights, normalized globally to mean one within
each rung. Weights are privileged provenance and never deployable inputs.

Validation, calibration, SBC and IID are direct target-generative draws. The
future proposal-v2 gate can pass only when the 95% lower confidence bound on
accepted pairs per active hour is at least 2.0; acceptance alone is secondary.
Every proposal component needs a reviewed executable sampler and normalized
density before the unauthorized A/B gate can be opened.

## D053 — Seal generation commitments, not unknowable accepted IDs

Selection and rejection history determine accepted IDs, so they cannot be
listed before materialization. RC.2 instead freezes a deterministic commitment
template containing generator/config identities, seed and attempt namespaces,
accepted-rank rules, counts, distributions, versions, grouping and validators.

Before training, its generator placeholder must be resolved and the commitment
re-hashed. Later accepted IDs and manifests must replay as deterministic
outputs. This preserves test sealing without making a false pre-generation ID
claim.

## D054 — Probe fit reuse and one-noise semantics reduce redundant work

The locked-rung 10-transform, width-256, three-seed probe results are reused in
the twelve-result architecture comparison whenever all training identities
match, limiting new fits to nine. Retraining identical probe fits without a
declared failure is forbidden.

Each independent scientific physical system stores exactly one Gaussian noise
realization. Future augmentation remains unauthorized, cannot cross the parent
split or count as another independent system, and must be frozen identically
across rungs before training.

## D055 — Proposal qualification is 512 pairs per arm and 1,024 total

RC.2 simultaneously described a 512-pair qualification and two 512-pair arms.
No data were generated, but the machine contract was ambiguous. RC.3 freezes
two arms: 512 RC.5 control pairs and 512 proposal-v2 candidate pairs, arranged
as 16 matched blocks of 32 in each arm. The hard future authorization maximum
is exactly 1,024 accepted engineering pairs across both arms.

The two arms have distinct dataset identities, manifests and checksums under
one parent A/B run and comparison manifest. Both are permanently excluded from
science. Prelaunch planning assumes RC.5 throughput for both arms; a future
candidate speedup cannot reduce the resource gate before it is measured.

## D056 — Proposal-v2 RC.1 uses an exact analytic central-source mixture

Phase 3C-0 implements the reviewed 0.2 RC.5, 0.6 wide, 0.1 narrow and 0.1
low-redshift mixture without acceptance conditioning. Wide/narrow source
coordinates are exact truncated normals; low-z uses an exact scaled Beta(1,2)
conditional. The complete density includes every inherited RC.5 latent factor,
angular source Jacobian and conditional normalization.

The 0.2 RC.5 component supplies full support. Stable log-sum-exp, caller-owned
RNG, half-open support and privileged-provenance rules are executable rather
than conceptual. The A/B runner remains dry-run-only.

## D057 — Frozen latent ESS failure rejects proposal-v2 RC.1 before A/B

The 200,000-draw preflight passed density finiteness, mean-weight, support and
deterministic replay checks but achieved only 0.09202 overall relative ESS,
0.11969 for SIE and 0.07433 for EPL. These fail the frozen 0.50/0.40 thresholds.

Proposal-v2 RC.1 is therefore rejected before waveform generation. It may not
be tuned in place or enter the 1,024-pair A/B run. RC.5 remains the qualified
fallback; a new candidate requires a new version and human review.

## D058 — Proposal-v3 anchors 55% of mass to the evaluation target

The new immutable candidate is 0.20 RC.5 + 0.55 evaluation target + 0.25
central-source evaluation target. This guarantees `q>=0.55p`, weight at most
1/0.55 and population relative ESS at least 0.55 overall and per family.

Empirical 200,000-draw ESS was 0.78532 overall and above 0.778 by family, with
zero anchor failures. Passing opens only a human decision about A/B; it does not
authorize pair generation.

## D059 — RC.5 generator qualification is distinct from weighted ESS

The diagnostic RC.5 latent ESS was 0.11776 overall and 0.09484 for EPL. No
retrospective gate is applied, but this evidence prohibits treating engineering
generator qualification as proof of statistically efficient weighted NPE
training. Lens structural mismatch is the dominant diagnostic marginal.

## D060 — A health-validator defect invalidates the Phase 3C-A run identity

The first official matched block completed atomically in both arms, but the
new health validator referenced a nonexistent alpha.3 metadata attribute. The
fail-closed runner stopped with `execution_failed` before further generation.

No throughput inference may be drawn from the retained first blocks. They stay
engineering-only and cannot be combined with a retry. Correcting the validator
requires a new generator commit, parent run ID and two new arm dataset IDs,
followed by full preflight and human review. Stage A remains unauthorized.

## D061 — One failed retry closes proposal A/B and selects direct-target fallback

The Phase 3C-A.1 typed health correction passed its real JSON/Parquet/Zarr
integration path and first matched-block health gate. The new run nevertheless
reached the frozen six-hour control-arm cap after 12 complete blocks per arm
and stopped with `execution_failed`. No bootstrap, post-selection ESS or
proposal decision was computed.

This is the one permitted full retry. The project will not create proposal-v4
or perform a third proposal A/B. All stopped artifacts remain engineering-only
and may not be reanalysed under a newly chosen endpoint.

The publication path instead falls back to direct evaluation-target generation
for scientific training. Because RC.3 names a proposal/weighted training route,
the direct-target route must be frozen in a new preregistration version with
`q_train = p_eval` and unit weights before Stage A execution is authorized.
This is one scientific-contract review, not another proposal microphase.

## D062 — Scientific contracts and engineering releases use separate governance

Preregistration versions are reserved for changes to estimands, target/data
distributions, selection/splits, weighting objectives, model decisions,
calibration/evaluation gates and claims. Field accessors, telemetry, resume,
environment fixes, storage plumbing and test coverage remain patch commits in
one major-phase branch and one final PR.

Future official runs require typed schema APIs, an end-to-end disposable
canary, immutable environment identity and a single release-gate result before
official identities are created. Each major phase permits at most one full
engineering retry; a second execution failure invokes the preregistered
fallback. A target-effective-systems-per-hour metric may be used only when
frozen before data exist, never retroactively on the incomplete Phase 3C-A.1
run.

## D063 — Stage A falls back to direct evaluation-target generation

RC.4 inherits the RC.5/RC.3 scientific target and changes only the training
sampling route. Stage A uses `q_train=p_eval`, exact log weight zero and unit
weight. Ordinary conditional NLP is therefore identical to the previously
frozen weighted objective under unit weights. RC.5 and proposal-v3 are not
authorized for weighted scientific training.

Stage A remains 32,768 train plus 6,144 validation systems. The 16k probe,
development-only stopping and all sealed final-evaluation rules remain
unchanged. This closes proposal optimization as a prerequisite while retaining
a fully target-correct scientific path.

## D064 — Canary precedes identities and one release certificate opens execution

Every future official execution uses typed schema APIs, an immutable dependency
and wheel identity, and a disposable canary before official dataset identities
exist. The Phase 4 canary is exactly 8+8 engineering pairs, uses separate seeds
and qualification split labels, and cannot inspect throughput or ESS.

The single release-gate command returns official parent/train/validation IDs
only when RC.4, the final commit/wheel, the canary hash, disk/PSD checks and the
exact scientific execution authorization all pass. During design it must
remain blocked and return no identities.

## D065 — The direct-target canary passes without opening Stage A

The frozen generator commit and non-editable wheel completed exactly 8+8
engineering canary pairs. The first namespace remained byte-identical across
intentional interruption/resume; q=p, unit weights, typed schema/array health,
cross-namespace grouped-ID disjointness, telemetry fields and independent
checksums passed. No throughput or ESS value was inspected.

This is engineering execution evidence only. It resolves the disposable
canary prerequisite but does not itself authorize scientific materialization,
official Stage A identities or model training. Those still require the single
release gate plus a separate exact 32,768/6,144 human authorization.

## D066 — A ready release certificate opens exact-count Stage A only

Delegated expert review accepted RC.4, the frozen generator wheel, the passed
8+8 canary and conservative resource evidence. A separate authorization opens
only 32,768 direct-target train plus 6,144 direct-target validation systems.
Training, calibration, SBC, final evaluation, real noise and GWOSC/GWTC remain
closed.

The hardened release gate verified the actual wheel artifact, dependency lock,
canary and PSD hashes before deriving the official identities. The run started
under `phase4-stage-a-2be777e727ef-d3a60034bbd6` and remains staging-only until
all 304 shards and full cross-split validation pass. Partial shards and
progress snapshots are never publication evidence.

## D067 — Training software may advance without opening scientific training

Stage A generation and probe-software implementation are separate engineering
workstreams. The training-stack branch may implement and test the lazy reader,
PSD whitening, mask-aware encoders, conditional NSF, checkpoint/resume and
development metrics using synthetic in-memory tensors. It may not read Stage A
staging or execute scientific optimization.

The reader opens only `noisy.zarr`; clean and noise products remain forbidden
model inputs. Policy 1.4 adds the analyst-selected adopted lens-family
hypothesis because the estimand is explicitly model-conditional; it does not
permit any continuous lens truth. Five astrometry items are supported because Phase 3A observed
five-image systems. Censoring is attached only to actually observed astrometry
items, avoiding a truth-multiplicity leak when astrometry is unavailable.

The 16k probe is the lowest SHA-256-ranked half of the complete 32k training
rung, not the first 16k generated. Training remains fail-closed until Stage A
is atomically published, the final-evaluation generation commitment is
finalized and hashed, and a separate scientific probe authorization exists.

Scientific model construction must apply the declared seed before any neural
weights or random flow permutations are initialized. Training order is
addressed by `(seed, epoch)`. Resumable checkpoints bind code, environment,
data manifests, ranked membership, the final-evaluation commitment,
preprocessing state and Python/NumPy/Torch RNG state.

## D068 — Freeze final-evaluation generators before training, keep data sealed

The deterministic generation commitment required before probe training binds
an executable generator, not a placeholder Stage A commit. The implementation
therefore freezes all 15 IID/tail/cross-family/OOD/waveform/PSD contexts, exact
seed namespaces, attempt allocation, streaming validation and atomic sealed
publication before a scientific optimizer can start.

Balanced-tail membership is a recorded priority-conditioned selection on
direct-target accepted systems. Parameter-OOD samplers preserve the exact
parent endpoint conventions, including left-open/right-closed high-shear,
high-convergence and upper-slope intervals. Waveform and PSD mismatch alter
truth generation only and retain their frozen assumed-model identities.

The commitment does not authorize early materialization or inspection. A
future runner requires training-size and architecture lock, a separate
authorization and release certificate, and publishes only to a sealed root.
Final evaluation remains forbidden for stopping, architecture selection and
calibration fitting.

## D069 — Probe execution binds the atomic parent publication and reuses rung preprocessing

Stage A publishes train and validation together by atomically renaming their
common parent. Training therefore resolves both child identities from the passed
parent `dataset_manifest.json`; it does not require nonexistent child dataset
manifests and never reads staging. The later authorization binds the exact parent
manifest hash, Stage A generator, RC.4 hash and finalized evaluation commitment.

Input and target standardizers use deterministic streaming moments over the
authorized rung's metadata. No waveform array is opened during preprocessing and
no full collection of prepared examples is retained. One hashed preprocessing
artifact is reused by seeds 0, 1 and 2. The three seeds may run concurrently on
three isolated GPUs because model initialization, epoch ordering, data membership
and checkpoints remain seed-addressed; architecture and best-seed selection remain
forbidden.

Development evaluation is fixed at 1,024 posterior draws per validation case.
Marginal coverage uses central intervals, while joint coverage uses conditional
posterior-density HPD ranks rather than the intersection of marginal intervals.
The final learning-curve decision uses only the identical 6,144 validation systems
and the frozen 10,000-replicate paired bootstrap. Calibration, SBC and final
evaluation stay closed.

Training order remains a complete deterministic permutation for every seed and
epoch, but is shard-local: shard order and row order within each shard are both
randomized. This prevents lazy Parquet/Zarr access from degenerating into one
store open per example without changing membership or example weights.

The executable AutoDL candidate uses Python 3.10.12, Torch 2.10.0+cu128, Zarr
2.18.3 and Numcodecs 0.13.1. The latter two are the Python-3.10-compatible Zarr-v2
pair already proven by the Stage A writer. A dependency pair requiring Python
3.11 is not an acceptable nominal lock for a Python 3.10 host.

## D070 — Accept the atomic direct-target Stage A publication

Stage A passed with exactly 32,768 train and 6,144 validation systems in 304
complete shards. The parent manifest SHA-256 is
`4f3e6b3a7ca1a995d7a7643c48410e479fb812e4a01ff66537232b9d64bf3314`.
The publication is the sole data parent eligible for the later 16k/32k probe
gate. The 16k membership is still resolved only from the complete 32k train ID
set by the frozen SHA-256 rank rule.

The publication does not itself authorize training. A later authorization must
bind the parent manifest, Stage A generator, finalized evaluation commitment,
training code and wheel, model configuration and normalized CUDA environment.
Calibration, SBC, final evaluation, Stage B and GWOSC/GWTC stay closed.

## D071 — Continue the nested training ladder to 65k

All six authorized 16k/32k probe fits completed under one frozen model,
optimizer, validation set and three-seed contract. The paired 10,000-replicate
bootstrap measured a 16k-to-32k NLP improvement of 0.236314 nat per target
dimension with 95% interval [0.223545, 0.248638]. The upper bound is not below
the 0.01 saturation threshold. Each seed also improved median CRPS by more than
22%, while development coverage and EM-cell conditions did not meet the
all-conditions lock rule.

The preregistered decision is therefore `continue_to_train_65k`. This decision
does not itself authorize Stage B. A separate exact-count gate must add 32,768
new direct-target training systems with group identities disjoint from Stage A,
then train the 65k rung from scratch for seeds 0, 1 and 2. Calibration, SBC and
final evaluation remain sealed. No extension beyond 65k is automatic.

## D072 — Materialize Stage B as an immutable second component

The 65,536-system rung is represented by one atomic reference over the
immutable 32,768-system Stage A train parent and one new 32,768-system Stage B
parent. Existing data are not copied, renamed or regenerated. Stage B uses the
unchanged direct evaluation target, root seed `2026071403`, an independent
attempt namespace and exact unit weights.

Publication requires cross-component pair, source, lens, physical-system and
noise-ID disjointness, plus disjointness from the unchanged validation split.
The combined manifest must not self-authorize training. A later exact gate must
bind both parent hashes, the combined hash, frozen training code/model/environment
and the finalized evaluation commitment before any 65k optimizer starts.

The terminal comparison has no automatic forward rung. Saturation locks 65k;
meaningful improvement or ambiguity stops for a new preregistration.

## D073 — Implement one closed four-architecture selection grid

Architecture selection retains the RC.3 grid exactly: transforms 6/10 by
conditioner widths 128/256, with seeds 0/1/2. The locked-rung 10-transform,
width-256 probe results must supply three of the twelve results; only the other
nine fits are new. The identical probe cannot be retrained without a recorded
failure.

Selection uses mean development-validation NLP across all three seeds and
never chooses a best seed. An exact metric tie is broken by lower trainable
parameter count. Calibration, SBC and final evaluation remain inaccessible,
and the selection result cannot open a later gate by itself. Implementation may
proceed against synthetic fixtures while Stage B runs, but execution requires a
locked 65k decision and a separate identity-bound authorization.

## D074 — Calibrate reported credible regions with split conformal maps

The earlier contracts fixed the calibration/SBC splits and acceptance
criteria, but not the post-hoc map itself. Before those data exist, RC.5 freezes
split-conformal level calibration rather than allowing a method to be chosen
after seeing coverage. For each of the three selected-architecture seeds,
4,096 calibration-fit cases provide central-marginal and joint-HPD inclusion
scores. Finite-sample order statistics produce one global map and eight primary
EM-cell conditional maps.

This calibration changes requested credible-region mass levels; it does not
claim to transform flow samples or define a recalibrated analytic density.
Tail-specific maps are not fitted. Tail coverage later applies the frozen
EM-cell map and is reported honestly.

SBC remains independent: 1,024 deterministically ranked systems from the 2,048
SBC split supply ranks for both targets, their sum, their difference and joint
log-density. Twenty-bin chi-square tests use the exact discrete-rank bin
expectation and Holm step-down familywise alpha 0.01. SBC never fits a map.

## D075 — Separate development-pool materialization from checkpoint inference

Calibration-fit and SBC pools will be materialized only after training size and
architecture lock, as two direct-target, unit-weight, group-disjoint atomic
datasets: 4,096 calibration-fit systems and 2,048 SBC systems. Materialization
does not grant checkpoint access or permission to fit a map or execute an SBC
test. The release gate must bind the complete Stage A and Stage B parent
publications and must reject staging roots.

Selected-checkpoint inference is a later, separately authorized operation. For
each retained model seed it produces immutable score artifacts with the model
seed, architecture, checkpoint hash, publication hash, split and inference
commit embedded in the NPZ. Calibration and SBC use six distinct deterministic
random namespaces. The statistics runner must prove the two physical-system ID
sets are unique and disjoint and reject mixed seed, architecture, checkpoint or
code identities before fitting any map. No best-seed selection is permitted.

## D076 — Make every final cross-family diagnostic executable before data exist

The finalized generator commitment named a SIE-truth cross-family cell whose
analysis assumed EPL at fixed density slope 2.08. The frozen estimator exposes
only the adopted lens-family one-hot and has no fixed-slope model input. That
label therefore described an analysis the deployed model could not perform.
The contradiction was found before final materialization, unsealing or metric
inspection.

RC.6 leaves the generator configuration, commitment, namespaces, seeds, counts
and truth distributions byte-identical. It maps the legacy materialization
context to EPL-family-conditioned inference, which marginalizes the frozen
training EPL slope prior. Family-marginalized contexts are an exact equal-
density mixture of SIE- and EPL-conditioned posteriors, evaluated separately
within each retained model seed. They are model-misspecification diagnostics,
not fixed-slope claims.

Final analysis evaluates all three seeds with no best-seed selection, applies
the matching seed's already-fitted EM-cell calibration map, reports raw counts
and Wilson intervals, and never uses a final case to refit or tune. GW-only and
EM-only ablations are separately trained at the locked architecture/size under
the same budget; the former retains GW timing and the latter removes it. The
matched non-neural and gold likelihood baselines remain a hard pre-unsealing
implementation obligation.

## D077 — Do not claim a full-latent correction from a target-only marginal NSF

DINGO-style importance correction requires a normalized neural proposal on the
same complete latent state used by the likelihood and prior. This project's
selected NSF intentionally outputs only the two log absolute magnifications.
The exact simulator likelihood additionally depends on lens/source structure,
external convergence and dynamics, BBH parameters, orientation and the
selected-image mapping. No full-latent proposal density exists, so the
inherited 10% importance-efficiency statistic is undefined.

RC.7 supersedes that obligation before final materialization and forbids the
terms likelihood-corrected and gold posterior for the selected estimator. It
does not change the model, target, data, split, calibration or evaluation gate.
Independent SBC and IID/tail coverage remain the primary validation evidence.

The matched non-neural comparator is instead a fixed selected-prior EM/timing
simulation reference: exactly 256 same-family, same-EM-cell training neighbors
under a frozen standardized deployable-input distance, followed by a weighted
two-target KDE. It excludes GW strain and is explicitly approximate. Its
finite-bank and distance limitations must be reported; no final result may tune
its neighbor count, distance or kernel.

## D078 — Bind final inference as immutable seed/namespace score jobs

Final evaluation is executed as 45 independently identified jobs: three
retained model seeds times 15 frozen generator namespaces. Every job binds one
selected checkpoint, the same seed's completed calibration/SBC statistics,
the sealed parent manifest, the selected architecture, an immutable inference
environment and a unique output path. A missing or mixed identity fails before
Parquet, Zarr or a checkpoint is opened.

Each case receives exactly 4,096 posterior draws in microbatches no larger
than 512. Draws are retained only for the current physical batch and never
persisted. The immutable output contains per-case score arrays and diagnostic
labels. Final data cannot refit calibration, select a model, change a threshold
or pool model seeds.

The two family-marginalized cross-family cells draw equally from the SIE- and
EPL-conditioned posteriors and evaluate the exact equal-density mixture. All
other cells use the frozen deployable condition. Calibration always uses the
matching model seed and EM cell. Phase 6 statistics summaries must publish
hashes for their calibration map, SBC summary and independent-coverage output
so a downstream gate cannot silently mix products.

## D079 — Accept the exact Stage B extension without opening the optimizer

The authorized direct-target Stage B run completed exactly 32,768 systems in
256 atomic shards and published a combined reference to the unchanged Stage A
train component. The combined manifest validates 65,536 unique train systems,
6,144 unchanged validation systems, exact unit weights and disjoint source,
lens, physical-system, pair and noise identities across components.

Publication is necessary but not sufficient for training. The Stage B
execution result explicitly keeps the 65k optimizer false. A separate gate
must bind both parent manifests, the combined manifest, the finalized
evaluation commitment, model configuration, immutable CUDA environment and a
reviewed training wheel before resolving membership or reading strain arrays.

## D080 — Correct numerical source-waveform pathologies without mutating data

The first 65k launch exposed a finite IMRPhenomXPHM isolated-bin failure before
any optimizer step. Exhaustive source-polarization regeneration found five and
only five failures in 71,680 published records, separated from every retained
record by more than four orders of magnitude. Because selection consumed the
corrupted waveform, training-time clipping or omission would not repair the
scientific sample.

Version `1.1.1-rc.1` therefore makes numerical source-waveform validity an
explicit pre-lensing, pre-selection simulator condition. For plus and cross
independently, the peak strictly positive amplitude at or above 20 Hz is divided
by the linear 99.9th percentile of strictly positive amplitudes. A maximum ratio
above 10 is a rejected attempt. The threshold is frozen from a numerical gap
(largest retained 1.705, smallest failure 12,983) before any 65k optimizer or
downstream result; it is not tuned to model performance.

The original Stage A/B publications remain immutable. A new atomic correction
view removes exactly two plus three affected physical systems and adds exactly
two plus three fresh direct-target, group-disjoint replacements. Validation and
all counts remain fixed. The 16k membership is recomputed and all 16k/32k fits
are rerun; 65k is rerun only if the unchanged stopping rule still requires it.

## D081 — Accept the immutable five-system correction without reviving training

The authorized correction published exactly two Stage A and three Stage B
direct-target replacements under one atomic parent. Independent closeout
reproduced the parent and tree hashes, every shard artifact, original parent
hashes, exact array decomposition, unit weights, numerical-validity criterion
and full grouped-ID disjointness. Corrected counts remain 32,768/32,768 train
with 6,144 unchanged validation systems.

This is an overlay, not an in-place rewrite: the five affected systems remain
in their immutable historical publications and are excluded only by the
versioned corrected views. The earlier 16k/32k metrics remain superseded.
Training can reopen only through a new exact authorization that binds the
correction manifest and a reader proven to include all five replacements and
exclude all five affected systems.
