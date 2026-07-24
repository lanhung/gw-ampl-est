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

## D082 — Treat the correction as a typed training view, not a rewritten dataset

Training resolves the correction parent beside the immutable Stage A and Stage B
parents. The reader filters base entries by the five frozen physical-system IDs
and lazily concatenates the corresponding replacement namespaces. It never
copies, edits or reorders files inside a base publication. The derived
training-view hash binds the base manifests, correction manifest, exclusions,
replacement validations and replacement physical-system IDs.

The 16k subset is recomputed by the existing order-invariant SHA-256 rank over
the complete corrected 32k membership. The 65k view uses the same correction
identity and can run only if the fresh 16k/32k comparison again requires it.
Old checkpoints and metrics remain immutable superseded evidence and cannot be
resumed into a corrected run.

## D083 — Reopen only the corrected 16k/32k probe under a new release

Expert review binds the real corrected-view resolution, implementation commit
`adcb1a79e1534e4d742238aa99869c57da95dd96`, exact wheel, unchanged model and
CUDA lock before data access. The rerun recomputes 16k membership from the full
corrected 32k population and fits all six seed/rung combinations from scratch.

The frozen 16k-to-32k paired decision is the only output that may open a later
65k review. The previous learning curve and failed 65k output remain immutable
negative/superseded evidence and are never resumed.

## D084 — Continue only to a fresh corrected 65k probe

All six corrected-view 16k/32k fits completed from scratch with identical code,
model, environment and 6,144-case validation identity. The independently
replayed 10,000-replicate paired bootstrap measured an NLP improvement of
0.211849 nat per target dimension with 95% interval [0.200116, 0.223464]. This
is incompatible with the frozen 32k saturation rule, so the decision is
`continue_to_train_65k`.

The decision opens review, not execution. A new authorization must bind the
corrected combined 65k view, exact decision hash and immutable training release.
The three 65k seeds start from scratch. The pre-correction failed 65k root and
the superseded 16k/32k checkpoints are never resumed. No extension above 65k is
permitted: the terminal result either locks 65k or stops for a new scientific
contract.

## D085 — Bind the terminal probe to the correction overlay

Delegated review authorizes the corrected 65k probe only after binding the
exact 32k continuation decision, corrected combined-view hash, immutable
training wheel and CUDA environment. The three seeds use a new output identity
and start from scratch; the failed pre-correction 65k root is immutable.

The terminal comparison uses the same 6,144 development systems and frozen
10,000-replicate rule. It may lock 65k for later architecture review or stop as
data-limited for a new preregistration. It cannot authorize another training
rung, calibration, final evaluation or external-data access.

## D086 — Carry the correction overlay through architecture selection

Every architecture result must use the same corrected 65k membership that
produced the terminal size decision. Reusing the corrected probe while fitting
the other architectures on the original five-pathology base view would make
the comparison scientifically invalid even if dimensions and counts matched.

The architecture runner now resolves the correction parent explicitly, binds
the corrected train-view hash and reuses the correction-derived preparation
identity. Missing correction paths or a drifted correction/tree/view hash fail
before data access. This is a software-release decision only; it does not open
architecture fitting or selection before the terminal 65k lock.

## D087 — Carry numerical rejection forward without rewriting old commitments

The five-system incident established an IMRPhenomXPHM source-polarization
numerical-validity condition before any calibration, SBC or final-evaluation
system existed. Future generator paths must therefore apply the same frozen
pre-lensing, pre-selection rejection rule rather than rediscovering the defect
after publication.

The Phase 6 data configuration and original final-evaluation commitment remain
byte-identical. Hash-bound prospective addenda bind them to correction
preregistration `1.1.1-rc.1`, and the typed generator builders inject the exact
20 Hz, 99.9th-percentile and ratio-at-most-10 contract. Counts, seeds, target
distributions and diagnostic contexts do not change. The intentionally distinct
SEOBNRv4PHM mismatch namespace is explicitly marked non-applicable to this
IMRPhenomXPHM-specific ratio rule and retains its independent finite-array and
boundary checks. This closes a software consistency gap; it authorizes no data.

## D088 — Revise the future generator identity, not the sealed test design

The original final-evaluation commitment named generator commit `bc02054c...`
before the waveform numerical incident. Requiring that historical code at
materialization would omit the now-frozen rejection rule; mutating the original
commitment would erase the pre-training audit trail.

The release path therefore preserves the original commitment byte-for-byte and
requires a future exact authorization to bind both it and the supplemental
numerical-validity addendum to one new immutable generator release. The only
permitted revision scope is implementation of the frozen numerical rejection;
counts, seeds, attempt streams and diagnostic distributions must be unchanged.

Final group-leakage validation uses the logical corrected training view: it
streams the Stage A/Stage B base data, excludes exactly the five frozen bad
physical-system IDs, includes the five published replacements and verifies the
resulting 65,536-system reference. Official final identities are derived only by
a ready release certificate after training size, architecture and all reference
pools are atomically locked.

## D089 — Use the corrected logical view in every downstream leakage gate

Calibration/SBC records must be group-disjoint from the data actually used for
training and validation, not merely from the two historical base parents. After
the numerical correction those sets differ by five physical systems. Ignoring
the replacement parent could miss a collision with a replacement identity;
counting the excluded base records would describe a superseded training view.

The future Phase 6 release therefore binds all four immutable reference parents
and resolves them through the same typed correction resolver used by training.
Its streaming leakage check skips exactly the five frozen exclusions, includes
the five replacement records, and requires 71,680 distinct reference systems:
65,536 train plus 6,144 validation. The change affects only a future fail-closed
gate and does not authorize materialization or statistics.

## D090 — Carry exact EM-cell identity through the metadata-only reference path

The RC.7 reference is stratified by exact adopted lens family and exact EM
availability cell. The prepared model-input signature is insufficient for this
bookkeeping because distinct frozen cells can share the same available
modalities. Every metadata-only published-record read must therefore attach the
exact Parquet `em_cell` label beside the allowlisted arrays, while keeping it out
of the deployable tensor collator.

The reference bank is built once as a deterministic vectorized index over the
selected training-rung standardizer. Bank identity hashes are order invariant;
queries must be group-disjoint; ties are resolved by physical-system ID. The
future score artifact contains only per-case metrics and identities, never the
4,096 posterior draws. This is an implementation decision under frozen RC.7:
it does not open a bank/query gate or change the distance, 256-neighbor rule,
KDE, metrics or scientific claims.

## D091 — Train input ablations only after the primary model is fully locked

The RC.6 GW-only and EM-only comparisons are separately trained estimators,
not post-hoc masks applied only at final inference. Both use the exact locked
65,536-system corrected training view, the selected primary architecture, all
three retained seeds and the primary optimizer, effective batch and stopping
budget. The primary locked-rung standardizer is applied first; the ablation
then replaces only the preregistered deployable tensors and masks with exact
zeros. Targets, family condition, membership and split identity do not change.

Each input view receives a distinct model-configuration identity while its
base-architecture hash is retained. A six-fit summary requires one shared
training/environment/data/commitment identity, distinct view hashes, all three
seeds and development-only metrics. It never selects a best seed or retunes an
architecture. Implementation can be completed with synthetic fixtures while
the terminal probe runs, but scientific ablation fitting requires a later gate
after both training size and architecture are locked. Calibration, SBC and
final evaluation cannot be used by the ablation training or its development
summary.

## D092 — Execute the RC.7 reference as a bounded metadata-only query job

The selected-prior reference is an offline simulation baseline, not an exact
likelihood and not a neural checkpoint. Its bank must therefore be built only
from the corrected locked scientific training rung through the metadata-only
reader, standardized with the selected primary rung's frozen input
standardizer, and indexed once by exact lens family and exact EM cell.

Each separately authorized query role is streamed one physical system at a
time. The job persists deterministic per-case CRPS, KDE NLP, central marginal
and joint coverage, interval widths and neighbor identities, followed by raw
success counts, rates and Wilson intervals. It never persists the 4,096
posterior draws or opens GW strain. Validation, IID and balanced-tail roles
retain their exact counts of 6,144, 8,192 and 4,096; final roles also require
the independent final-unsealing gate. A future execution authorization must
bind the terminal 65k lock, selected architecture, corrected publication,
primary preprocessing artifact, exact query publication, immutable software
and output identity. The current implementation gate opens none of them.

## D093 — Keep the legacy SIS regressor as a read-only descriptive control

The PDF-era regressor was trained on 2,500 SIS systems with one synthetic ET
detector and reports a model-selected 500-row validation partition. Its saved
point predictions are traceable, but the checkpoint is neither a conditional
posterior nor compatible with the new H1/L1/V1, SIE/EPL, multimessenger input
contract. Applying it to v2 final records would therefore manufacture an
invalid matched comparison.

The executable obligation is limited to a read-only reproduction of the
historical descriptive metrics. The verifier binds the opaque checkpoint and
saved-prediction hashes, never deserializes the checkpoint, recomputes point
MAE/RMSE/MAPE/Pearson and checks the SIS signed-magnification identity. It
records that the rows are model-selected validation rather than independent
test evidence. Legacy inode/size/mtime identity must remain unchanged, and any
new result is written only under `/root/autodl-tmp/lensing-4`. This completes a
historical stress control without calling it calibrated, posterior-valued,
matched or scientifically comparable to the primary final evaluation.

## D094 — Stop after the corrected 65k probe and require a new scientific contract

All three corrected 65,536-system probe fits completed successfully under the
frozen training code, data view, model and CUDA environment. The paired
32k-to-65k development comparison measured an NLP improvement of 0.201437 nat
per target dimension with 95% interval [0.191498, 0.211788]. Even the lower
bound is more than nineteen times the 0.01 saturation threshold. All three
seeds also improved median CRPS, while no seed passed every EM-cell tolerance
and the extreme-relative-magnification development view remained below its
minimum case requirement.

The preregistered result is therefore
`stop_data_limited_and_new_preregistration`, not `lock_train_65k`. It is a
successful terminal measurement, not an optimizer failure. The exact decision
SHA-256 is `90c238a0d85d941c9e90a68e8a215a8d9025f57ffe7757ff89dd14c267f6d72f`
and an independent replay is byte-identical.

No larger rung may be generated automatically. The existing architecture grid
also remains non-executable because its gate requires a locked training size.
A new versioned scientific preregistration must decide the next data scale or a
narrower scientific scope without consulting final evaluation. Calibration,
SBC, final evaluation, real noise and GWOSC/GWTC remain closed.

## D095 — Use one terminal 131k resource cap and an independent development-tail pool

Human review accepted a prospective `1.2.0-rc.1` response to the successful
65k data-limited stop. The corrected 65,536 systems remain an immutable strict
subset and a future exact gate may add 65,536 direct-target systems. The new
terminal size is 131,072; automatic extension above it is prohibited.

The fixed 6,144 validation cases contain only 40 extreme-relative-
magnification examples, so more training data alone cannot satisfy the
inherited 128-case development-tail minimum. A new group-disjoint 512-system
development-only pool therefore contains exactly 128 systems in each frozen
priority tail stratum. It cannot reuse final-evaluation identities or be used
for training, architecture selection, calibration or final claims.

The prospective 65k-to-131k saturation label remains based on paired core-
validation NLP, CRPS and three-seed agreement. Raw coverage and EM-cell
coverage remain mandatory diagnostics but no longer block a terminal
resource-cap lock; calibrated claims remain owned by the already frozen
split-conformal and independent SBC stages. If saturation fails, the project
must report `lock_train_131k_resource_capped_data_limited` rather than pretend
convergence. Either honest terminal label permits later architecture review,
but neither permits another data rung.

## D096 — Freeze one atomic release for the terminal train and development-tail data

The terminal runner reuses the qualified direct-target generator and applies
the frozen numerical-waveform rejection to every new IMRPhenomXPHM attempt. It
publishes the 65,536-system increment and the four tail namespaces separately,
then creates a small logical 131k reference only after streaming validation
proves all group identifiers are disjoint from the corrected 65k view and the
unchanged validation set.

The release is frozen at `a4e6bac014ccd521d510c97593cb1368e826d5eb` and exact
wheel SHA-256 `c7bc8ecadb373ed5d7307ee9c96b131cc68cc9ad8ea10ae2100c54aed2a8958f`.
Its prelaunch free-space threshold is 201,596,510,484 bytes; 221,613,056,000
bytes were observed at review. This roughly 20 GB margin is sufficient under
the conservative measured projection but not large enough to waive the launch
remeasurement or the 100 GB post-peak floor.

The second-image development-tail condition is executed as the already frozen
half-open interval `[10,12)`. The change from the old inclusive endpoint is a
contract-alignment bug fix made before any development-tail data or identity
existed. It does not alter training, selection or final-evaluation counts.

## D097 — Separate terminal scheduling concurrency from scientific identity

The initial terminal run used the frozen 16-worker configuration and was
stopped before its first complete shard when the owner requested greater CPU
use. The 16 partial shards are immutable interruption evidence and are not
reused. Because scheduler concurrency does not enter proposal sampling, shard
seeds, accepted-rank allocation or record identity, changing only the external
process-pool width does not require a new scientific preregistration or dataset
identity.

An independently frozen orchestration layer permits exactly 32 workers while
passing the original namespace configuration to every shard worker. It binds
the unchanged generator, wheel, configuration hash, root seed and official
identities, and records its own commit in release/run evidence. Requests for 64
workers fail before execution: the host has 64 logical CPUs, and consuming all
of them would leave no headroom for the parent, storage, checksums or operating
system. Thirty-two workers approximately doubles the CPU-parallel portion while
retaining a conservative resource margin.

## D098 — Build the terminal probe path while materialization remains immutable

The terminal train extension is CPU-bound and runs independently from the GPU
training software. To remove a later serial delay without opening active
staging, the project implements the 131k publication resolver, bounded-memory
reader, retained-65k tail evaluator, three-seed launcher and terminal comparison
using synthetic fixtures only.

The 65k-to-131k decision differs deliberately from the inherited 32k-to-65k
helper. The same 6,144 core validation IDs own the paired NLP/CRPS saturation
test. Raw marginal/joint and EM-cell coverage and all four 128-case tail strata
are mandatory reports, but they are nonblocking because split-conformal
calibration and independent SBC own later calibrated claims. The result must be
either `lock_train_131k_saturated` or
`lock_train_131k_resource_capped_data_limited`; both lock 131,072, retain all
three seeds and forbid automatic extension.

Implementation readiness is not data or optimizer authorization. A later gate
must bind the completed atomic train/tail/combined manifests, the exact code and
wheel, the frozen model configuration and CUDA environment before the resolver
may open even one scientific record.

## D099 — Adapt architecture selection to the terminal lock without rewriting 65k history

The existing correction-aware architecture runner is preserved for audit
replay. A separate terminal adapter accepts either
`lock_train_131k_saturated` or
`lock_train_131k_resource_capped_data_limited`, requires the exact logical
131k manifest and reuses the three `10 transforms × width 256` probe fits.

Only the remaining three architectures by three seeds may be fitted. All use
the same 131,072 systems, 6,144 validation cases, standardizers, optimizer,
budget and finalized evaluation commitment. Selection remains mean validation
NLP across three seeds with parameter-count tie-break; selecting a best seed,
consulting the 512 tail pool for architecture choice or opening final data is
forbidden. This is software readiness only until an exact post-lock gate binds
the scientific artifacts and environment.

## D100 — Require an explicit terminal-rung binding in every downstream gate

Historical Phase 6/7 score software accepted checkpoints from the originally
reachable 32k/65k locks. Merely broadening that implicit list to 131k would let
an old authorization accept a new scientific identity without recording the
terminal decision. Instead, legacy authorizations remain limited to their old
rungs, and a 131k checkpoint is valid only when the exact later authorization
contains `selected_architecture.locked_training_rung: 131072`.

A common terminal adapter owns the two valid terminal labels, twelve-result
architecture-selection contract and compact publication reference. The adapter
does not execute a decision or open data. It exists so calibration/SBC, final
inference, ablations and non-neural references can share one fail-closed size
identity rather than drifting independently after the terminal lock.

## D101 — Extend downstream leakage references, not frozen evaluation counts

The terminal rung changes which scientific training groups must be excluded
from later calibration/SBC and final pools; it does not change those pools'
counts, seeds or distributions. Future release certificates therefore gain a
`terminal_131k` reference mode while the historical corrected-65k mode remains
available only for audit replay.

Calibration/SBC leakage checks stream 131,072 train, 6,144 validation and all
512 development-tail identifiers before publication. Final materialization
streams the same 131,072 train groups plus its separately published validation,
calibration and SBC references. No final case is unsealed by these checks. A
reference mode cannot be inferred from a checkpoint; it must be explicit and
hash-bound to the terminal size and architecture decisions.

## D102 — Use the locked 131k rung consistently in ablations and references

Once the terminal size and architecture are locked, input ablations must use
the same complete training membership, preprocessing, optimization budget and
three seeds as the selected primary architecture. Continuing to train ablations
on the historical 65k view would confound input removal with sample size and is
therefore forbidden.

The RC.7 non-neural bank likewise uses the selected complete training rung. It
does not use the 512-case development-tail pool, calibration, SBC or final data.
This preserves its role as a selected-prior simulation reference while keeping
the independent tail pool available only for the preregistered 65k-to-131k
development comparison.

## D103 — Independently close out the terminal atomic publication

The official terminal runner validates every namespace before publication and
writes the final execution result only after train, tail and combined parents
are atomic. Acceptance still requires a second read-only closeout path. That
path must bind the generator, worker-32 scheduler, configuration, identities,
counts, unit weights and closed downstream flags through the typed publication
resolver. It must also recompute the train and development-tail tree hashes and
byte counts rather than copying the runner's values.

The large checksum replay is intentionally deferred until the atomic parents
exist. A synthetic-only skip option may test control flow but cannot support
publication acceptance or a later probe-training authorization.

## D104 — Separate terminal probe review readiness from execution permission

Atomic publication and independent closeout are necessary but not sufficient
to start the terminal optimizer. A machine-readable review packet must also
bind the exact Git commit and wheel, a non-editable AutoDL installation test,
the normalized CUDA environment, observed three-GPU inventory, frozen probe
model and final-evaluation commitment.

The packet deliberately cannot authorize execution. It reports readiness for
delegated review only. A later exact authorization must bind its hash, the
training output identity and every publication root before the resolver may
index one scientific record or start one optimizer step.

## D105 — Prove exact-wheel imports without trusting repository pytest paths

The terminal review packet accepts an AutoDL test summary only when a dedicated
verifier proves that `gwlens_mm` came from the installed wheel. A normal pytest
invocation is insufficient evidence because a repository configuration can add
`src` to the import path and silently test checkout code instead.

The verifier therefore binds the installed PEP 610 archive hash to the exact
wheel, rejects editable installs and any module below the repository `src`
tree, and launches pytest with `-c /dev/null`. It exposes only the repository
root for `scripts/` imports. CUDA availability and the frozen RTX 5000 Ada GPU
inventory are recorded in the same atomic result. This remains a software
release check: it reads no scientific publication and cannot authorize or
execute training.

## D106 — Bind terminal training outputs to the separately reviewed release packet

An exact wheel-test JSON is insufficient if a later authorization can omit or
replace the release packet that consumed it. Every future terminal-probe
authorization must therefore carry one repository-relative packet path, its
SHA-256 and an explicit delegated-review acceptance state. The runtime gate
independently loads that packet before resolving any scientific publication.

The packet's publication manifests, counts, training commit, wheel, model,
environment, CUDA inventory, final-evaluation commitment and closed downstream
flags must agree with the execution authorization. The packet itself remains
non-authorizing. Its hash is then persisted in shared rung preprocessing, all
new 131k probe evidence and retained-65k development-tail summaries. This
prevents the optimizer outputs from becoming detached from their exact release
and review evidence.

## D107 — Require a second review object before constructing terminal authorization

The non-authorizing release packet must never create its own execution gate.
Authorization assembly therefore requires a separate machine-readable
delegated-review decision whose SHA-bound scope is exact: one 131,072-system
rung, seeds 0/1/2, retained-65k input root, fresh terminal output root,
publication access, optimizer execution and terminal comparison only.

The closed-boundary mapping must contain exactly the registered model-tuning,
architecture, calibration, SBC, final-evaluation, extension, real-noise and
GWOSC/GWTC keys, all false. Missing or extra keys fail rather than being
ignored. The builder derives atomic publication identities from the packet's
independent closeout, constructs the YAML and passes it through the same
release-binding validator used at runtime before an atomic write. This reduces
manual transcription without allowing the software to infer scientific
approval.

## D108 — Keep terminal release evidence portable across Vultr and AutoDL

The terminal closeout and release packet are produced in a disposable AutoDL
checkout, reviewed and committed in the authoritative Vultr repository, then
consumed again by the AutoDL training checkout. Host-absolute evidence paths
cannot identify the same committed file on both machines and are therefore
forbidden.

Closeout, packet and delegated-review evidence must live below
`results/phase4/` and be referenced by repository-relative paths without
parent traversal. Runtime resolution is always relative to the explicitly
supplied repository root, followed by the existing SHA-256 check. The wheel
itself remains an absolute immutable AutoDL artifact because it is an external
binary input rather than committed evidence. This is a release-engineering
correction only; it changes no publication, model, seed or scientific gate.

## D109 — Permit a clean evidence-only descendant to assemble the release packet

Independent closeout cannot exist in the same Git commit that froze the
training software: it is produced only after the terminal publication
finishes. Requiring packet assembly at the exact training `HEAD` would either
make the closeout unavailable or require an uncommitted evidence file. Both
outcomes are incompatible with the release contract.

The packet script therefore requires the frozen training commit to be an
ancestor of its clean review checkout. Every path changed since that commit
must belong to an exact closeout-evidence allowlist; any software, model,
configuration or unregistered path fails closed. The packet records both the
immutable training commit and the evidence-review checkout commit. Training
still imports only the exact wheel built from the former.

## D110 — Parallelize physical tail shards without changing the tail population

The terminal train increment is already atomic and must not be regenerated.
The initial development-tail execution demonstrated that one rare conditional
128-case shard cannot satisfy the frozen 12-hour per-worker resource cap: after
91,839 attempts it contained only 15 partial cases, and even its 95% upper rate
bound made cap completion effectively impossible.

This is handled as a software-release correction, not a new scientific phase.
The four conditional strata, 128 accepted cases per stratum, direct-target
measure, unit weights, master seed and terminal training cap remain unchanged.
Each namespace is physically partitioned into 32 atomic four-case shards and
uses new parent/dataset/ID-prefix/attempt-namespace identities. The stopped
one-by-128 partial is immutable and excluded. Only the existing 32 physical
workers are authorized; 64-worker oversubscription remains forbidden.

## D111 — Replace fixed tail quotas with deterministic atomic microshards

The reviewed 32-by-4 layout improved aggregate concurrency but retained a
hidden order-statistic failure: every one of 32 attempt streams had to find
four rare cases. The extreme-relative-magnification rate measured from
24,636,731 already published direct-target attempts was only
`1.408466e-5`. The optimistic probability that all fixed streams could finish
by their 12-hour caps was `1.414149e-26`, so continuing was not a credible use
of the frozen resource budget.

The next same-phase implementation uses 128 one-case atomic microshards per
stratum and the existing process-pool scheduler. A completed worker immediately
takes another deterministic shard; accepted index, EM-cell balance, attempt
sequence, root seed and dataset identity remain deterministic within the new
release. No pooled arrival-order assignment or shared RNG is introduced.

This is a storage/work-partition release, not a scientific distribution
change. All four strata remain independently generated from the frozen direct
target with 128 accepted cases each. Both prior failed parents remain
immutable and excluded. The measured projection is about 11.2 million total
attempts; the new resource gate budgets 42 projected hours and a 96-hour hard
cap while retaining the 100 GB post-run free-space floor. Execution stays
closed until an exact commit/wheel/environment authorization is recorded.

## D112 — Use 32 physical workers for the dynamic microshard recovery

The AutoDL host exposes 64 logical CPUs but only 32 physical cores. At the
execution review it had about 61 GB available memory and about 140 GB free
disk, only 15 GB above the frozen prelaunch floor. Running 64 generator
processes would consume every logical CPU, remove operating-system and I/O
headroom, and approximately halve the memory available per worker without a
measured throughput guarantee.

The official dynamic scheduler therefore uses exactly 32 processes. This is
already twice the former 16-worker allocation and, unlike the failed fixed
layout, immediately gives each free process another deterministic one-case
microshard. Sixty-four workers remain fail-closed. This decision changes only
resource scheduling; all four tail strata, seeds, accepted counts, target,
weights and terminal 131,072-system cap remain frozen.

## D113 — Hash-bind the retained 65k checkpoints before terminal comparison

The terminal comparison evaluates the already completed corrected-65k probe
on the new development-tail pool. A reviewed output-directory name is not an
immutable checkpoint identity: files below that directory could drift without
changing the directory path.

The terminal release packet therefore validates and records the SHA-256 of all
three retained `best.ckpt` and `run_summary.json` files. It also binds their
common training manifest, validation manifest, membership, standardizers,
model, environment, training commit and finalized-evaluation-commitment
identity. Authorization copies this exact mapping, the release-binding gate
requires packet equality, and the runtime checks each hash before `torch.load`
then requires the checkpoint's embedded identity to equal the bound summary.
This is an execution-integrity correction and changes no model, data, metric or
scientific stopping rule.

## D114 — Disable repository conftest files in exact-wheel verification

Disabling the repository pytest configuration is not sufficient to isolate an
installed wheel. The maintained `tests/conftest.py` independently prepends the
checkout's `src` directory to `sys.path`, so an otherwise passing verifier
could silently test source files that are not present in the wheel runtime.

The exact-wheel test command therefore uses both `-c /dev/null` and
`--noconftest`. The repository root remains on `PYTHONPATH` only for maintained
`scripts/` imports. A subprocess regression fixture contains a conftest that
raises on import and proves it is not loaded. Wheel installation also uses a
PEP 610 `file:` URL with the frozen SHA-256 fragment. This is a release-integrity
correction made before publication, authorization or optimizer execution; it
changes no model, data, training objective or scientific rule.

## D115 — Accept the atomic terminal 131k reference, but keep training closed

The dynamic 128-by-1 tail layout completed all four frozen 128-case
development strata and independently closed out without reusing either failed
tail parent. The train increment remains the immutable 65,536-system
publication generated at commit `a4e6bac...`; the combined logical reference
now binds it with the corrected 65,536-system base for exactly 131,072 unique
train systems. The validation set remains exactly 6,144 systems and the 512
tail cases remain development diagnostics only.

The accepted closeout identities are the combined manifest
`ad26d51d...5cee5`, tail manifest `58fcafd5...60c7`, train tree
`0ab14492...a8ed` and tail tree `90ca582f...19c0`. Proposal equals evaluation,
all importance weights are one, no failed evidence was reused and no
GWOSC/GWTC product was accessed.

Atomic publication is necessary but not sufficient to start the optimizer.
The terminal-probe gate must separately bind the committed closeout evidence,
exact installed wheel, CUDA environment, model configuration, finalized
evaluation commitment and all three retained corrected-65k checkpoint hashes.
Architecture selection and every downstream scientific gate remain closed.

## D116 — Authorize the terminal probe only after exact-wheel and checkpoint binding

The post-publication release packet was assembled from a clean evidence-only
descendant of the frozen training commit. It binds the independently closed
131k publication, the exact non-editable wheel, four identical RTX 5000 Ada
devices, the normalized CUDA environment, model configuration, final-evaluation
commitment and the summary/checkpoint hashes for all three retained corrected-
65k fits.

The exact wheel passed 68 focused tests with one optional skip and 482 full
tests with six optional skips. A separate delegated-review JSON then approved
only one training rung, seeds 0/1/2 and the terminal comparison. The generated
authorization keeps model tuning, architecture selection, calibration, SBC,
final evaluation, real noise, GWOSC/GWTC and extension above 131,072 false.

Both allowed terminal decision labels stop at 131,072. A scientific
architecture fit therefore cannot begin automatically after the probe; it
requires a new exact gate bound to the terminal decision and all twelve
development-only architecture results.

## D117 — Treat the singular terminal validation manifest as reader compatibility

The first terminal reader invocation failed before preprocessing because the
new train-increment publisher records its one child validation under
`validation`, while the inherited generic reader accepted only the Stage A
multi-child `validations` mapping. The parent manifest itself was valid,
hash-bound and independently closed out.

The reader now accepts either an unambiguous plural mapping or one singular
mapping whose `dataset_id` equals the child directory. If both forms exist and
differ, it fails closed. This correction changes no publication byte,
membership, model tensor, optimizer, target, seed or stopping rule. The
previous training authorization is superseded before any optimizer step; the
new wheel and release packet use fresh identities.

## D118 — Bind the terminal architecture phase through a reviewed release packet

The terminal 131k architecture adapter must not infer execution authority from
a successful learning-curve decision. A non-authorizing release packet first
binds the exact terminal decision, all three reused probe summaries and best
checkpoints, the frozen architecture grid, candidate model hashes, exact wheel,
CUDA environment and a fresh output root. A separate delegated-review JSON is
then required to create the runtime authorization.

The runtime now hashes each reused `best.ckpt` before accepting it; existence
alone is insufficient. This change affects release integrity only. The model
grid, training data, optimizer, validation-only selection rule and terminal
resource cap are unchanged. Until the future packet is reviewed, the nine new
fits, architecture selection and every downstream scientific action remain
closed.

## D119 — Bind Phase 6 materialization to both terminal decisions

Calibration/SBC materialization must not infer authority from the existence of
an architecture-selection JSON. A non-authorizing packet now validates and
hash-binds the terminal 131k size decision, the twelve-result development-only
architecture decision, the exact direct-target generator wheel and environment,
the immutable Stage A/Stage B/correction/terminal publication manifests and the
frozen 4,096/2,048 split counts.

A separate delegated-review document is required to create the materialization-
only authorization. That authorization permits generation and atomic
publication of exactly 6,144 direct-target development systems while keeping
checkpoint access, calibration fitting, SBC statistics, final evaluation,
model tuning and GWOSC/GWTC false. Official identities remain prospective until
the runtime release certificate passes on AutoDL.

## D120 — Separate Phase 6 score extraction from calibration/SBC statistics

The selected architecture retains all three model seeds, so downstream
calibration cannot be implemented as one best-seed job. The execution chain is
therefore split into two independently reviewed releases.

The first release binds the locked 131,072-system architecture, all three exact
`best.ckpt` and run-summary hashes, the atomic 4,096-system calibration-fit and
2,048-system SBC publication, a fixed inference configuration and six fresh
score paths. It may create exactly one calibration-fit and one SBC score
artifact per model seed. It may not fit a calibration map or run an SBC test.

Only after all six artifacts are complete and hash-validated may a second
release authorize three seedwise calibration maps and three independent SBC
analyses. The same physical development cases must be scored by every seed,
calibration and SBC IDs must remain disjoint, maps are never pooled across
seeds, and no best seed is selected. Checkpoint access closes before the
statistics step; final evaluation, retraining and GWOSC/GWTC remain closed
through both releases.

## D121 — Bind final-evaluation children to atomic parent manifests

The scientific publications use one atomic parent manifest for one or more
dataset children. Requiring every child directory to duplicate
`dataset_manifest.json` was inconsistent with the real Stage A, Stage B,
correction and calibration/SBC layouts and would have blocked the final
materialization preflight.

The final-evaluation release now represents each child by its dataset ID and
root plus the exact parent root and parent-manifest SHA-256. The execution
runner revalidates that binding before streaming record identifiers. The
terminal train view contains exactly five child roots: Stage A train, Stage B
train, the two correction-replacement children and the terminal increment.
Validation, calibration-fit and SBC each bind one disjoint child. The five
superseded waveform-pathology IDs remain explicit train exclusions.

A non-authorizing packet binds this catalog to the terminal size decision,
twelve-result architecture decision, exact generator wheel/environment and
the frozen 20,480-case sealed contract. A separate delegated review is still
required before materialization. No final identity, pair, checkpoint or
scientific result is created by this engineering correction.

## D122 — Require one reviewed identity set for all 45 final score artifacts

Final evaluation retains all three selected model seeds and contains fifteen
frozen generation namespaces. The final inference handoff therefore cannot be
authorized as one best-seed run or as independently assembled per-namespace
commands.

A non-authorizing release packet now binds the sealed parent manifest, all
three selected `best.ckpt` hashes, the exact terminal architecture decision,
three same-seed calibration/SBC result bundles, immutable inference
wheel/environment and 45 fresh output paths. The output arithmetic is fixed as
15 namespaces times seeds 0, 1 and 2. Posterior draws remain transient; only
bounded score artifacts may be persisted.

A separate delegated-review JSON is required before one runtime authorization
can unseal final records, load checkpoints or apply calibration maps. The
authorization keeps retraining, calibration refitting, architecture/size
selection, threshold changes, ablation/reference execution and GWOSC/GWTC
closed. This release-control implementation used synthetic fixtures only.

## D123 — Require one reviewed release for the six terminal input ablations

The two RC.6 input ablations are scientific fits, not informal views of the
primary checkpoint. Their future execution must therefore be released as one
exact six-fit set: GW-only and EM-only, each for seeds 0, 1 and 2.

A non-authorizing packet now binds the terminal 131k decision, twelve-result
architecture lock, selected primary model hash, primary 131k membership and
standardizers, immutable training wheel/environment, finalized evaluation
commitment and six fresh output paths. The selected architecture, optimizer,
effective batch and budget are unchanged. Four data-loader workers per fit and
at most three concurrent fits are explicit engineering settings.

A separate delegated-review JSON is required to create the runtime
authorization. No best seed may be selected, no ablation architecture may be
tuned, and calibration/SBC/final evaluation cannot be opened by this release.
The implementation used synthetic fixtures only.

## D124 — Bind each RC.7 query to its real atomic publication layout

The future reference runner must not invent a `dataset_manifest.json` inside a
publication child. Validation, calibration and final datasets are atomic
children of a common parent, and only that parent owns the publication
manifest. Each reference-query release therefore binds the exact child
directory, parent directory and parent-manifest SHA-256.

The frozen balanced-tail diagnostic is not one 4,096-case child. It consists
of four independently generated 1,024-case children, one for each priority
stratum. The runner validates all four parent bindings, concatenates only their
metadata views and rejects duplicate identities before scoring.

Every validation, IID or balanced-tail query receives its own non-authorizing
release packet and separate delegated review. Validation cannot unseal final
data; IID and balanced-tail cannot execute without an explicit final-unsealing
scope. The implementation creates no scientific authorization and preserves
RC.7's prohibition on exact-likelihood, gold-posterior and importance-sampling
efficiency claims.

## D125 — Release the legacy SIS control as a read-only descriptive reproduction

The inherited SIS point-regression checkpoint is an out-of-domain descriptive
stress control, not a matched competitor and not a calibrated posterior. Its
future reproduction must therefore bind the exact checkpoint and saved
validation-prediction hashes without deserializing the checkpoint or applying
the model to v2 scientific data.

A non-authorizing packet now binds the implementation commit, exact wheel,
environment lock, frozen legacy paths and hashes, descriptive metric contract
and one fresh evidence path below the new project root. A separate delegated
review is required to create the runtime YAML. The runtime gate permits only
legacy byte reads and saved-prediction metric reproduction; it forbids legacy
writes, scientific/final data access, manuscript-claim finalization and
GWOSC/GWTC access.

The verifier records checkpoint and prediction inode, size and mtime before
and after use, rejects any identity change and never loads the checkpoint.
This preserves the legacy roots while still making the narrow historical
comparison reproducible.
