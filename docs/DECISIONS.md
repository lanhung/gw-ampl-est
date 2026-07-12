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

## D039 — Exact source-plane density precedes selection

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

## D040 — Finite source support is a truncated benchmark

The RC.3 boundary probe showed that steep singular EPL models can remain
multiply imaged at finite square boundaries. RC.4 therefore makes no claim that
`[-2.5,2.5)^2` contains the full multiply-imaged cross-section. It is a finite,
normalized benchmark support. The boundary hard gate is agreement between the
primary and finer reference solver unions, including image multiplicity and
positions. This clarification changes no density formula or finite support.

## D041 — Waveform publication uses a fixed long-grid construction

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
