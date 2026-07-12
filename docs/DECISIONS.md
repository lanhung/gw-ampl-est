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
