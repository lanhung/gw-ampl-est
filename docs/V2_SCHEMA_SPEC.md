# V2 logical schema specification

Current scientific schema version: `2.0.0-alpha.3`.

Frozen Phase 1B engineering schema version: `2.0.0-alpha.2`.

Alpha.1 contained metadata examples only and was never materialized as a v2
dataset. Alpha.2 supersedes it without a physical data migration.

Alpha.3 adds an optional-valued but explicitly masked environment observation
for external convergence. It is the first schema for future scientific data.
The immutable alpha.2 smoke dataset is not migrated or regenerated; the reader
retains alpha.2 compatibility and round-trips it without inserting alpha.3
fields.

Phase 1B adds `engineering_smoke` as an explicit non-scientific split value.
It prevents engineering records from being mistaken for training or
evaluation data; the dataset manifest independently denies scientific use.

## Design boundary

The schema describes logical metadata and references to arrays. It does not
embed strain or repeat a full time vector per pair. Time sampling is recovered
from segment start, sample rate, and sample count.

The physical lens system and the selected observed GW pair are separate
objects. A physical system may contain more than two images; exactly two image
IDs are selected for the current pair representation, while all other images
remain in lens truth as unselected or censored.

## Record sections

### Pair index

`PairIndex` stores pair/source/lens/system IDs, selected image IDs, explicit
primary definition, split, lens family, proposal/evaluation-prior IDs, root
seed, dataset version, and optional augmentation parent. IDs are grouping and
provenance metadata, never deployable inputs.

### Source truth

`SourceTruth` stores physical luminosity distance, intrinsic and extrinsic
parameters, and waveform name/version. It is inaccessible to deployable
inference except where a separately defined observation product exists.

### Lens and image truth

`LensTruth` stores family parameters, external shear, external convergence,
model discrepancy, all `ImageTruth` records, and extra-image status. Each image
stores position, signed/absolute magnification, amplitude factor, arrival time,
parity, Morse class, and physical detectability. Validators check redundant
quantities for consistency.

### GW observations

`GWObservation` stores references to exactly three separate products:

- noisy strain;
- clean injected signal;
- noise realization.

Every reference has logical shape `(2 selected images, 3 detector slots,
samples)`. Detector slots are always `H1, L1, V1`; availability is a separate
`(2, 3)` mask. Metadata stores sample rate/count, one start time per selected
image, preprocessing, PSD, calibration, and DQ references. The observed event
time difference is a typed product with value, uncertainty, measurement method,
and optional reference. Ordinary observations require positive uncertainty;
zero is reserved for an explicitly declared deterministic control.

Noise provenance is an image-major six-slot grid over primary/secondary and
H1/L1/V1. Available slots require distinct nonempty detector-qualified segment
IDs and an explicit source; unavailable slots require a null ID and source
`unavailable`. The grid must agree element-by-element with the detector mask.
Synthetic IDs are provenance identifiers, not GWOSC segment claims.

### EM observations

`EMObservation` contains image-ID-keyed `ImageAstrometryObservation` records,
lens center, Einstein scale, lens/source redshift, velocity dispersion,
an `ExternalConvergenceObservation`, uncertainties, aperture metadata,
censoring flags, and a complete availability mask. The environment observation
stores posterior mean, positive posterior standard deviation, inference method
and optional reference. It never exposes `LensTruth.external_convergence`.
Astrometry may include selected and non-selected physical images; tuple order
never defines identity. IDs are unique and must exist in lens truth. A missing
modality is `null` with a false mask; exact truth is never imputed. Covariances
must be finite, symmetric, and positive definite.

Einstein radius and velocity dispersion must be positive; redshifts must be
nonnegative. `redshift_ordering_valid` records whether point summaries satisfy
`z_l < z_s`. It is a quality flag, not a hard rejection, because broad
photometric posteriors can overlap for a physically valid system. Later
likelihood code must use the observation model rather than treat this flag as
truth.

### Proposal/evaluation metadata

The proposal log density, evaluation-prior log density, importance weight,
validity, and clipping metadata are nested in provenance. They are evaluation
fields, not deployable features. The schema does not select the final
astrophysical prior.

### Provenance

`Provenance` requires a full generator Git commit, SHA-256 configuration hash,
package/solver/waveform versions, seed hierarchy, detector labels,
image-detector noise references, optional source release, and distribution
metadata.

## Artifacts

- `examples/v2_metadata_example.json` is one alpha.3 metadata-only,
  deliberately ungenerated example containing three physical images, a
  selected pair and an environment observation.
- `examples/v2_metadata_schema.json` is generated logically by
  `v2_json_schema()` and tested for equality.
- `configs/data/v2_smoke.yaml` is an execution-disabled Phase 1B
  specification.

The generated JSON boundary defines astrometry, timing, and detector-noise
objects and explicitly rejects removed alpha.1 positional, bare timing, and
image-only noise fields. Cross-record scientific invariants remain enforced by
typed validators and tests.

## Round trip and versioning

`V2Record.from_dict()` converts enum/list representations to typed objects,
runs every validator, and `to_dict()` returns canonical JSON-compatible values.
Tests require exact field-preserving round trips.

Schema versions use semantic-style identifiers. Alpha.3 does not redefine or
mutate the materialized alpha.2 engineering artifact. Readers accept both and
serialize according to the record's original version. Future scientific data
start directly at alpha.3. Migrations preserve original manifests and record
source/target versions; in-place mutation is prohibited.

## Cross-record invariants

- every non-selected physical image is exactly one of unselected or censored;
- earliest, brightest, and minimum primary definitions are checked against
  truth with tight equality tolerances; catalog anchor adds no ordering claim;
- astrometry IDs belong to the physical image set;
- alpha.3 environment availability equals presence of the typed environment
  observation, whose uncertainty is positive;
- detector-noise availability equals the GW mask;
- unavailable noisy/clean/noise slots are exactly zero-filled;
- available slots satisfy `noisy = clean + noise` within tolerance;
- NaN is never the routine missing-detector representation.
