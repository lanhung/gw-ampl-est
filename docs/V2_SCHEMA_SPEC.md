# V2 logical schema specification

Schema version: `2.0.0-alpha.1`.

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
image, observed event-time difference, preprocessing, PSD, calibration, and DQ
references. Clean/noise arrays are validation artifacts, not model inputs.

### EM observations

`EMObservation` contains noisy image positions, lens center, Einstein scale,
lens/source redshift, velocity dispersion, covariances/standard deviations,
aperture metadata, censoring flags, and a complete availability mask. A missing
modality is `null` and has a false mask; exact truth is never imputed.
Covariances must be finite, symmetric, and positive definite.

### Proposal/evaluation metadata

The proposal log density, evaluation-prior log density, importance weight,
validity, and clipping metadata are nested in provenance. They are evaluation
fields, not deployable features. The schema does not select the final
astrophysical prior.

### Provenance

`Provenance` requires a full generator Git commit, SHA-256 configuration hash,
package/solver/waveform versions, seed hierarchy, detector labels, independent
noise-segment IDs, optional source release, and distribution metadata.

## Artifacts

- `examples/v2_metadata_example.json` is one metadata-only, deliberately
  ungenerated example containing three physical images and a selected pair.
- `examples/v2_metadata_schema.json` is generated logically by
  `v2_json_schema()` and tested for equality.
- `configs/data/v2_smoke.yaml` is an execution-disabled Phase 1B
  specification.

The JSON Schema is intentionally a boundary/shape overview in this alpha
version. Scientific invariants are enforced by typed Python validators and
tests. Before an external producer is supported, nested JSON Schema definitions
must be expanded or a validated code-generated schema library adopted.

## Round trip and versioning

`V2Record.from_dict()` converts enum/list representations to typed objects,
runs every validator, and `to_dict()` returns canonical JSON-compatible values.
Tests require exact field-preserving round trips.

Schema versions use semantic-style identifiers. Patch versions may clarify
validation without changing stored fields. Minor versions may add optional
fields with explicit defaults. Removing/renaming fields or changing units
requires a major version and a pure, tested migration function. Migrations
must preserve the original manifest and record source/target schema versions;
in-place mutation of immutable datasets is prohibited.
