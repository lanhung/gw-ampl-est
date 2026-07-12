# Grouped split policy

## Fixed logical splits

The v2 schema enumerates:

- `engineering_smoke`, which is never a scientific split;
- `train`;
- `validation` for model selection;
- `calibration_fit` for post-hoc calibration fitting only;
- `sbc_diagnostic` for independent SBC and calibration diagnostics;
- `iid_test` untouched by model selection and calibration;
- `balanced_tail_diagnostic`;
- `cross_family_misspecification_test`;
- `parameter_region_ood`;
- `real_noise_test`;
- `waveform_mismatch_test`;
- `psd_mismatch_test`.

Calibration-fit, SBC diagnostics and IID testing are mutually separate. SBC
realizations never fit a calibration map. Validation gold cases may guide
development; final IID gold cases are report-once and never trigger retuning.
Diagnostic/OOD sets do not replace the IID test.

Records assigned to `engineering_smoke` are excluded from training,
calibration, IID/OOD evaluation, and reported performance. Their dataset
manifest must additionally set `scientific_use_authorized=false`.

## Grouping invariants

No identifier may cross logical splits for:

- source realization;
- lens realization and physical lens system;
- selected pair;
- every non-null image-detector noise segment;
- augmentation parent.

Multiple selected pairs from one physical system must stay in the same split.
Augmentations inherit their parent's split. Every available H1/L1/V1 slot for
both images has a detector-qualified segment ID, and none may cross splits.
Unavailable null slots do not create fake grouping IDs.

In particular, `calibration_fit` and `sbc_diagnostic` must be disjoint in
source, lens, physical-system, pair, every non-null noise segment, and
augmentation-parent identifiers.

`validate_grouped_splits()` fails with the identifier and conflicting split.
It also rejects duplicate pair rows. Tests deliberately corrupt source, lens,
system, noise, and augmentation groups.

## Assignment procedure

Future generation must assign groups from stable IDs before row materialization
and record the split in the manifest. Row-wise random splitting is prohibited.
The mapping and root seed are frozen with the dataset configuration. Creating
an OOD split is a population-design decision, not a later reassignment of
inconvenient test cases.
