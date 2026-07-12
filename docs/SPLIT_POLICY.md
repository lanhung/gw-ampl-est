# Grouped split policy

## Fixed logical splits

The v2 schema enumerates:

- `train`;
- `validation` for model selection;
- `calibration` for post-hoc calibration only;
- `iid_test` untouched by model selection and calibration;
- `balanced_tail_diagnostic`;
- `lens_family_ood`;
- `parameter_region_ood`;
- `real_noise_test`;
- `waveform_psd_mismatch_test`.

Calibration and IID testing are separate. Diagnostic/OOD sets do not replace
the IID test.

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

`validate_grouped_splits()` fails with the identifier and conflicting split.
It also rejects duplicate pair rows. Tests deliberately corrupt source, lens,
system, noise, and augmentation groups.

## Assignment procedure

Future generation must assign groups from stable IDs before row materialization
and record the split in the manifest. Row-wise random splitting is prohibited.
The mapping and root seed are frozen with the dataset configuration. Creating
an OOD split is a population-design decision, not a later reassignment of
inconvenient test cases.
