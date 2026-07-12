# Lens solver interface

## Separation of responsibilities

`LensSolver.solve()` maps source position and lens parameters to one
`LensSystemSolution` containing every physical image returned by the solver.
Pair selection occurs later and produces a separate `SelectedPair`.

This permits, for example:

```text
physical system: image_0, image_1, image_2, image_3
selected GW pair: image_0 + image_2
unselected: image_1
censored below threshold: image_3
```

No solver adapter may discard physical images merely because the downstream
GW representation has two selected slots.

## Required solver output

Each `PhysicalImage` provides:

- stable image ID;
- two-dimensional angular position;
- signed magnification;
- dimensionless Fermat/arrival-time coordinate or documented arrival time;
- parity and Morse class;
- validity flag and reason.

The system result provides lens family, solver name/version, validity, and all
images. Image IDs must be unique and a valid strong-lensing result contains at
least two images.

## Pair selection contract

`SelectedPair` records two distinct physical image IDs, the explicit primary
definition, selection reason, per-detector visibility, unselected images, and
censored images. Validation rejects unknown IDs, overlapping status labels,
selected images repeated as extras, or any non-selected physical image without
exactly one unselected/censored status. Schema validation also checks earliest,
brightest, and minimum primary claims against physical-image truth; catalog
anchor deliberately has no physical ordering constraint.

## Implemented and planned adapters

- `SISSolver` is fully dependency-free and implements the analytic control.
- SIE plus external shear adapters must report
  `LensFamily.SIE_EXTERNAL_SHEAR`.
- elliptical power-law plus external shear adapters must report
  `LensFamily.EPL_EXTERNAL_SHEAR`.
- external convergence or a model-discrepancy parameter is passed explicitly,
  never folded invisibly into a magnification.

`validate_solver_contract()` checks family consistency, finite arrival
coordinates, unique IDs, and valid images. It is not a replacement for
physics comparison against trusted reference configurations. Phase 1B must
add deterministic reference fixtures for the selected external solver before
generation.

## SIS details

`SISSolver` uses angular `theta_E` and source radius `beta`. The double-image
domain is `0 < |beta|/theta_E < 1`. Exact alignment is an Einstein ring and is
rejected rather than coerced to two images. Images are returned in increasing
arrival-time order: plus/minimum, then minus/saddle.
