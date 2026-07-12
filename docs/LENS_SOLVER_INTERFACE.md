# Lens solver interface

## External convergence in future alpha.3 data

`apply_mass_sheet_transform` is the Phase 2.1 reference convention for a
line-of-sight convergence sheet. For `lambda = 1 - kappa_ext`, it preserves
image positions, parity and Morse class; multiplies source coordinates,
dimensionless Fermat differences and physical delay differences by `lambda`;
and divides signed magnifications by `lambda**2`. Absolute magnification and
strain amplitude follow from the transformed signed value.

The transformation is applied to a complete baseline lens solution, not merely
stored as truth metadata. Its analytic contracts are mandatory in Phase 3A.
The bound stellar-dynamics model excludes the line-of-sight sheet from its
mass profile; environment data constrain it through a separate deployable
observation.

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
- optional dimensionless Fermat potential with explicit unitless naming;
- optional physical arrival-time delay in seconds with explicit units;
- parity and Morse class;
- validity flag and reason.

The system result provides lens family, solver name/version, validity, and all
images. At least one of the two time coordinates is required and every supplied
value must be finite. Image IDs must be unique and a valid strong-lensing result
contains at least two images. A solver result must provide one common coordinate
across all images so ordering can be checked without inspecting the solver name.

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

`validate_solver_contract()` checks family consistency, explicit common
ordering coordinates, sorted output, unique IDs, and valid images. It is not a
replacement for physics comparison against trusted reference configurations.
The Lenstronomy adapter supplies both the raw dimensionless Fermat potential
and an earliest-normalized physical delay in seconds. The analytic SIS control
supplies the dimensionless coordinate only; any physical SIS time scale belongs
in a separate generator model and must never be inferred from the solver field
name.

## SIS details

`SISSolver` uses angular `theta_E` and source radius `beta`. The double-image
domain is `0 < |beta|/theta_E < 1`. Exact alignment is an Einstein ring and is
rejected rather than coerced to two images. Images are returned in increasing
arrival-time order: plus/minimum, then minus/saddle.
