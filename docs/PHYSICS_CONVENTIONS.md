# Physics conventions

## Scope

This document is the authoritative vocabulary for v2. Absolute magnification
is always conditional on a lens model and its electromagnetic constraints.
GW observations alone primarily constrain relative image information and an
apparent luminosity distance.

## Magnification and distance

For a physical image `i`:

- `mu_signed_i` is the signed geometric-optics magnification;
- `mu_abs_i = abs(mu_signed_i)` is the flux/power magnification;
- `amplitude_factor_i = sqrt(mu_abs_i)` multiplies GW strain;
- `apparent_luminosity_distance_i = physical_luminosity_distance /
  amplitude_factor_i`.

The sign is not an amplitude sign. It identifies image parity and contributes
through the Morse phase. Signed magnification, absolute magnification, and the
strain-amplitude factor must never share one field.

For an explicitly ordered selected pair:

```text
relative_flux_magnification = mu_abs_secondary / mu_abs_primary
relative_strain_amplitude = sqrt(relative_flux_magnification)
```

The v2 name `mu_rel` is forbidden because it does not identify numerator,
denominator, or whether the quantity is a flux or amplitude ratio.

## Image identity and ordering

Physical images have stable IDs independent of array slots. A selected GW
pair names a `primary_image_id`, a `secondary_image_id`, and one explicit
`primary_definition`:

- `earliest_arriving`;
- `brightest`;
- `minimum_image`;
- `catalog_anchor`.

These definitions may coincide for a simple SIS control but are not
interchangeable. The primary/secondary ordering of one selected pair does not
discard other physical images. Extra images remain present and are marked
unselected or censored.

Metadata validation enforces the declared semantics: earliest means primary
arrival time is not later, brightest means primary `mu_abs` is not lower, and
minimum means the primary Morse class is minimum. Equality uses `1e-9` seconds
absolute tolerance for arrival time and `1e-12` relative/absolute tolerance
for magnification. Catalog anchor deliberately imposes neither ordering.

Arrival times are physical times in seconds when a dimensional model is
available. `signed_time_delay = arrival_time_secondary -
arrival_time_primary`; its absolute value is a separately derived quantity.
The corresponding deployable timing input is not this truth difference: it is
a `TimingObservation` with value, uncertainty, measurement method, and optional
reference. Ordinary observations require positive uncertainty. Zero is allowed
only for an explicitly labeled deterministic analytic control.

## Parity and Morse class

Parity is `positive` or `negative`, determined by the sign of the lens-map
Jacobian and therefore by `mu_signed`. Morse class is represented by a typed
enum:

| Class | Half-integer index | Integer index | Positive-frequency factor |
|---|---:|---:|---:|
| minimum | 0 | 0 | `1` |
| saddle | 1/2 | 1 | `-i` |
| maximum | 1 | 2 | `-1` |

The canonical stored label is the class name. Conversion properties expose
the two numerical conventions so external libraries cannot silently impose a
different encoding.

## Fourier convention

The project uses

```text
H(f) = integral h(t) exp(-2*pi*i*f*t) dt.
```

A positive delay `Delta t` means `h(t-Delta t)` and multiplies the frequency
domain signal by `exp(-2*pi*i*f*Delta t)`. For positive frequency, an image of
Morse index `n` receives `exp(-i*pi*n)`. Negative frequencies receive the
complex conjugate factor, preserving conjugate symmetry and real time-domain
strain. The DC factor is real.

## Coordinates and lens quantities

- source-plane positions use explicit `source_position_*` names and stated
  angular/dimensionless units;
- image-plane positions use `position_arcsec` unless explicitly transformed;
- `einstein_radius_arcsec` is an angular scale, not an exact deployable input;
- external shear is stored as two components;
- external convergence is a distinct nuisance parameter;
- lens family uses `sis`, `sie_external_shear`, or `epl_external_shear`.

Exact source coordinates, lens truth, and magnifications belong to truth or
diagnostic roles and are prohibited as deployable neural inputs.

## SIS analytic control

For `0 < y < 1`, the adopted plus/minus solution is:

```text
mu_plus_signed  = 1 + 1/y
mu_minus_signed = 1 - 1/y
r_flux = abs(mu_minus_signed) / abs(mu_plus_signed)
```

The plus image is the primary only when the pair explicitly uses the
earliest-arriving SIS convention. In that case:

```text
abs(mu_plus)   = 2 / (1-r_flux)
mu_minus_signed = -2*r_flux / (1-r_flux)
y = (1-r_flux) / (1+r_flux)
```

Invalid domains raise errors; no clipping is permitted. Log absolute
magnification and logit relative flux are the stable transformed quantities.
These identities are an analytic control and do not generalize to SIE or EPL.
