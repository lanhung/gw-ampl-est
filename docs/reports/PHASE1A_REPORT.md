# Phase 1A report — physics API, v2 schema, and leakage safeguards

> Historical gate report: schema alpha.1 was metadata-only and was superseded
> by the Phase 1A.1 alpha.2 hardening before any v2 data were generated.

Date: 2026-07-12
Branch: `phase1/physics-schema`
Gate checkpoint: `166f781`

## Outcome

Phase 1A acceptance criteria are satisfied for the lightweight foundation.
No waveform data were generated, no neural model was trained, no GWOSC/GWTC
product was downloaded, and no legacy file was modified.

The implementation establishes:

- explicit magnification, amplitude, parity, Morse, timing, distance, and
  image-primary conventions;
- a tested positive/negative-frequency Morse and time-delay implementation;
- stable SIS analytic forward/inverse controls with no clipping;
- a solver protocol that returns all physical images separately from a
  selected observed pair;
- a fail-closed deployable-input allowlist and privileged-variable denylist;
- a versioned logical metadata schema with exact round trip;
- covariance, missing-modality, detector-axis, and array-reference contracts;
- grouped split leakage detection including a dedicated calibration split;
- deterministic config hashing, child seeds, dataset IDs, and manifests;
- a storage ADR and execution-disabled 48-pair smoke specification.

## Scientific decisions

### Relative quantities

The canonical ratio is always
`mu_abs_secondary / mu_abs_primary`, never an implicit faint/bright ratio.
General selected pairs may have a value above one because the catalog anchor or
earliest image need not be brightest. The bounded `(0,1)` requirement and logit
transform apply only where explicitly justified, including the SIS plus-primary
analytic inversion.

### Multiple images

`LensSystemSolution`/`LensTruth` stores every physical image. `SelectedPair` and
`PairIndex` store two selected IDs plus a primary definition. The metadata
example contains three physical images, selects image 0 and image 2, and marks
image 3 censored.

### Truth versus observations

Truth records remain available for targets and diagnostics, but exact source
position, lens truth, magnification, clean signal, isolated noise, optimal SNR,
IDs, and proposal/prior weights are rejected as model inputs. EM input fields
are explicitly noisy observations with covariance and availability masks.

## Verification results

The completed local check produced:

```text
65 passed in 0.41s
Ruff: all checks passed
mypy: success, no issues in 12 source files
```

Coverage includes SIS limiting/round-trip cases, domain rejection, legacy
fixture agreement, all Morse classes, conjugate symmetry, time-shift direction,
multi-image pair selection, a lightweight non-SIS adapter contract, exact and
alias leakage rejection, schema round
trip, covariance, non-finite metadata and missing-mask failures, detector-mask
failures, deliberately corrupted grouped
splits, deterministic seeds/config IDs/manifests, storage arithmetic, and smoke
configuration validation.

The GitHub Actions workflow runs Python 3.9 pytest, Ruff, and mypy.
The package was also built into a wheel and imported from an isolated `/tmp`
target using the host's installed build backend. Its reported schema version
was `2.0.0-alpha.1`.

## AutoDL contract check

The lightweight repository was synchronized only into
`/root/autodl-tmp/lensing-4/repo`. A remote import plus SIS solver contract
passed. `lenstronomy` was not installed on the checked AutoDL Python path, so no
non-SIS numerical reference configuration was run and no package was installed
remotely. SIE/EPL adapters remain Phase 1B implementation work after selection
of a pinned solver environment.

The approved smoke output path did not exist at validation time. Therefore
`configs/data/v2_smoke.yaml` remains a specification, not a generated dataset.

## Storage decision

ADR-001 selects Zarr v2 plus Parquet with unique staged shards and single-writer
manifest publication. Required raw strain storage for the three float32
products is approximately:

- 48 pairs: 13.50 MiB;
- 10,000 pairs: 2.7466 GiB;
- 100,000 pairs: 27.4658 GiB.

Zarr/Parquet dependencies were not installed during Phase 1A. Phase 1B must pin
and record versions on AutoDL; sharded HDF5 plus Parquet is the documented
fallback requiring an ADR amendment.

## Failures and limitations

- No external non-SIS solver was available on AutoDL; only the generic adapter
  contract and SIS implementation were executed.
- The alpha JSON Schema provides top-level boundary shape while typed Python
  validators enforce nested scientific constraints. It must be expanded before
  accepting records from an independent producer.
- Phase 1A does not validate waveform projection, amplitude preservation,
  whitening, noisy-plus-clean decomposition, or generator resumability because
  those require Phase 1B artifacts.
- The first sync exposed over-broad `data/` exclusion and missing type/lint
  cache exclusions. The script was corrected to root-anchor large-data paths
  and exclude caches; no legacy path was touched.
- A default isolated editable install attempted to download build requirements
  through unavailable sandbox DNS, and the managed user site was read-only.
  Re-running a no-network build with `PIP_CACHE_DIR` and install target under
  `/tmp` successfully built and imported the wheel.

## Gate recommendation

Phase 1A is ready for human review. Do not execute Phase 1B until the reviewer
accepts the conventions, policy files, schema, ADR, tests, and execution-disabled
smoke configuration. After acceptance, Phase 1B must generate exactly 48
accepted pairs and stop without training.
