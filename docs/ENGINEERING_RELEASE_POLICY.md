# Engineering release policy

## Two-layer governance

Scientific contracts and software implementation are reviewed at different
levels.

A preregistration version and scientific review are required when changing an
estimand, posterior target, generative distribution, selection or split,
importance-weighting objective, model-selection rule, calibration/evaluation
criterion, or scientific claim.

Schema accessors, logging, telemetry, checkpoint/resume behavior, environment
repair, output plumbing, manifests, test coverage and distribution-preserving
performance work are patch releases inside the current major phase. They use
one branch and one final PR rather than additional scientific phase numbers.

## Release sequence

Every future official data execution must follow one release path:

1. patch implementation and typed integration tests;
2. CI and authoritative remote tests;
3. a disposable execution canary using the official code path but separate
   seeds, identities and `dataset_purpose: execution_canary`;
4. freeze a clean commit and immutable environment identity;
5. run a single release-gate command;
6. create official identities only after the gate reports
   `ready_for_official_execution`;
7. execute, validate and publish without changing code.

Canaries are permanently non-scientific and cannot be counted toward an
official run. Their purpose is schema, storage, resume, telemetry, manifest and
checksum validation; they cannot expose or decide a preregistered scientific or
A/B endpoint.

## Typed schema boundary

Scripts may not access schema dataclass fields directly. Typed helpers in
`src/gwlens_mm/` own schema validation and serialization. The same helper must
be used by canary, first-block and final validators. Integration coverage must
exercise `V2Record -> JSON -> Parquet/Zarr -> validator`.

## Environment identity

An official run records a container digest when available. Otherwise it
records the wheel hash, dependency lock or environment freeze hash, Python and
scientific-package versions, PSD hashes and synchronized Git commit. A mutable
remote environment description is not sufficient identity.

## Retry and fallback

Each major phase permits at most one full official engineering retry. A second
engineering execution failure closes that optimization path and activates its
preregistered fallback. This rule prevents ordinary software defects from
creating an open-ended sequence of scientific subphases.

The Phase 3C proposal path has now used that retry. No proposal-v4 or further
proposal A/B is planned. Future scientific production must use a separately
reviewed direct-evaluation-target contract, with unit importance weights, so
proposal optimization cannot block the publication path.

Future engineering comparisons should measure target-effective systems per
active hour when statistical efficiency differs. Such an endpoint must be
frozen before the compared run begins; it may not be applied retroactively to
partial Phase 3C-A evidence.
