# Provenance, seeds, and manifests

## Canonical configuration identity

Mappings are serialized as sorted, compact UTF-8 JSON and hashed with SHA-256.
The dataset ID is derived from schema version, full generator Git commit,
configuration hash, and root seed:

```text
gwlens-v2-<schema-version>-<16-hex digest>
```

Changing any identity component creates a different dataset ID.

## Seed hierarchy

Child seeds use SHA-256 over the root seed, a registered domain, and stable
identifiers. The first unsigned 64 bits form the child seed. Python's
process-randomized `hash()` is never used.

Registered domains are:

- source;
- lens;
- pair selection;
- detector noise;
- EM measurement noise;
- missing modalities;
- augmentation.

For example, detector noise includes pair/image/detector IDs, ensuring stable
but distinct streams. Adding workers or changing iteration order must not
change existing child seeds.

## Manifest contract

`DatasetManifest` records dataset identity, schema and code/config versions,
root seed, accepted and attempted counts, all grouping IDs, artifacts, and
generation status. It rejects:

- accepted counts not accounted for by pair IDs;
- attempted count below accepted count;
- duplicate pair/source/lens/noise IDs;
- unsafe absolute or parent-traversal artifact paths;
- complete artifacts without byte size and SHA-256;
- duplicate dataset IDs across a registry.

Artifacts may be `pending` in a Phase 1A plan. A completed generation must
replace placeholders with hashes and byte counts. Rejected attempts and their
reasons are separate append-only provenance records.

## External versions

Future manifests must include solver, waveform, preprocessing, detector, PSD,
calibration/DQ, data-release, and package versions. The Phase 1A metadata
example is explicitly marked as planned and is not evidence of generation.
