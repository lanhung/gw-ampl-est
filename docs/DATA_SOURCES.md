# Data sources

## Authoritative code repository

- Host role: Vultr control plane
- Repository: `/root/work/lensing-4`
- GitHub: `git@github.com:lanhung/gw-ampl-est.git`
- Rule: all new source edits and commits originate here.

## AutoDL compute roots

- SSH alias: `autodl-lensing`
- New writable root: `/root/autodl-tmp/lensing-4`
- Large v2 data, GWOSC cache and runs remain on AutoDL.

## Resolved legacy roots

### Original 0222/0228 catalogs

`/root/autodl-tmp/qkzhang`

This is the physical location of the audited SIS, point-mass and unlensed
0222/0228 arrays. It has no root Git repository and contains unrelated projects.
Treat the lensing assets as immutable.

### Downstream pair-verification project

`/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main`

This is a separate Git repository for pair verification/classification. Its
reproducibility manifests reference the qkzhang 0222/0228 catalogs. It also
contains later H1/L1 Gaussian-noise regenerations. It is not the source of the
magnification point-regression PDF.

### Magnification point-regression baseline

`/root/autodl-tmp/tmp`

The PDF-named generator, training entry, dataset, checkpoint and report sources
were found here. The exact baseline data root is:

`/root/autodl-tmp/tmp/数据生成/data_lens_randomu_realobs_quality_mu0`

The `(1)` sibling is a later observable-fix regeneration, not the dataset used
by the PDF checkpoint.

## Data handling

- All three legacy locations are read-only inputs.
- Do not rename, modify, delete, overwrite or normalize them in place.
- Do not recursively copy waveform catalogs to Vultr.
- The curated small-file snapshot under `vendor/legacy_snapshot/` is provenance
  evidence, not an implementation dependency.
- New outputs belong only under `/root/autodl-tmp/lensing-4`.
