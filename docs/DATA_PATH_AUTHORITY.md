# Data path authority

## Binding decisions

| Purpose | Authoritative location | Status |
|---|---|---|
| New source code | `/root/work/lensing-4` | sole writable code authority |
| New AutoDL compute copy | `/root/autodl-tmp/lensing-4/repo` | disposable |
| New v2 datasets | `/root/autodl-tmp/lensing-4/data_v2` | future writable authority |
| Original 0222/0228 | `/root/autodl-tmp/qkzhang/{SIS,PM,Unlensed}_data_{0222,0228}` | immutable legacy |
| Pair-verification research artifact | `/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main` | immutable legacy Git repo |
| Exact PDF baseline data | `/root/autodl-tmp/tmp/数据生成/data_lens_randomu_realobs_quality_mu0` | immutable legacy |
| Later PDF-baseline obsfix data | `/root/autodl-tmp/tmp/数据生成/data_lens_randomu_realobs_quality_mu0(1)` | immutable diagnostic legacy |

## Reconciliation result

The former two-path conflict is resolved: qkzhang is the data origin for the
0222/0228 catalogs, while the wjx path is a later downstream pair-verification
project that consumes those catalogs and contains additional regenerated data.
Neither path is the PDF magnification baseline.

The PDF itself supplied a third exact path under `/root/autodl-tmp/tmp`; that
path was found and verified against the checkpoint. This expansion was read-only.

## Prohibited assumptions

- A matching SIS filename does not establish shared provenance.
- wjx H1/L1 arrays must not be relabeled as corrected qkzhang 0222/0228 without
  a versioned data manifest.
- `realobs` means simulated observation proxies in Gaussian design noise; it
  does not mean real GWOSC detector noise.
- The `(1)` data directory is not interchangeable with the exact PDF dataset.
