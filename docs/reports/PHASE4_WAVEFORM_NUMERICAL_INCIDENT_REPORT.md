# Phase 4 waveform numerical incident

## Outcome

The first authorized 65k probe launch stopped before its first optimizer step.
The stop was correct: whitening encountered a finite but catastrophically large
IMRPhenomXPHM source-polarization bin in a published training record. No valid
65k checkpoint, development metric, or terminal learning-curve decision exists.

## Exhaustive audit

Every 71,680 published Stage A/Stage B record was regenerated through the
unlensed source-polarization call without producing a new image pair. Five
training records failed an isolated-bin diagnostic; validation contained none.
Normal records had peak-to-positive-in-band-99.9%-quantile ratios at or below
1.704718. The five failures ranged from 12,983 to 2.562e24. Four carried grossly
unphysical recorded network SNR values; the fifth showed why an SNR-only audit
would be insufficient.

The machine-readable evidence is
`results/phase4/waveform_spectral_audit.json`.

## Scientific impact

Two failures are in Stage A train and three in the Stage B extension. One Stage
A failure entered the deterministic 16k subset and both entered the 32k rung.
The existing 16k/32k learning-curve decision is therefore superseded. The
published Stage A and Stage B roots remain immutable and retain their original
hashes; they must not be silently edited, clipped, or partially dropped.

## Frozen correction

Preregistration `1.1.1-rc.1` adds only a numerical-validity rule before lensing,
detector projection, and selection. For plus and cross separately, it computes
the maximum strictly positive in-band amplitude divided by the linear 99.9th
percentile of strictly positive in-band amplitudes. A maximum ratio above 10 is
a deterministic rejected attempt. Clipping, waveform repair, and parameter
substitution are forbidden.

The threshold is a numerical-separation criterion, not a performance-tuned
selection: the exhaustive audit was completed before any 65k optimizer step,
the largest retained ratio is 1.705, and the smallest rejected ratio is 12,983.
The estimand, direct evaluation target, q=p unit weights, physical selection,
waveform approximant, PSD, counts, model, and stopping rule are unchanged.

The correction published an immutable overlay excluding two Stage A and three
Stage B systems and adding exactly two plus three fresh, group-disjoint,
direct-target replacements. Validation remains unchanged. The corrected views
contain exactly 32,768 Stage A train, 32,768 Stage B extension, 65,536 combined
train, and 6,144 validation systems.

## Correction publication and independent closeout

The authorized run completed under parent
`phase4-waveform-correction-499f86b3159a-1db109b08189`. Its parent manifest
SHA-256 is
`0fcfb117c620d58a2e0ccd8b19c0d3f3a371dd844fb637b50c8b565eee6864f2`,
and the publication-tree SHA-256 is
`a57aa2691e256b34403392f595e964dceec1325cfc54a38ed4d2a0b714d38c12`.
The two replacement shards required 338 and 890 attempted proposals and occupy
5,667,197 bytes together.

An independent read-only closeout recomputed the complete publication tree,
every shard artifact and all three immutable base-manifest hashes. It reopened
the five Parquet records and all three Zarr products, verified finite float32
arrays and exact `noisy = clean + noise`, regenerated all five source
polarizations, and rescanned grouped identities across the 71,680 base records.
Replacement peak-to-99.9%-quantile ratios ranged from 1.0008 to 1.0957, below
the frozen threshold of 10. Proposal and evaluation log probabilities are
equal, every importance weight is exactly one, and no replacement group
overlaps Stage A train, validation or Stage B train. The original publications
were not modified and the staging parent is absent after atomic publication.

This closeout resolves the data correction only. It does not revive the
superseded learning-curve result or authorize an optimizer. The corrected 16k
membership must be recomputed and all 16k/32k fits rerun under a separately
bound training release.

## Boundaries

The failed 65k output root is retained as immutable evidence. No checkpoint or
metric from it may be resumed. Architecture selection, calibration, SBC, final
evaluation, extension beyond 65k, real noise, and GWOSC/GWTC remain closed.

## Implementation preflight

The correction implementation is frozen at
`499f86b3159af82612e38c134cd81003eedcc4e4`; wheel SHA-256 is
`1088b2be49e879cbc44fc834b09c67947b45f2da444e15a3f41856abf60729f2`.
Local tests passed (321 with seven optional skips), as did maintained-scope
Ruff, mypy and package build. AutoDL passed 331 tests with one optional PyTorch
skip. The exact real-record regression rejected all five known pathologies and
accepted the largest known valid boundary record. All three immutable base
manifest hashes reproduced. Delegated review therefore authorizes only the
five-system replacement and atomic corrected-view publication; training stays
closed.
