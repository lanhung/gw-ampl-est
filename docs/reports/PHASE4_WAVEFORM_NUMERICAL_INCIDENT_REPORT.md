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

The correction will publish an immutable overlay excluding two Stage A and
three Stage B systems and adding exactly two plus three fresh, group-disjoint,
direct-target replacements. Validation remains unchanged. Corrected views must
contain exactly 32,768 Stage A train, 32,768 Stage B extension, 65,536 combined
train, and 6,144 validation systems before any training is reopened.

## Boundaries

The failed 65k output root is retained as immutable evidence. No checkpoint or
metric from it may be resumed. Architecture selection, calibration, SBC, final
evaluation, extension beyond 65k, real noise, and GWOSC/GWTC remain closed.
