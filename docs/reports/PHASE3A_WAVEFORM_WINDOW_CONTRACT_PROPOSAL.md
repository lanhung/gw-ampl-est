# Phase 3A waveform-window contract proposal

Status: **approved by the human project owner and frozen as RC.5**.

This document responds to the RC.4 waveform-boundary hard failure. The human
project owner subsequently approved all ten items; they are frozen in RC.5 at
canonical hash
`4dde279cf1bea78d1ddbd4fab99d88e88e334c80c180dc7850679736c5e53edb`.
No microbenchmark pair, shard or dataset was generated while preparing it.

## Diagnosed causes

The frozen generator commit
`a2b8a02b4631e86c39e1b682e4424ecc2f2c5ca9` failed all four boundary fixtures.
Additional read-only AutoDL diagnostics show:

- the 8-second and 32-second crops are aligned correctly: the best circular lag
  is zero samples in all four cases;
- a fitted scalar is only `1.0008`--`1.0026`, so amplitude rescaling does not
  explain the `0.0338`--`0.0609` mismatch;
- after excluding both 0.25-second edge guards, the mismatch remains
  `0.0321`--`0.0576`;
- 32-to-64-second frequency-grid differences are `0.00545`--`0.00840`, and
  64-to-128-second differences are `0.00231`--`0.00417`;
- the independently generated 32-second reference crop itself violates the old
  `1e-6` integrated edge-energy threshold.

The old gate therefore compares distinct finite-frequency-grid approximations
at a tolerance they cannot satisfy. It is not merely a crop-index error.

The audit also found a separate implementation defect: production clean strain
uses `numpy.fft.irfft`, whereas Bilby's normalized inverse transform is
`bilby.core.utils.infft`, equal to `irfft * sampling_frequency`. At 2048 Hz the
stored clean strain would be too small by a factor of 2048 relative to the
noise, even though the pre-transform optimal-SNR selection value is correctly
scaled. Official generation must not proceed with that inconsistency.

## Recommended RC.5 contract

Human review should accept, reject or amend every item below as one versioned
contract before implementation.

1. Keep the published product at exactly 8 seconds, 2048 Hz and 16,384 samples.
2. Construct IMRPhenomXPHM detector responses internally on a fixed 64-second
   frequency grid at 2048 Hz, with the geocentric merger at internal time 62
   seconds.
3. Convert frequency-domain detector responses with Bilby's normalized
   `infft`; add an explicit forward/inverse normalization and Parseval test.
4. Crop internal samples `[56 s, 64 s)` so the published merger remains at
   output time 6 seconds.
5. Set the first and last 0.25 seconds of the crop to zero. Apply a deterministic
   raised-cosine transition over the adjacent 0.25 seconds on each side. Record
   the window name and exact sample indices in configuration and provenance.
6. Recompute detector-specific selection SNR from the conditioned 8-second clean
   product and the declared PSD. Do not select using the unconditioned
   64-second response while storing a different signal.
7. Use a separately generated 128-second response, with merger at 126 seconds,
   as the numerical reference. Crop and condition it by the identical rule.
8. Require, for every frozen boundary fixture:
   - finite frequency- and time-domain products;
   - exactly zero energy in each 0.25-second guard;
   - 64-to-128-second conditioned relative difference no greater than `0.005`;
   - energy outside the 8-second crop in the 64-second construction no greater
     than `0.005` of total construction energy;
   - conditioned crop energy at least `0.999` of the unconditioned crop energy;
   - normalized inverse-transform and stored-clean SNR agreement within a
     separately unit-tested floating-point tolerance.
9. Keep the existing four boundary fixtures and add detector-frame chirp-time
   reporting so containment is explained rather than inferred only from edge
   zeros.
10. Treat any failure as a hard stop. Do not alter thresholds after an RC.5
    result.

The proposed numerical limits are deliberately disclosed as post-RC.4 choices.
They become legitimate only through explicit human review and a new
preregistration version/hash. The diagnostic maxima motivating them are
`0.0041659` for the raw 64-to-128 comparison, `0.0039481` after conditioning,
`0.0023538` for outside-crop energy and a minimum conditioned energy retention
of `0.9995028`.

## Required restart sequence after approval

After human approval, and not before:

1. create a new preregistration release candidate and canonical hash;
2. update the separate Phase 3A authorization and configuration to that exact
   version, hash and authorizing commit;
3. fix the transform normalization and implement the reviewed construction;
4. add unit and AutoDL boundary tests, then freeze a new clean generator commit;
5. derive new run and dataset IDs and rerun every source-plane, mass-sheet,
   Galkin, waveform, whitening, input-policy and resource gate;
6. rerun the 32-accepted-pair microbenchmark from the new frozen commit;
7. proceed to the three-shard interruption/resume test and 4,096 accepted pairs
   only if every gate, including the 24-hour projection, passes.

Full production, training, calibration, scientific testing, GWOSC/GWTC access
and Phase 3B remain unauthorized.
