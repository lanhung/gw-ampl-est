# Phase 2.4 waveform-window contract report

## Trigger

The frozen RC.4 8-second construction failed all four boundary fixtures against
independently generated 32-second references. Read-only diagnostics excluded a
crop-alignment error, measured slow finite-frequency-grid convergence and found
that production clean strain omitted Bilby's sampling-frequency normalization.
No microbenchmark pair or qualification shard was generated.

## RC.5 contract

RC.5 preserves the published 8-second, 2048 Hz, 16,384-sample product and all
RC.4 scientific distributions. It freezes a 64-second internal
IMRPhenomXPHM detector-response construction, merger at internal second 62,
Bilby `infft`, crop `[56 s, 64 s)`, 0.25-second zero guards and adjacent
0.25-second raised-cosine transitions. Selection SNR must be recomputed from the
conditioned published clean signal.

The numerical reference is a separately generated 128-second response. The
hard limits are 0.005 conditioned relative difference, 0.005 energy outside
the crop and 0.999 conditioned energy retention, together with exact zero
guards, finite products, detector-frame chirp-time reporting and normalized
transform/SNR consistency.

## Scope and authorization

The human project owner explicitly approved all ten items in the reviewed RC.5
proposal. This preregistration remains execution-disabled. Phase 3A requires a
new separate authorization commit referring to the RC.5 hash. Full production,
training, calibration, scientific testing, GWOSC/GWTC and Phase 3B remain
closed.

Canonical RC.5 configuration hash:

`4dde279cf1bea78d1ddbd4fab99d88e88e334c80c180dc7850679736c5e53edb`
