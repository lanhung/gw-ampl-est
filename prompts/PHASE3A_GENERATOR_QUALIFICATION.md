# Phase 3A — Production-generator qualification on 4,096 engineering pairs

Execute Phase 3A only.

Read before doing any work:

- `AGENTS.md`
- `configs/execution/phase3a_qualification_authorization.yaml`
- `configs/statistics/phase2_preregistration.yaml`
- `docs/PHASE2_PREREGISTRATION.md`
- `docs/reports/PHASE2A_PREREGISTRATION_HARDENING_REPORT.md`
- `docs/PHYSICS_CONVENTIONS.md`
- `docs/LENS_SOLVER_INTERFACE.md`
- `docs/V2_SCHEMA_SPEC.md`
- `docs/PRIVILEGED_INPUT_POLICY.md`
- `docs/SPLIT_POLICY.md`
- `docs/PROVENANCE_AND_SEEDS.md`
- `docs/DECISIONS.md`
- `docs/FAILURES.md`

Work only on:

`phase3a/generator-qualification`

The frozen authoritative preregistration is:

- version: `1.0.0-rc.5`;
- configuration hash:
  `4dde279cf1bea78d1ddbd4fab99d88e88e334c80c180dc7850679736c5e53edb`;
- scientific schema: `2.0.0-alpha.3`;
- base main commit:
  `80167ea690914bb18be1fd1994b4dc626490e146`.

Do not change the frozen RC.5 distributions or waveform-window contract merely to make the generator
pass. A contradiction or unimplementable requirement is a hard failure and
must stop execution for human review.

Do not train a model.
Do not generate more than 4,096 accepted pairs.
Do not start full production.
Do not download GWOSC or GWTC data.
Do not modify the frozen engineering smoke artifact.
Do not use qualification data scientifically.
Do not proceed beyond Phase 3A.

======================================================================
1. VERIFY HUMAN AUTHORIZATION AND FROZEN CONFIGURATION
======================================================================

Before modifying production code:

1. Confirm the current branch is `phase3a/generator-qualification`.
2. Confirm the branch descends from the stated main merge commit.
3. Confirm the working tree contains only intentional gate changes.
4. Load the authorization YAML.
5. Recompute the canonical preregistration hash.
6. Require an exact match to:

   `4dde279cf1bea78d1ddbd4fab99d88e88e334c80c180dc7850679736c5e53edb`

7. Confirm:
   - exact accepted count is 4096;
   - shard size is 128;
   - full production is false;
   - training is false;
   - scientific use is false;
   - GWOSC/GWTC access is false.
8. Fail closed if any authorization value is absent or inconsistent.

Do not flip execution flags inside the frozen Phase 2 preregistration file.
The separate Phase 3A authorization file is the sole execution gate.

======================================================================
2. REMOTE PREFLIGHT AND RESOURCE GATES
======================================================================

Use:

`ssh autodl-lensing`

Use only the new project root.

Before creating large files, record:

- hostname;
- Python version;
- package versions;
- CPU count;
- available memory;
- GPU information for provenance only;
- filesystem and free bytes;
- exact Git checkout;
- environment freeze;
- PSD file locations and SHA-256 values.

Require at least:

`105807608883`

free bytes before launch.

If free space is below the gate, stop without generating data.

Create before generation:

- run ID;
- pre-run manifest;
- log path;
- staging path;
- resume plan;
- resource estimate;
- expected artifact list.

Do not use `rsync --delete`.
Do not delete failed staging evidence.
Do not write under any legacy directory.

======================================================================
3. CREATE A PRODUCTION PIPELINE, NOT A SCALED SMOKE PIPELINE
======================================================================

Do not increase the Phase 1B smoke pair count and reuse its in-memory
`np.stack` publication path.

Implement a separate production package, preferably:

`src/gwlens_mm/production/`

The production generator must:

- remain bounded-memory as total pair count grows;
- never load all 4,096 pairs into memory;
- write sharded products;
- use process-local deterministic RNG;
- never share Bilby's global RNG between threads;
- support deterministic interruption and resume;
- use atomic same-filesystem renames;
- record every attempted and rejected proposal;
- prevent duplicate pair, source, lens, noise and physical-system IDs;
- publish the dataset manifest only after every shard passes validation.

Create or complete:

`docs/adr/ADR-002-production-generation-and-storage.md`

The ADR must define:

- shard directory layout;
- Zarr v2 layout;
- Parquet partitioning;
- manifest hierarchy;
- attempt journal;
- checksums;
- staging and publication states;
- interrupted shard treatment;
- final atomic publication;
- corruption recovery;
- maximum memory behavior.

======================================================================
4. PHASE 3A CONFIGURATION
======================================================================

Create:

`configs/data/phase3a_qualification.yaml`

It must reference rather than duplicate the frozen preregistration wherever
possible.

It must include:

- exact preregistration version and hash;
- exact authorization file and commit;
- root seed;
- dataset purpose `generator_qualification`;
- scientific use false;
- accepted pair count 4096;
- shard count 32;
- pairs per shard 128;
- output paths;
- environment and package pins;
- qualification-only GPS schedule;
- interruption-test settings;
- microbenchmark settings;
- runtime stop gate;
- disk stop gate;
- checksum policy.

The qualification GPS schedule is an engineering schedule used to exercise
different detector antenna responses. It is not an astrophysical observing
time or detector-duty-cycle population and must be labeled accordingly.

Do not invent a future scientific duty-cycle distribution in Phase 3A.

======================================================================
5. SCIENTIFIC SCHEMA ALPHA.3
======================================================================

Use schema:

`2.0.0-alpha.3`

Maintain backward reading of the frozen alpha.2 smoke records, but do not
migrate or regenerate them.

Alpha.3 qualification records must include:

- environment/external-convergence observation;
- environment availability mask;
- kinematics observation;
- aperture and seeing metadata;
- tracer effective radius;
- dynamics model reference;
- exact detector-specific PSD references;
- proposal and evaluation log probabilities;
- selection metadata as privileged provenance;
- all physical images;
- selected observed image pair;
- complete extra-image status;
- source/lens/noise/physical-system group IDs.

Truth quantities, proposal weights, optimal SNR and selection statistics
must remain forbidden as deployable model inputs.

======================================================================
6. LENS AND EXTERNAL-CONVERGENCE GENERATION
======================================================================

Generate only the two declared scientific families:

- SIE plus external shear;
- EPL plus external shear.

Draw from the frozen broad proposal distribution.

Retain every physical image returned by the solver.

Apply the frozen mass-sheet external-convergence model with:

    lambda = 1 - kappa_ext

Validate before any large generation:

- source-position scaling;
- image-position invariance;
- signed-magnification scaling by `1 / lambda^2`;
- absolute-magnification scaling;
- Fermat-difference scaling by `lambda`;
- physical time-delay scaling by `lambda`.

External convergence must affect the generated magnifications and delays.
It must not exist only as disconnected metadata.

Stop on degenerate, nonfinite or contract-violating lens solutions.

======================================================================
7. SOURCE POPULATION AND WAVEFORMS
======================================================================

Sample the joint source distribution exactly as frozen.

In particular:

- primary source-frame mass follows the frozen power law;
- mass ratio uses the conditional lower bound
  `max(0.25, 10 / m1)`;
- secondary source-frame mass must therefore be at least 10 solar masses;
- convert source-frame masses to detector-frame masses using source redshift;
- derive luminosity distance from the declared cosmology;
- record normalized proposal and evaluation log densities.

Generate baseline waveforms with:

- IMRPhenomXPHM;
- 8 seconds;
- 2048 Hz;
- 20 Hz minimum frequency;
- 50 Hz reference frequency;
- H1/L1/V1 detector slots.

Each physical image has its own geocentric arrival time and detector response.

Galaxy-scale image delays belong in metadata and separate image windows.
Never shift multi-day delays inside one 8-second array.

Apply:

- `sqrt(abs(mu_signed))`;
- the correct Morse phase;
- detector projection at each image arrival time.

Generate noisy, clean and noise products separately.

Store only the selected two images, while retaining truth metadata for every
physical image.

======================================================================
8. WAVEFORM-BOUNDARY QUALIFICATION
======================================================================

Before the 4,096-pair run, create deterministic boundary fixtures covering:

- minimum and maximum source masses;
- extreme mass ratios;
- high source redshift;
- high aligned and precessing spin examples;
- both lens families;
- high magnification;
- long delay.

Construct the published 8-second product from the frozen 64-second internal
grid and compare it against the separately generated 128-second reference.
Apply the exact RC.5 crop, zero guards and raised-cosine transitions. Verify:

- no cyclic wraparound occurs;
- merger and relevant waveform support are contained;
- no edge-truncated waveform is accepted;
- time-domain and frequency-domain products are finite;
- detector projections remain finite;
- Bilby `infft` sampling-frequency normalization is preserved;
- both 0.25-second guard regions have exactly zero energy;
- conditioned 64-to-128-second relative difference is no greater than 0.005;
- construction energy outside the 8-second crop is no greater than 0.005;
- conditioned crop energy retention is at least 0.999;
- detector-frame chirp time is reported;
- selection SNR is recomputed from the conditioned published clean signal.

Define the numerical boundary criteria in the Phase 3A configuration before
running the 4,096 pairs.

Do not relax the criteria after observing failures. A failure requires
stopping or a new reviewed configuration.

======================================================================
9. PSD, NOISE AND WHITENING
======================================================================

Verify exact baseline curve files and hashes:

- H1/L1: `aLIGO_O4_high_asd.txt`;
- V1: `AdV_psd.txt`.

Use synthetic Gaussian curve-conditioned noise only.

Do not label it design sensitivity without the exact curve name.
Do not label it real detector noise.

Preserve unwhitened float32 products:

- noisy strain;
- clean injected signal;
- noise realization.

Derive whitening from the declared detector PSD, sample rate and duration.

Do not normalize each event using its own observed standard deviation.

Before the full qualification run, document the exact whitening
normalization and freeze aggregate acceptance criteria.

For qualification data report by detector:

- finite fraction;
- mean;
- standard deviation;
- selected quantiles;
- frequency-band variance behavior;
- outlier count;
- dependence on detector and PSD curve.

Hard fail on NaN, Inf, undeclared PSD, PSD hash mismatch, or systematic
failure of the frozen unit-scale whitening contract.

Do not store an additional full whitened product unless an explicit reviewed
storage change is made. Small summaries and fixtures may be committed.

======================================================================
10. SYNTHETIC SELECTION MODEL
======================================================================

Compute selection statistics from clean signals and declared PSDs.

For an image to pass:

- network optimal SNR >= 10;
- at least two detectors contribute;
- each contributing detector has SNR >= 4.

If more than two physical images pass, select the earliest two passing
images.

If fewer than two pass, reject the attempted physical system.

Record privileged selection information in provenance only:

- per-detector optimal SNR;
- per-image network SNR;
- passing image IDs;
- rejection reason;
- selected pair rule.

Never expose these values as deployable inputs.

Append every attempted proposal to a deterministic append-only attempt
journal.

Report acceptance rates by:

- lens family;
- image multiplicity;
- EM availability cell;
- source and lens parameter region;
- rejection reason.

======================================================================
11. EM AND ENVIRONMENT OBSERVATION GENERATION
======================================================================

Implement all eight frozen EM availability cells exactly.

The assignment must be deterministic and balanced by physical-system ID.

Exercise all cells in the 4,096 accepted qualification set.

Generate noisy observations, never truth substitution, for available
modalities:

- image-ID-keyed astrometry;
- lens center;
- Einstein scale;
- lens redshift;
- source redshift;
- environment/external convergence;
- velocity dispersion;
- timing.

For unavailable modalities:

- store null observation values;
- store false availability masks;
- never insert latent truth.

Environment observation states must support:

- informative;
- weak;
- unavailable.

The observed external-convergence mean must be drawn from the frozen
observation likelihood around latent kappa_ext with the declared standard
deviation.

======================================================================
12. STELLAR-KINEMATICS FORWARD MODEL
======================================================================

Use the frozen Lenstronomy Galkin model:

- circularized spherical power-law dynamics mass model;
- Hernquist light model;
- Osipkov-Merritt anisotropy;
- declared shell aperture;
- declared Gaussian PSF;
- luminosity-weighted aperture averaging;
- 4,000 Monte Carlo samples;
- process-local deterministic seed.

Do not infer velocity dispersion directly from Einstein radius.
Do not use a deterministic analytic shortcut.

Before full qualification, run a deterministic convergence subset balanced
across:

- both lens families;
- all eight EM cells;
- low/high slope;
- low/high effective radius;
- low/high anisotropy radius.

Compare 4,000 samples to the frozen 16,000-sample reference setting.

Require maximum relative difference <= 0.02.

Record per-case values, runtime and relative difference.

If this contract fails, stop before generating 4,096 accepted pairs.

======================================================================
13. MICROBENCHMARK GATE
======================================================================

Before the full Phase 3A run:

1. Generate a deterministic 32-accepted-pair microbenchmark.
2. Exercise:
   - both lens families;
   - all eight EM cells;
   - doubles and quads where available;
   - external convergence;
   - kinematics;
   - waveform generation;
   - PSD noise;
   - whitening;
   - selection and rejection logging.
3. Measure:
   - attempts per accepted pair;
   - lens-solver time;
   - dynamics time;
   - waveform time;
   - noise time;
   - storage time;
   - checksum time;
   - peak RSS;
   - disk amplification.
4. Project the 4,096-pair wall time and peak disk use.

If projected Phase 3A wall time exceeds 24 hours, stop and create a blocked
qualification report. Do not silently reduce physics fidelity.

If projected output exceeds 10 GB, stop.

If projected post-run free space is below 100 GB, stop.

Do not count the microbenchmark as part of the final published 4,096 pairs
unless it uses the exact final generator commit, configuration and dataset
ID and passes deterministic resume requirements.

======================================================================
14. SHARDED GENERATION AND ATOMIC PUBLICATION
======================================================================

Generate exactly 4,096 accepted pairs in 32 shards.

Each shard contains exactly 128 accepted pairs.

Use a layout equivalent to:

    staging/<dataset_id>/
        run_manifest.json
        attempts/
        shards/
            shard-00000.partial/
            shard-00000/
            ...
        validation/
        environment/

Each complete shard must contain:

- three Zarr v2 strain products;
- Parquet metadata partitions;
- shard manifest;
- accepted and rejected attempt range;
- artifact SHA-256 values;
- byte counts;
- complete marker.

A partial shard must never be treated as complete.

Use same-filesystem atomic rename from partial to complete shard.

Do not combine all shards in memory.

After all 32 shards pass:

- run cross-shard duplicate and grouped-leakage validation;
- create the final dataset manifest;
- recompute artifact checksums;
- atomically rename the complete dataset root into the publication path.

The final dataset must state:

- `dataset_purpose: generator_qualification`;
- `scientific_use_authorized: false`;
- `training_use_authorized: false`;
- `calibration_use_authorized: false`;
- `test_use_authorized: false`;
- accepted count 4096;
- shard count 32;
- exact preregistration version/hash;
- exact generator commit;
- exact authorizing commit;
- exact environment and PSD identities.

======================================================================
15. INTERRUPTION AND RESUME TEST
======================================================================

Perform a real deterministic interruption test.

Required sequence:

1. Complete exactly three shards.
2. Record all artifact and shard hashes.
3. Stop cleanly.
4. Resume using the same run ID, dataset ID, configuration and code.
5. Finish all 32 shards.
6. Verify the first three shard hashes remain byte-identical.
7. Verify no attempt, pair or seed is duplicated.
8. Verify no completed shard was rewritten.
9. Verify a partial shard is safely resumed or regenerated according to
   the documented ADR.

A resume mismatch is a hard failure and must prevent publication.

======================================================================
16. QUALIFICATION VALIDATION
======================================================================

Validate:

- exactly 4096 accepted pairs;
- exactly 32 complete shards;
- exactly 128 accepted pairs per shard;
- schema alpha.3 round trip;
- backward alpha.2 reading;
- all IDs unique;
- all groups valid;
- no qualification group enters any future scientific split;
- all physical images retained;
- selected-pair identity valid;
- all extra images explicitly classified;
- primary ordering valid;
- magnification, parity and Morse consistency;
- external-convergence transformation contracts;
- environment observations and masks;
- kinematics forward-model metadata;
- EM cell consistency;
- noise provenance;
- exact PSD identity and hashes;
- noisy = clean + noise;
- finite float32 arrays;
- unavailable detector slots obey mask semantics;
- waveform boundary checks;
- whitening checks;
- proposal/evaluation log-density support;
- finite importance weights;
- deployable-input policy;
- append-only attempts;
- checksums;
- interruption/resume behavior;
- bounded-memory behavior.

Measure and report:

- total attempts;
- accepted count;
- overall acceptance rate;
- acceptance rate by family and multiplicity;
- rejection-reason frequencies;
- wall time;
- accepted pairs per hour;
- attempts per hour;
- CPU utilization;
- peak RSS;
- actual staging bytes;
- published bytes;
- peak disk bytes;
- checksum time;
- projected full-production wall time;
- projected full-production peak disk;
- remaining free space.

Do not authorize full production from inside the report.

======================================================================
17. REQUIRED SMALL OUTPUTS
======================================================================

Keep large arrays and full Parquet/Zarr data only on AutoDL.

Commit only small artifacts, including:

- production generator code;
- tests;
- Phase 3A configuration;
- ADR-002;
- environment lock/freeze;
- copied final dataset manifest;
- validation summary JSON;
- acceptance summary CSV;
- rejection summary CSV;
- throughput summary JSON;
- whitening summary CSV/JSON;
- waveform-boundary summary;
- kinematics-convergence summary;
- interruption/resume hashes;
- small diagnostic figures;
- updated experiment registry;
- updated decisions, failures and project state.

Create:

- `docs/reports/PHASE3A_GENERATOR_QUALIFICATION_REPORT.md`
- `results/phase3a/qualification_manifest.json`
- `results/phase3a/qualification_validation.json`
- `results/phase3a/throughput.json`
- `results/phase3a/acceptance_summary.csv`
- `results/phase3a/rejection_summary.csv`
- `results/phase3a/whitening_summary.json`
- `results/phase3a/kinematics_convergence.csv`
- `results/phase3a/waveform_boundary_validation.json`
- `results/phase3a/pre_resume_hashes.sha256`
- `results/phase3a/post_resume_hashes.sha256`

Do not commit:

- Zarr arrays;
- full Parquet records;
- waveform caches;
- complete attempt journals if large;
- checkpoints;
- credentials;
- temporary shards.

======================================================================
18. TESTS AND COMPLETION GATE
======================================================================

Run locally where lightweight:

    python -m pytest -q
    ruff check src tests scripts
    python -m mypy src
    python -m build

Run the authoritative scientific and optional tests on AutoDL.

Phase 3A passes only when:

1. all tests pass;
2. the preregistration hash matches exactly;
3. all external-convergence contracts pass;
4. kinematics convergence passes;
5. waveform boundaries pass;
6. whitening validation passes;
7. the microbenchmark passes;
8. exactly 4096 accepted pairs are published;
9. all 32 shards are complete;
10. resume is byte-identical;
11. output is below 10 GB;
12. at least 100 GB remains free;
13. no scientific split or training uses the dataset;
14. no GWOSC/GWTC data were accessed;
15. no legacy or smoke artifact changed;
16. full production remains unauthorized.

At completion:

- inspect Git status and diff;
- update project state, decisions and failures;
- commit with:

  `feat: qualify production generator on 4096 engineering pairs`

- push `phase3a/generator-qualification`;
- close the phase gate;
- stop for human review.

Do not merge automatically.
Do not begin full production.
Do not train a model.
