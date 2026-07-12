# Phase 3A implementation gap audit

Checkpoint audited: `2bb1ea43cb50f0fbaf8eea9f88750a90603b596a`.

This checklist distinguishes code and tests from execution evidence. Nothing is
considered qualified merely because it is specified in documentation.

| Master-prompt section | Implemented at checkpoint | Tested at checkpoint | Missing implementation | Execution evidence required |
|---|---|---|---|---|
| 1 Authorization | Fail-closed loader, exact frozen hash and denial checks | Local authorization mutation tests | Verify authorizing commit and branch in runner | Saved preflight identity |
| 2 Remote resources | Disk/PSD preflight helper | PSD helper pending AutoDL runner test | Environment freeze, collision and path manifests | Host/package/memory/disk/PSD record |
| 3 Production pipeline | Independent package and pair-at-a-time shard writer | Structural unit tests only | End-to-end accepted-count runner and dataset publisher | Peak RSS and bounded-memory measurement |
| 4 Phase 3A config | Exact 4096, 32x128, resource and numerical gates | Authorization tests | Complete runtime validation | Frozen config hash in every artifact |
| 5 Alpha.3 schema | Environment, dynamics and detector PSD fields; alpha.2 reader | Schema round trip and alpha.2 tests | Selection metadata and qualification-specific validation | Alpha.3 records from final run |
| 6 Lens/MST | SIE/EPL adapter and MST helper | Analytic and optional solver fixtures | Proposal sampler and connected generator use | Saved MST contract results |
| 7 Source/waveform | Smoke-only waveform components exist | Smoke engineering tests | Frozen source sampler and 8-second production waveform engine | Boundary and runtime evidence |
| 8 Waveform boundary | Criteria frozen in config | None | Deterministic boundary suite | JSON fixture summary |
| 9 PSD/noise/whitening | PSD hash verifier; separate array schema | PSD hashes previously verified | Production noise/whitening diagnostics | Detector summaries and hard-gate result |
| 10 Selection | Thresholds only in preregistration | None | Clean-signal SNR selection and rejection provenance | Acceptance/rejection tables |
| 11 EM/environment | Alpha.3 typed fields | Schema validation | Eight-cell noisy observation generator | Balanced-cell counts and mask checks |
| 12 Galkin | Frozen fixture test | One AutoDL reference fixture | Per-system process-local forward model and balanced convergence suite | Convergence CSV |
| 13 Microbenchmark | Count/resource gates in config | None | Benchmark orchestrator and metrics | 32 accepted pairs and projections |
| 14 Shards/publication | Atomic 128-pair shard writer/checksums | Unit structure only | Cross-shard validation and atomic dataset publication | 32 complete shard manifests |
| 15 Resume | Immutable complete-shard verifier and attempt journal | Journal tests | Three-shard stop/resume orchestration and hash capture | Pre/post byte-identical hashes |
| 16 Validation | Existing schema/array/split/policy validators | Broad local suite | Streaming dataset-level validator and summaries | Final validation JSON |
| 17 Outputs | Required paths specified | None | Evidence export and report generation | All committed small artifacts |
| 18 Completion | Local test commands available | 120 local and 124 authoritative tests before this audit | Build/final tests/state updates/close gate | Final commits and clean status |

The gap audit authorizes no execution by itself. Generation remains blocked
until all missing implementation is committed and every pre-microbenchmark
gate passes.
