# Direct-target Stage A plan

## Scientific delta from adaptive RC.3

Version `1.1.0-rc.4` changes one scientific contract: Stage A training systems
are sampled directly from the frozen pre-selection evaluation target rather
than RC.5 or proposal-v3. The estimand, posterior target, lens/source support,
selection, observations, waveform/PSD contract, split counts, learning-curve
rule and architecture-selection rule are inherited unchanged.

For every Stage A training record,

```text
q_train(theta) = p_eval(theta)
log q_train(theta) = log p_eval(theta)
log w(theta) = 0
w(theta) = 1
```

Ordinary conditional negative log probability is therefore exactly the frozen
importance-weighted objective evaluated with unit weights. Proposal-v3 and
RC.5 weighted scientific training are closed. Density and weight fields remain
privileged provenance and cannot become deployable inputs.

## Materialization

Stage A contains exactly:

| Split | Accepted systems | Atomic shards | Systems per shard |
| --- | ---: | ---: | ---: |
| train | 32,768 | 256 | 128 |
| validation | 6,144 | 48 | 128 |
| total | 38,912 | 304 | — |

The first 16,384 ranked training systems remain the probe subset. Generation
continues to 32,768 regardless of the 16k probe result. Calibration, SBC, final
evaluation and the conditional 65k extension are not part of Stage A.

Train and validation use distinct roots, attempt namespaces and dataset IDs.
All source, lens, physical-system, pair and noise IDs must be group-disjoint.
Phase 3A, Phase 3C, canary and legacy IDs are forbidden.

## Disposable canary

Before Stage A, a separately authorized execution canary produces exactly
eight train-namespace and eight validation-namespace engineering pairs. Both
use `generator_qualification` split semantics and `dataset_purpose:
execution_canary`, never scientific split labels.

The canary uses the same direct-target sampler, lens solver, waveform,
selection, observation, storage and validator paths as Stage A. It stops after
the first namespace, resumes under the same commit/environment, and proves the
first shard byte-identical. It may inspect schema, arrays, unit weights,
storage, telemetry, manifests and checksums, but not throughput or ESS.

## Release and execution sequence

1. review RC.4 and its canonical hash;
2. freeze a clean generator commit;
3. build a wheel and immutable environment identity;
4. separately authorize and run the 8+8 canary;
5. review the canary manifest and hash;
6. separately authorize exactly 32,768 train plus 6,144 validation systems;
7. run `python -m gwlens_mm.release_gate phase4`;
8. create official identities only from a ready release certificate;
9. materialize, validate and atomically publish Stage A;
10. open probe training only under a later training authorization.

The conservative RC.5 planning bound is 56.48 active hours, 42.28 GB
published and 64.53 GB peak project storage for Stage A. These are projections,
not direct-target measurements, and the canary cannot lower the resource gate.
