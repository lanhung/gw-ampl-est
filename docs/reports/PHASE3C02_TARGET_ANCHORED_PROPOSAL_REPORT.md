# Phase 3C-0.2 target-anchored proposal report

Status: **latent preflight passed; stopped for human review; A/B unauthorized**.

Phase 3C-0.2 started from merge commit
`9b9a6a3fcad1487622a2ec1d37e592fe0301e4e6`. Adaptive RC.3 and rejected
proposal-v2 RC.1 evidence remained unchanged.

## Identities

- version: `proposal-v3-target-anchored-mixture-1.0.0-rc.1`;
- config hash: `2d7998ca099c1ecddbb5d9cb1d824f37d3d398826a88831b2bccddbda814cbf4`;
- draws: 200,000 each for RC.5 and v3;
- v3 replay: `650a9ca7831d1342b06cfb89ab645cf10c9294f4cd917ea49ccc1a25f70fc8d5`.

## V3 gates

| Metric | Result | Gate |
|---|---:|---:|
| finite fractions | 1.0 | 1.0 |
| mean weight | 0.99836 | [0.98,1.02] |
| overall ESS | 0.78532 | >=0.50 |
| SIE ESS | 0.79184 | >=0.40 |
| EPL ESS | 0.77886 | >=0.40 |
| maximum weight | 1.80537 | <=1.81818 |
| anchor failures | 0 | 0 |
| replay | identical | required |

The theoretical certificate gives population ESS >=0.55 overall and per
family. Empirical results exceed it.

## RC.5 diagnostic

RC.5 mean weight was 0.99239 and overall ESS 0.11776; SIE/EPL ESS was
0.15222/0.09484. This is diagnostic only. Factor marginals locate the primary
variance in lens structure (ESS 0.19855) and external convergence (0.58682),
not redshift, source plane, masses or orientation, which are target-identical.

## Verification

- full pytest: 186 passed, three optional Lenstronomy skips;
- v2/v3 proposal-focused tests: 23 passed;
- maintained-scope Ruff: passed;
- mypy: passed for 32 source files;
- source distribution and wheel: built successfully;
- v3 configuration hash, replay, RC.1 immutability and result arithmetic passed.

## Safety and stop

Zero waveform pairs were generated; accepted-pair, lens, selection and A/B
runners were not called. No training or GWOSC/GWTC access occurred. V3 passing
does not authorize A/B. Phase 3C-0.2 stops for human review.
