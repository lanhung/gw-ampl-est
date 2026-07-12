# Proposal-v3 target-anchored executable specification

Status: latent-only preflight passed; A/B remains unauthorized.

Proposal `proposal-v3-target-anchored-mixture-1.0.0-rc.1`, configuration hash
`2d7998ca099c1ecddbb5d9cb1d824f37d3d398826a88831b2bccddbda814cbf4`, is

```text
q_v3 = 0.20 q_rc5 + 0.55 p_eval + 0.25 q_central.
```

The evaluation component samples every frozen pre-selection target factor
exactly. The central component differs only by using independent sigma-0.8
truncated-normal source coordinates. The RC.5 component retains broad support.
All use the same 0.5/0.5 lens-family probabilities and no selection
conditioning.

Because `q_v3 >= 0.55 p_eval`, weights satisfy `w<=1/0.55` and population
relative ESS is at least 0.55 overall and within either lens family, subject to
the explicit normalization/support assumptions in the certificate.

The 200,000-draw preflight passed: mean weight 0.99836, overall ESS 0.78532,
SIE/EPL ESS 0.79184/0.77886, maximum weight 1.80537, zero anchor failures and
byte-identical replay. Passing is latent evidence only and does not authorize
A/B, waveform generation or scientific use.
