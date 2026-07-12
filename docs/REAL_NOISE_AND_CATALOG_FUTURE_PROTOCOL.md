# Future real-noise and catalog protocol boundary

Status: design-only boundary; GWOSC/GWTC access is unauthorized.

Future work proceeds in this order: Gaussian scientific benchmark, real-noise
injections, counterfactual companion injections, catalog pair scan, and only
then model-conditional absolute-magnification analysis for a declared EM lens
candidate.

Before any access, a separate authorization must freeze:

- GWTC release and every product version;
- exact event inclusion/exclusion rules;
- the event-name list and its SHA-256;
- detector and data-quality requirements;
- off-source noise, PSD and preprocessing procedures;
- ranking statistic and pair background;
- multiple-testing correction;
- null-result and candidate-follow-up policy.

The previously discussed 91-event set is a proposed future selection size, not
a fact frozen by Phase 3B. Its exact membership and count must be regenerated
from the reviewed release and criteria. Synthetic OOD authorization can never
implicitly authorize GWOSC/GWTC access.
