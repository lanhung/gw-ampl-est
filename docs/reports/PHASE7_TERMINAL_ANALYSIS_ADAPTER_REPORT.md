# Phase 7 terminal analysis-adapter report

## Outcome

The two preregistered input-ablation fits and the RC.7 non-neural simulation
reference can now consume a future logical 131,072-system publication without
rewriting their historical corrected-65k paths.

For terminal ablations, a later exact gate must bind the terminal size decision,
selected architecture, 131k preparation/membership, all terminal publications,
training wheel and CUDA environment. Both GW-only and EM-only views use the
same 131,072 systems, target, architecture, optimizer, budget and seeds 0/1/2.
The primary model is not retrained and no new architecture may be introduced.

For the non-neural reference, the bank is built from all 131,072 training
systems grouped by exact lens family and EM cell. The existing 256-neighbor,
4,096-draw kNN/KDE score contract is unchanged. The validation/final query
boundaries and the statement that this is not a likelihood or gold posterior
are unchanged.

The independent 512-case development-tail pool is not included in either
ablation training or the reference bank. It remains only a terminal scale
diagnostic.

## Verification

- terminal ablation/reference focused tests: 17 passed;
- full local suite: 392 passed, seven optional dependency skips;
- maintained-scope Ruff: passed;
- mypy: passed for 65 source files.

The tests exercised both legitimate 131k terminal labels and the historical
blocked execution paths. No scientific dataset, checkpoint, query or final case
was opened.

## Remaining gate

Ablation fitting and reference execution remain unauthorized until exact
post-architecture gates bind the publications, decisions, standardizers,
software/environment and output identities. Final unsealing and GWOSC/GWTC
remain separate gates.
