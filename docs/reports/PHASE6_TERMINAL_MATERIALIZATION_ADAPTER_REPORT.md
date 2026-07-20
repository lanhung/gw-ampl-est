# Phase 6/7 terminal materialization-adapter report

## Outcome

The future calibration/SBC and sealed final-evaluation materialization paths
now understand the terminal 131,072-system training reference without changing
their frozen counts, seeds, distributions, numerical-validity commitments or
unsealing rules.

Historical `corrected_65k` authorization parsing remains intact for audit
replay. A new `terminal_131k` mode requires:

- exact terminal combined, train-increment, validation and development-tail
  manifest hashes;
- corrected 65k as a strict immutable subset;
- exactly 131,072 train, 6,144 validation and 512 development-tail systems;
- direct-target `q=p` and unit-weight semantics;
- one of the two preregistered terminal size decisions and its SHA-256;
- the exact twelve-result architecture decision and locked rung 131,072;
- no extension above 131,072.

For future calibration/SBC leakage validation, the streaming reference set is
expanded from corrected train+validation to terminal train+validation+the
development-tail pool. It must contain exactly 137,728 group-disjoint systems;
the five rejected base systems remain excluded and the five replacements remain
included. The development-tail pool is never training data.

For future final materialization, the sealed-data runner accepts five training
reference roots in terminal mode (the corrected four-root view plus the terminal
increment) and requires exactly 131,072 training groups. Calibration-fit and SBC
remain separate published references, and the final 20,480 cases remain sealed.

## Verification

- materialization-focused tests: 27 passed;
- full local suite: 388 passed, seven optional dependency skips;
- maintained-scope Ruff: passed;
- mypy: passed for 65 source files;
- Python script compilation: passed;
- sdist and wheel build: passed.

The frozen implementation commit is
`45d05287fbd9a8b7f9bc1999b749be5c521d7931`; the exact wheel SHA-256 is
`bc5d3cd2fd6f898b08590be7f348dc4970edb7fe5f23f4422ffc29185336f4cd`.

No official Phase 6/7 identity was created. No active staging path, publication,
checkpoint, calibration/SBC system or final case was opened.

## Remaining gate

Execution remains blocked until the terminal train and tail parents publish,
the 131k probe and architecture decisions complete, and separate exact gates
bind every manifest, checkpoint, wheel/environment and output identity.
