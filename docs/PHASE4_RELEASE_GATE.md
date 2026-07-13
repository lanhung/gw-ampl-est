# Phase 4 release gate

The only command allowed to create official Stage A identities is:

```bash
python -m gwlens_mm.release_gate phase4
```

During design it must return `blocked_preexecution` and
`official_identities: null`. A ready certificate requires all of the following
at once:

- exact clean branch and generator commit;
- accepted RC.4 version/hash;
- separate scientific execution authorization with exact 32,768/6,144 counts;
- immutable dependency-lock identity and generator wheel SHA-256;
- exact PSD files and hashes;
- sufficient disk and empty publication identity;
- passed 8+8 canary manifest from the same generator commit;
- canary manifest SHA-256 and byte-identical resume evidence;
- model training, calibration, evaluation and GWOSC/GWTC flags false.

Only `ready_for_official_execution` includes derived parent, train and
validation dataset IDs. A blocked result cannot be converted into readiness by
the runner. The canary and Stage A scripts independently recheck their own
authorization and commit identities.

The immutable environment uses
`configs/environment/phase4-autodl-requirements.lock.txt`. Official execution
forbids an editable project install; the reviewed release certificate must
bind the final wheel SHA-256 and Git commit.
