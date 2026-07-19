# Phase 6 corrected-reference gate report

## Outcome

The future calibration/SBC materialization path now binds and validates the
same immutable corrected 65k training view used by the active scientific probe.
No calibration/SBC pair, official identity, release certificate, checkpoint or
statistic was created or opened.

## Resolved gap

The earlier implementation streamed the original Stage A and Stage B parents
when checking future group leakage. After the waveform incident, those parents
contain five superseded systems and omit the five replacement systems. A future
calibration or SBC draw could therefore collide with a replacement identity
without being detected by that incomplete reference set.

The release gate now requires four authorization-bound atomic parents:

- Stage A train and validation;
- Stage B train extension;
- the combined-base 65k reference;
- the waveform-correction overlay.

It resolves those parents through the typed correction contract, requires the
known parent/tree/view hashes, two Stage A plus three Stage B exclusions, and
two plus three replacements. Official identities remain null unless this exact
reference and every later size/architecture gate pass.

## Execution validation

Before publication, the runner streams the Stage A and Stage B base records and
both correction replacement datasets. It excludes exactly the five frozen bad
physical-system IDs, requires every exclusion to exist, rejects duplicate pair,
source, lens, system, noise and augmentation-parent IDs, and requires exactly
71,680 distinct reference systems: 65,536 corrected train plus 6,144 unchanged
validation. The new calibration/SBC namespaces must be group-disjoint from that
logical reference and from each other.

## Safety

The original Phase 6 data configuration and all preregistration hashes are
unchanged. Materialization, checkpoint access, calibration fitting, SBC,
final-evaluation access, model tuning and GWOSC/GWTC remain false. This change
is prospective software hardening under the existing implementation-only gate.

## Verification

- Phase 6 plus shared overlay focused tests: 26 passed;
- full local suite: 330 passed, 7 optional dependency skips;
- Ruff: passed;
- mypy: 58 source files passed;
- source distribution and wheel build: passed.
