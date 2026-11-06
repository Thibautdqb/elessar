# Elessar 1.2 release review

The review below was made on the 1.2.0rc1 candidate; 1.2.0 adds the checks listed first in CHANGELOG.md.

This branch prepares 1.2 and keeps publication separate from validation. No tag or package-index upload
is created by the preparation task, and main is not changed.

## Review scope

- Earlier reproduced training/validation, transformation, matrix p-value, ADF and binomial defects.
- Conservative serial Kendall/categorical/mixed inference and separate final-consensus evaluation.
- TTS numeric-target omnibus and missingness tests, preserving temporal grids and indicator semantics.
- Provenance-based exclusions and an explicit exact-alias quarantine policy, retaining legitimate
  near-deterministic predictors.
- Exact finite-sample median intervals, scale-stable moments and subnormal harmonic means.
- Correct integer CDF gaps at both sides of observed jumps; post-fit diagnostic p-values are no longer
  presented as calibrated goodness-of-fit significance.
- Repeated tests, OS/Python coverage, declared dependency floors and clean wheel/sdist installations.

The [scientific contract](SCIENTIFIC_CONTRACT.md) gives assumptions and unavailable/approximate cases.
It is the scope of the candidate's claims, rather than a promise about arbitrary data-generating processes.

## Required release gates

| Gate | Evidence required |
|---|---|
| Ordinary checks | Two complete runs per supported CI combination, no failures or unacknowledged skips |
| Statistical checks | Complete fixed benchmark suite plus two independent release seed streams; finite p-values on supported calibration designs and unchanged declared bounds |
| Numerical references | Before/after regressions and independent coefficient/coverage/CDF/statistic references |
| Compatibility | Linux Python 3.10–3.14, Windows/macOS 3.12; exact declared runtime floors on 3.10 |
| Packages | Strict build metadata check; fresh isolated wheel and sdist imports, CLI and public report API |
| Documentation | Current contract, changelog and API/report labels consistent; historical measurements preserved |
| Review/publication | Candidate review and explicit approval before merging main or publishing |

CI artifacts contain JUnit XML, numerical measures and dependency snapshots. A configured gate is not
reported as passed until its actual job succeeds. Publishing workflows depend on ordinary/package and
statistical workflows, including both seed streams, before any index-upload job may run.
