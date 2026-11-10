# Normalized defect classification

`classification.json` classifies all 425 declared corpus records.  Counts exclude the two
duplicate records and five withdrawn records, leaving 418 unique active defects.

| memory-safety axis | active defects |
|---|---:|
| temporal | 263 |
| spatial | 139 |
| other memory safety | 11 |
| not memory safety | 4 |
| both spatial and temporal | 1 |

The axis describes the violated property; `root_cwe` separately describes the causal weakness.
For example, a type confusion that produces an out-of-bounds read is spatial with root CWE-843.
`manifestation_cwes` records the resulting CWE-125.  A use-after-free that happens to read out of
bounds remains temporal, not both; `both` requires independently evidenced violations.

## Method and controls

`tools/classify-cases-gemini.py` enumerates cases through each `corpus.json`, never by recursively
counting `case.json`.  That distinction excludes retired and auxiliary records.  Two independent
Gemini reviews classify each unique active defect from its structured record, provenance and
trigger or reduction.  Exact agreement on both axis and root CWE is accepted.  Disagreements receive
a third adjudication.  The 70 PHP sanitizer-derived CWEs are authoritative controls, as are explicit
withdrawal and duplicate records.  Five conflicts between model reasoning and existing axis labels
were resolved directly from the committed provenance; the generator carries those decisions as
named manual overrides.

The resulting review-status ledger is:

| status | declared records |
|---|---:|
| two-pass model agreement | 234 |
| third-pass adjudicated | 109 |
| existing authoritative record | 77 |
| manually adjudicated from repository evidence | 5 |
| unresolved | 0 |

The most common root classification is CWE-416 (182 active defects).  Thirty-three active defects
have no defensible root CWE and deliberately use `null`; the artifact contains the complete CWE
distribution, final evidence, confidence, rationales and review status.  Independent raw votes
remain in the resumable `$CAPSTONE_TMP_ROOT` checkpoints rather than the repository.

Raw CLI sessions and checkpoints stay under `$CAPSTONE_TMP_ROOT` and are not committed.  The API key,
session identifiers and usage metadata are never written to the artifact.
