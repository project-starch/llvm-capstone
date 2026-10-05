# Results, 2026-10-04

Summaries. Raw serial and console logs are not committed (SCHEMA rule 6).

| arm | detected | hang | no-marker | not-reached | not-run | silent |
|---|---|---|---|---|---|---|
| `spatial` | 10 | 3 | 2 | 0 | 0 | 28 |
| `sublet` | 34 | 1 | 1 | 0 | 3 | 4 |
| `cheribsd-revocation` | 2 | 0 | 0 | 3 | 12 | 26 |

`silent` means the arm's own mechanism did not report. On
`cheribsd-revocation` the oracle column says whether the defect site was
reached and whether the defective access itself was witnessed, which is the
difference between a usable negative and no information at all. On `spatial`
there is no defect-site probe yet, so a silent row there means the exit code
was clean and nothing more.
