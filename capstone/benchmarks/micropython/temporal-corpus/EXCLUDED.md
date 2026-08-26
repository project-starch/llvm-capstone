# Excluded from the corpus

`cases/` holds only the rows whose temporal classification is CERTAIN. These are
the rest. They are kept, with their evidence, because a quietly dropped row cannot
correct anything, and one of them corrects a reading of the results: MPY-T25 halts
with `cause 24`, exactly like the temporal rows that fault, so `cause 24` does not
distinguish a lifetime defect from an uninitialised one.

Verdicts come from the fix commit's own message where one exists. The original
`class` column was assigned from issue titles, and the 2026-08-18 audit found a
third of it wrong.

| case | source | verdict | why |
|---|---|---|---|
| `MPY-T03` | CVE-2026-1998 | no | fix: 'Make import-all support non-modules'; type confusion, and NVD says CWE-119/787 not 416 |
| `MPY-T08` | #12670 | no | fix: 'casts away the constness and assigns -1 to the object fd member'; writing a const ROM object |
| `MPY-T17` | #3627 | no | a deadlock between a finaliser exception and the GIL; liveness, not memory safety |
| `MPY-T18` | #19413 | no | closed NOT_PLANNED and redirected to the LVGL binding project; not a MicroPython defect |
| `MPY-T20` | #6988 | uncertain | an ESP32 collector panic with no fix commit and no mechanism recorded upstream |
| `MPY-T21` | #8550 | uncertain | reporter records only a silent stop without gc.collect(); no mechanism established, closed as no longer reproducible |
| `MPY-T22` | #17442 | uncertain | a data race on allocator state; whether it manifests as a lifetime violation is not established |
| `MPY-T23` | #12638 | uncertain | suspected freed lwip buffer, but closed as unreproducible and the thread ends in an ESP-IDF version question |
| `MPY-T24` | #11781 | no | fix: 'embed: Improve stack top estimation'; a stack overflow, no lifetime component |
| `MPY-T25` | #10402 | no | subclassing a stream in pure Python is unsupported; the C stream state is never initialised |
| `MPY-T26` | #18645 | no | a malformed .mpy drives relocation off a NULL table; input validation |
| `MPY-T27` | #5272 | no | a socket is never cleaned up after a failed connect; a leak |
| `MPY-T28` | #322 | no | gc_realloc refusing an allocation is an allocation failure, and it was measured not to reproduce |
| `MPY-T30` | #11698 | no | gc.collect retaining more than expected is over-retention, the opposite of premature free |
