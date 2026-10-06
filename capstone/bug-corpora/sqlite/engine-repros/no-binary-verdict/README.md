# No binary verdict on every arm

These five are real defects at the 3.22.0 pin, with measured results. They are
parked because on at least one arm they produce neither "detected" nor "not
detected", and the corpus is kept to those two outcomes.

| case | arm | what it produced | why it is not "not detected" |
|---|---|---|---|
| `26_2c7a73eaea` | cheribsd-revocation | probe `hits=0` | the defect site never executed, so the arm had nothing to miss |
| `29_634ac14488` | cheribsd-revocation | probe `hits=0` | same |
| `32_bfe33f80dd` | cheribsd-revocation | probe `hits=0` | same |
| `14_12439f9c16` | spatial | the domain never returned | neither completed nor faulted |
| `27_33cf194218` | sublet | the domain never returned | neither completed nor faulted |

Scoring any of these as "not detected" would credit the arm with a silent miss
on code that did not run, or on a program that never finished. That is the
conflation this corpus exists to avoid, so they are removed rather than
reclassified.

## What removing 26 costs, recorded because it is not free

`26_2c7a73eaea` passes a negative length to `memcpy`, which as a `size_t` is
enormous. It was **the only case in the collection whose damage leaves the
memsys5 arena** -- every other spatial case runs a few bytes past an object
and lands in the neighbouring chunk, still inside the arena. Table 1b's
"leaves the arena" column therefore has no measured member any more, and the
claim that a `malloc`-granular mechanism can see a defect once it escapes the
arena now rests on the SQLITE_ZERO_MALLOC argument alone.

If that column is wanted back, this is the case to repair: it needs a trigger
that reaches `fts3SegWriterAdd` on a real VFS, where the probe currently shows
it does not.
