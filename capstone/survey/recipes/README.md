# Recipes: one build and run per level

`run-survey.py` owns everything the levels share. A recipe owns the one thing
they cannot share: how this program is fetched at its pinned revision, built
with the wiring inlined, and driven by its workload.

A recipe is `bash`, named `<level-id>.sh`, and is called once per repetition.

## What the driver hands it

| variable | meaning |
|---|---|
| `A1_HOOK_DIR` | the manuscript's instrument, holding `a1hook.h` and `a1core.inc` |
| `A1_LEVEL` | the name to compile in as `A1_L1_NAME` |
| `A1_LEVEL_BELOW` | the level 0 name, `libc` or another custom allocator |
| `A1_WIRING` | this level's `.inc` |
| `SURVEY_OUT` | where this repetition writes its reports |
| `SURVEY_REP` | the repetition number, from 1 |
| `SURVEY_REPO` | this repository's root |

## What it must leave behind

One `rep.<pid>` file per instrumented process in `SURVEY_OUT`, in the format
`a1core.inc` writes. The driver refuses a repetition that leaves none, and
records the hash of every file it finds.

## Where the source comes from

Not from the system, and not from a fresh download the recipe invents. Each
ported program already pins its upstream by URL and SHA-256 in the port's
`upstream.json`, and the port's `cmake/prepare-source.py` already knows how to
unpack and patch it. A recipe reuses that pin, so the surveyed revision and
the protected revision are the same artifact rather than two that happen to
share a version string.

The one exception is Wireshark, whose port has no native workload at all: its
`src/native` is a synthetic driver. Its recipe has to build `tshark` and drive
it over the captures the wmem defect corpus names, which is new work rather
than a reuse.

## Build directories

Outside the repository, under the shared temporary root the ports already use.
Sources, builds and raw logs are not committed. What is committed is the
compact report tree and its manifest.
