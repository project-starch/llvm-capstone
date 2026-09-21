# Allocator-level survey

One host recording per **ported allocator**, on the revision its port pins.
Nothing here has been recorded yet. What is here is the map, the consistency
check and the driver, so that a recording, when it happens, cannot quietly
cover fewer allocators than the manuscript claims.

    ./check-levels.py --manuscript ../../../nested-allocators-paper
    ./run-survey.py --plan

## Why

The manuscript's section 2 asks how often an object dies inside a block the
system allocator still owns. Its table answers that for **one allocator per
program**, and the manuscript now ports **seventeen allocators across seven
programs**. Six of the seventeen are covered today, and one of those six
aggregates four PostgreSQL managers into a single `context` level, so the
motivating table and the evaluated set are not the same set.

Two further mismatches follow from that. The surveyed revision is not always
the ported one, SQLite 3.42 against 3.53.3 and PostgreSQL 17.5 against 17.0,
which is why the manuscript currently prints two revision columns. And
Wireshark, whose four allocators carry twelve of the corpus's defects, has no
host recording at all.

## What the map adds that one row per program cannot

Fifteen levels sit directly on the system allocator. **Two do not**, and they
are the point:

| level | its level 0 | what it shows |
|---|---|---|
| `lookaside` | `memsys5` | a slot released while the buddy allocator beneath still owns the block |
| `apr_bucket` | `apr_pool` | a bucket node returned while the pool beneath still owns the node |

A release at either level is invisible not merely to libc but to the *custom
allocator underneath it*. The present survey cannot express that, because it
records one level per program and calls everything below it `libc`. The
existing Apache wiring says so in its own comment: it notes that
`apr_bucket_alloc` is not measured, and that counting one of a program's two
allocators can only understate the share of invisible frees. This branch is
that sentence being acted on.

## The map

`levels.json` carries the seventeen levels. Each one names the program, the
allocator, the level name the instrument compiles in, the level beneath it,
the pinned upstream and its source files, the port that does the pinning, the
workload, and the seam: which function allocates, which releases, and which
ends a container in one go.

Every seam is marked `port-patch` when it was read out of a port's patch,
`port-document` when out of a port's or a corpus's README, and `to-confirm`
otherwise. Nothing in the map is currently `to-confirm`. A build is what turns
either of the first two into a fact, and if a build disagrees the map is
corrected before the wiring is.

`check-levels.py` enforces that the map and the manuscript hold the same
seventeen allocators, that no level sits beneath itself, that a level naming a
port claims it is either present or waiting on a named pull request, and that
at least one level still sits on another custom level. `--self-test` breaks
the map six ways and requires every check to fire.

## What it waits on

Seven of the seventeen levels name a port that is not on this branch's base:

| levels | waits on |
|---|---|
| the four `wmem` levels | PR #73 |
| `memcached-slabs`, `memcached-objcache` | PR #71 |
| `apr-bucket` | PR #74 |

The map records that rather than assuming it. `run-survey.py --plan` prints
what blocks each level, and a level whose port is absent is refused by name.

## What is not here

**The wirings and the recipes.** They are per program, each is a build and a
run, and neither can be written honestly against a source that has not been
fetched. `wiring/README.md` states the contract and, more importantly, which
of the two level-0 shapes each level needs: the two nested levels must use the
instrument's *recorded* shape rather than interposing malloc, or they measure
the wrong boundary and say nothing new. `recipes/README.md` states what the
driver hands a recipe and what it must leave behind.

**Any measurement.** No numbers are produced here and none may be quoted from
this branch. The manuscript's section 2 keeps its present table until a
campaign has run and been reviewed.

## Relation to the existing survey

The instrument is not copied here. `run-survey.py` resolves the manuscript's
`experiments/a1/hook` and records its hash in every manifest, because two
copies of a counting rule are two rules. The report layout is the one
`scripts/survey_metrics.py` already reads, so a recorded level needs a row in
that script's specs and nothing else.

The existing W1 and W2 recordings are not superseded by anything here. They
stay as they are until a level actually replaces one, and a replacement says
so in its own manifest.
