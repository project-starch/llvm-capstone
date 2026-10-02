# memcached: triage for allocator lifetime defects, and why none can be live in the 1.6.45 pin

The `allocator-repros` corpus had no triage inventory under `docs/ref/`; the generated `INDEX.md`
lists that as a gap, and the corpus README carries the selection instead. This file is that
inventory, for the question asked of all three ported applications: **is there a memory defect on
the upstream tracker that is still present in the version we compile?**

## The decisive fact: our pin IS upstream's head

| | |
|---|---|
| clone | `github.com/memcached/memcached`, full history, 2,360 commits |
| pin | tag `1.6.45` |
| `origin/master` | `2d51e36`, 2026-07-09 — **the same commit the tag points at** |
| commits after the pin | **0** |

So for memcached the live-in-pin question is closed by construction: there is no upstream fix we are
missing, because there is no upstream commit we are missing. **Every memcached case must be
historical** — the upstream fix reversed — except for a defect upstream has chosen not to fix, which
is exactly what corpus case 02 is (`live_in_pin: true`, tail repair, reachable with
`-o tail_repair_time=N`).

This also means the memcached half of any "find bugs still in the shipped code" request is answered
by *re-reading what upstream has not fixed*, not by watching for new fixes. Re-run this check when
the pin moves; it is two commands and it is the whole answer.

## What is already triaged, and is not repeated here

The corpus README records a search of **upstream's whole history to the pin — 2,360 commits** — on
2026-09-21, which found **nine candidates in scope, five built**, and rejected four with reasons.
That search does not need redoing. Its rejections are worth keeping in view because they are the
sharpest statements of what disqualifies a case anywhere in this tree:

- `e3b7d33`, `bc080ab` — the same refcount overflow as case 03 reached through the binary and meta
  protocols. One defect, three front ends: cited in case 03's provenance rather than built twice.
- `f4983b2` and the 2011–12 `do_item_alloc`/`do_item_get` races — that era reused the LRU tail **in
  place**, so the reuse never crosses the allocator's seam and a protected arm would not see it
  either. The case would fail its own oracle.
- `41aa0a5` (2008) — predates most of the structure the port builds.
- `#1213` `do_cache_alloc` NULL dereference — **a NULL dereference faults in the protected and the
  spatial arm alike, so it cannot tell them apart.** A case must discriminate the arms, not merely
  crash.
- `#1306`, `#1308`, and `CVE-2026-90698` (fixed in 1.6.44, before the pin).

## The standing backlog, now partly unblocked

The README defers 18 defects "until another allocator is ported": the page mover (7, including
`d67d187`, which frees busy items deliberately), the proxy's own pools (4), extstore (3), the
logger's bipbuffer (2), the response bundles (1) and the crawler (1).

**The slab port now exists** (`ports/memcached/app`, patch 0006, with the five corpus defects run as
in-process fixtures 12–16 on 2026-10-02), so the page-mover group is worth re-reading against it:
the page mover's whole purpose is to move items between slab pages, which is an allocator-internal
reuse of exactly the kind the slab hooks were built to observe. That re-read is the highest-value
memcached work available and has not been done.

## The plain-heap half

Every corpus in this tree is a nested-allocator corpus, so a plain `malloc`/`free` defect in
memcached would have no home today. Candidates exist — the proxy and the vendored `mcmc` client both
use plain `malloc`, and `#1268` ("Minor ASAN crash-triggering Bug in `_mcmc_token()`") is one — but
note what that implies: a defect ASan already reports is, by construction, **not** evidence for the
mechanism this project claims. It is a control, and should be written as one or not at all.

## Counts

| | |
|---|---|
| commits after the pin | 0 — no live-in-pin candidate is possible |
| history already searched | 2,360 commits, 2026-09-21 |
| in scope then | 9 — 5 built, 4 rejected with reasons |
| deferred pending a port | 18, of which the 7 page-mover defects are now worth re-reading |
| live in the pin today | 1 (case 02), unchanged |
