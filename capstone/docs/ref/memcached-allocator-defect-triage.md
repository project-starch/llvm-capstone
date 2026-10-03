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
in-process fixtures 12–16 on 2026-10-02). **It does not, however, unblock the page-mover group,**
and an earlier version of this paragraph said it was "worth re-reading against it" and "the
highest-value memcached work available". That overstated it: the slab arms run with the mover
**disabled**, `-o no_slab_reassign`, and the port's own results say why — *"the page mover is not
hooked: it walks a page by pointer arithmetic, which per-chunk bounds refuse"*
(`ports/memcached/app/results/2026-10-01-qemu-slab-sublet/README.md`). Reaching those seven
defects needs the mover hooked, which is **port work, not corpus work**, and is the real
next step for memcached's nested class.

## WIDENED 2026-10-03: the "0" re-checked two-sided, and the wording filter measured

**The zero stands, and it is now a tested negative rather than a one-branch reading.** The earlier
check looked only at `origin/master`. memcached develops on `origin/next`, so that was the same
single-branch mistake that cost the Wireshark search 4,239 commits. Re-checked:

| | |
|---|---|
| `origin/master` | `2d51e36` — still exactly the pin, **0** commits after |
| `origin/next` | `853112d`, 2026-07-20 — **1** commit ahead of master, `slabs: minor cleanup`, nothing lifetime-related |

So no fixed-upstream-after-the-pin candidate exists for memcached, on either branch. Every memcached
case must be historical.

**And a lesson about the instrument, which is why the counts below are not an inventory.** A
lifetime-vocabulary grep over the whole history returns **13** commits. That list is *not* a superset
of the 9 candidates the 2026-09-21 subsystem search found: the intersection is **2** (`7af02b0`,
`0ad4de6`). The corpus's three strongest cases — `59bd02ce29`, `a8c4a82787`, `152ddb68f7` — and four
of its rejections are **absent** from the 13. **Three of the five built cases would never have been
found by wording.** Read subsystems and diffs; treat a wording filter as a sampler, never as a
population.

Of the 11 unused members of those 13: two were worth building and became app fixtures 17 and 18 (see
the next section); five are rejected with reasons — `3eb7773` touches `slabs.c` but is a
wrong-*metadata* bug where both arms read freed chunks anyway, `34e4604` reverses into a NULL
dereference that cannot discriminate the arms, `683bb98` and `f8a55c4` sit on substrate deleted
before the pin, and `d195dfe` is not an ancestor of 1.6.45 at all (its `daemon/` layout is gone);
one, `acdfe1a`, is **not a defect** — a `configure.ac` change that matched the filter only because its
message says "use after frees, double frees". The remaining three are proxy-side and blocked with the
rest of the proxy.

**No unused member of the 13 is a slab (nested-class) defect.** The only nested shape among them is
`5267f14`, a double return into the proxy's own rctx cache, and it needs a Lua state. That is
consistent with the deferral below rather than a new finding.

## The plain-heap half — now built, as the CONTRAST rather than as filler

~~a plain `malloc`/`free` defect in memcached would have no home today~~ — it has one: the port's app
fixtures. Two were added on 2026-10-03, and the reason is specific rather than "more cases".

Fixtures 12–16 end their objects **inside** memcached's own allocators — `cache_free` pushes onto a
STAILQ, `do_slabs_free` onto a class's slots list — so the runtime's `free` is never called, `sublet`
cannot revoke, and the expect file registers `sublet` as the **negative control that must RETURN**.
That silence is the nested-allocator blindness claim. But a silence is only attributable if the same
arm is shown to speak when the release *does* reach the allocator. That is what the two new fixtures
are for:

- **17**, commit `204019d` (a 2006 contributed patch): `realloc` moves the connection read buffer and
  only the base pointer is updated, leaving the parser's interior cursor in the released block. Same
  *object* as fixture 12, different allocator seam.
- **18**, `e779381` logger: the close routine clears the global slot and frees the watcher; the caller
  writes `w->failed_flush` through its own stale pointer. The corpus's only **write**-after-free.

Both are plain `malloc`/`calloc`/`realloc`, so `sublet` must FAULT on them — and does. Results:
[`../../ports/memcached/app/results/2026-10-03-qemu-plain-heap-contrast/`](../../ports/memcached/app/results/2026-10-03-qemu-plain-heap-contrast/README.md).

The older caution still applies and is worth keeping: a defect ASan already reports is **not**
evidence for the mechanism this project claims. These two are controls, and are written as such. What
they add is that the nested rows' silence is now attributable.

## Counts

| | |
|---|---|
| commits after the pin | 0 — no live-in-pin candidate is possible |
| history already searched | 2,360 commits, 2026-09-21 |
| in scope then | 9 — 5 built, 4 rejected with reasons |
| deferred pending a port | 18; the 7 page-mover defects stay blocked until the mover is hooked |
| live in the pin today | 1 (case 02), unchanged |
