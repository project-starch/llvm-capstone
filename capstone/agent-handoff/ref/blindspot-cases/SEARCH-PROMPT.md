# Search prompt: nested-allocator blind spots in one program

Template. Replace `<PROGRAM>` and `<OUTPUT_PATH>` and run once per program.
Everything else is fixed on purpose — the constraints below were each paid for
by a mistake.

This is the tool that fills the scheme in `TAXONOMY.md`: Phase 1 establishes
facets 1 and 2 (arena owner, arena discipline), Phase 2 fills the corpus. The
output schema matches `sqlite-bugs.csv`, so a run merges with `merge-cves.py`.
`sqlite-arenas.md` is a worked example of what Phase 1 should produce.

---

You are surveying **<PROGRAM>** for a study of memory-safety defects that
capability hardware (CHERI purecap) structurally cannot catch.

**The idea you are testing.** CHERI creates bounds when memory is allocated and
revokes authority when memory is passed to `free`. Software that runs its own
allocator on top of the system one recycles objects without either event, so the
hardware never learns the object died. **The defect itself does not have to be in
the allocator.** It can be anywhere in the program. The only thing that matters
is where the memory it touched came from.

## Phase 1 — establish the arena test, from source, BEFORE looking at any bug

Do not start from the bug list. Start from the allocator. Answer, with
`file:line` you have actually read:

1. **Does <PROGRAM> have its own allocator?** Name it. Typical shapes: a pool or
   region freed wholesale, a freelist of fixed slots, a GC heap with a sweep,
   size-class bins over large chunks.
2. **Is the underlying memory one allocation, or many?** Quote the call that
   obtains it and the code that carves children out of it.
3. **What happens on free?** Quote it. The decisive question is whether the
   system `free` is ever called at the granularity of a single object, or only
   when the whole arena is torn down.
4. **How do you tell, for an arbitrary object, whether it lives in the arena?**
   State this as a test someone else can apply — an allocation function name, a
   size threshold, a type. This test is the deliverable of Phase 1.
5. **Dominance: what share of the program's allocations go through it?** An
   estimate with reasoning is fine; say how you got it. This number decides how
   much of the bug history is in scope, so it matters more than any single row.
6. **What takes an allocation OUT of the arena?** Exhaustion, a disable flag, a
   size threshold, growth past a slot, a debug mode. List every one you find.
   These are what a later measurement has to control for.

If <PROGRAM> turns out to have no arena, say so and stop. That is a useful
negative result, not a failure.

## Phase 2 — intersect with the defect history

Only now go to the bug history: CVE databases, the project's own advisories and
commit history, its bug tracker, distro security trackers, fuzzing infrastructure.

For each memory-safety defect, apply the Phase 1 test and record which side it
falls on. **Include both sides.** Bugs that touched ordinary heap memory are not
noise — they are the control group, and a corpus without them cannot tell a blind
spot from a broken instrument.

## In scope

A defect counts as a blind-spot candidate when **both** hold:

- the affected object's memory came from the arena, **and**
- the failure mode is one where recycling matters:
  - **temporal-reuse** — access after the object went back to the arena
  - **spatial-intra-arena** — a write crossing from one sub-object into a
    neighbour inside the same underlying allocation
  - **uninitialised-recycled** — a read of arena memory still holding the
    previous occupant's bytes

## Explicitly the control group, not the result

Record these, marked as controls: out-of-bounds **past** the whole arena block;
double free through the real allocator; any defect on ordinary heap memory; null
dereference. These must be catchable, and a measurement run that fails to catch
them is broken rather than interesting.

## Traps that have already cost this project real work

- **Read the right arm of the `#ifdef`.** A default configuration was once
  reported from a line that only compiles when a feature is disabled. Check which
  branch a default build actually takes before quoting a constant.
- **Counting allocator call sites is not an arena test.** One component was put
  on the safe side because it calls the system allocator — and it calls it once,
  for a bulk block it then carves into objects. Read the allocation path; do not
  count.
- **"No fuzzer issue" is not evidence of no bug.** One project's fuzz target does
  not link its extension code at all, so silence there says nothing about those
  components. Before treating absence as a finding, establish that the instrument
  could have produced the opposite result.
- **Do not accept a commit hash you have not resolved.** A fix hash offered in a
  summary turned out not to exist in the repository. Leave a field empty rather
  than fill it with something plausible.
- **Do not confuse "the bug is in the allocator" with "the object came from the
  allocator".** The second is the criterion; the first is rare and not required.
- **A stale slot may hold a still-valid pointer.** Where an arena does not scrub
  on free, an uninitialised read can return a usable capability rather than
  untagged junk. Do not assume such a read traps.

## Output

Write a JSON array to `<OUTPUT_PATH>`. One object per defect:

    {"id","year","component","code_category","bug_class","affected_function",
     "trigger","poc","fix_commit","fixed_in","alloc_arena","cheri_expectation",
     "rationale","verification","source","notes"}

- `alloc_arena`: the arena's name, or `system-malloc` for a control, or
  `UNKNOWN` when nothing in the record identifies the allocation site. **`UNKNOWN`
  is a correct answer.** A wrong classification is worse than an absent one.
- `cheri_expectation`: `BLINDSPOT-candidate` | `CAUGHT-candidate` |
  `ARENA-DECIDES` | `N/A-not-heap`
- `verification`: how you reached it — `read-the-source` > `named-in-advisory` >
  `inferred-from-file` > `unverified`. Be honest; the column is how a reader
  knows what to trust.
- Every row needs a `source` URL you actually fetched. No researcher or reporter
  personal names anywhere in the output.

Return a short summary only: the Phase 1 arena test and dominance figure, counts
per `cheri_expectation`, the three most reproducible candidates, and anything you
could not determine. The data goes in the file, not in the reply.
