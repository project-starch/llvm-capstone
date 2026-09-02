# Search prompt

Replace `<PROGRAM>` and `<OUTPUT_PATH>`. Nothing else changes between runs.

---

Find **memory-safety bugs in `<PROGRAM>` whose memory came from the program's own
allocator rather than from `malloc`.**

CHERI bounds memory when it is allocated and revokes authority when it is handed
to `free`. A program that carves objects out of its own pool and recycles them
internally triggers neither event, so the hardware sees one long-lived block
instead of the objects inside it.

**The bug is not in the allocator.** You are not auditing the pool code, just as
nobody looking for heap bugs audits `malloc` and `free` themselves. You are
looking at ordinary code — parsers, request handlers, callbacks, whatever the
program does — that mishandles memory the pool handed it. The allocator is
healthy and behaving as designed; it is the reason the mistake stays invisible,
not the mistake. So the defect can sit anywhere in the codebase, and the only
question that decides a case is: where did the memory it touched come from?

Every class counts: use after free, uninitialised read, an overflow from one
object into its neighbour inside the same block, a slot reused as the wrong
type. What does not count is a bug that leaves the block entirely, since bounds
still apply at that edge.

**Step 1 — the allocator, from source.** Does `<PROGRAM>` keep its own pool,
freelist, region or GC heap? If not, say so and stop; that is a useful answer.
If it does, produce **one test another person can apply**: "an object is in the
arena when …". Quote `file:line`. Say roughly what share of allocations go
through it — a program where nearly everything does is worth far more than one
where it is a side path.

Read the allocation and free paths; do not count call sites. A single `malloc`
for a bulk block that is then carved into objects is an arena, and counting
would call it ordinary heap.

**Step 2 — apply that test to the bug history.** CVEs, vendor advisories, the
project's commit history, its bug tracker, fuzzing infrastructure. Record bugs on
**both** sides: the ones on ordinary `malloc` memory are the control group, and
without them nobody can tell a blind spot from a broken instrument.

Rules:

- Every entry needs a source URL you actually fetched.
- Never invent a CVE id or a commit hash. Leave the field empty instead.
- `UNKNOWN` is a correct answer when the record does not say where the memory
  came from. A wrong classification is worse than a missing one.
- Do not assume an uninitialised read traps: where an arena does not scrub on
  free, the stale bytes can still be a valid pointer.
- No personal names anywhere in the output.

**Output** — a JSON array at `<OUTPUT_PATH>`, one object per bug:

    {"id","year","component","bug_class","affected_function","trigger","poc",
     "fix_commit","fixed_in","alloc_arena","cheri_expectation","rationale",
     "verification","source","notes"}

`alloc_arena`: the arena's name, `system-malloc` for a control, or `UNKNOWN`.
`cheri_expectation`: `BLINDSPOT-candidate` | `CAUGHT-candidate` | `ARENA-DECIDES`.
`verification`: how you know — `read-the-source` > `named-in-advisory` >
`inferred-from-file` > `unverified`.

**Reply with**: the step 1 test, the share estimate, counts per verdict, and the
three most reproducible cases. The data goes in the file.
