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

- **Collect as many candidates as you can, each as fully filled in as you can.**
  Volume and detail are the goal of this pass.
- **No unknown origins.** A bug goes in when you can say where its memory came
  from; if you cannot establish that, leave it out. But establishing it means
  **reading the allocation site in the source** — find the fix diff, see which
  object the bug touched, follow that object to the call that allocated it. It
  does **not** mean building the program or running a proof of concept. Do not
  narrow the table to cases you were able to execute.
- **Read diffs through `github.com/sqlite/sqlite`-style git mirrors** where a
  project's own web interface blocks automated access. The affected function and
  file come out of the diff, and without them most rows cannot be placed.
- Fill every column you can: the fix commit, the commit that introduced the bug
  if the record names one, the affected version range, whether a proof of
  concept exists publicly and where. Leave a field empty rather than guess it.
- Every entry needs a source URL you actually fetched.
- Never invent a CVE id or a commit hash. Leave that field empty.
- Do not assume an uninitialised read traps: where an arena does not scrub on
  free, the stale bytes can still be a valid pointer.
- No personal names anywhere in the output.

**Output** — a CSV at `<OUTPUT_PATH>`, ready to open in Excel. Write it with a
CSV writer, not by hand, so commas and quotes in the text cannot break it.
One row per bug, these columns:

    id, year, component, bug_class, affected_function, trigger, poc,
    fix_commit, fixed_in, alloc_arena, verdict, evidence, source, notes

- `alloc_arena` — the arena's name, or `system-malloc` for a control-group bug.
- `verdict` — `blind` or `caught`.
- `evidence` — `read-the-source` (you followed the object to its allocation
  site), `named-in-advisory` (the advisory says where the memory came from), or
  `measured` (you ran it). `read-the-source` is the normal case and is enough.

**Reply with**: the step 1 test, the share estimate, how many rows are `blind`
against `caught`, the three most reproducible cases, and how many bugs you had
to leave out because their allocation site could not be established. The data
goes in the file.
