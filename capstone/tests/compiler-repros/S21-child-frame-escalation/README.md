# S-21 — the positive control for the child-frame escalation

`s21-escalate.c` is the control the S-21 commit said was owed. It discriminates: the escalation
SUCCEEDS on a pre-fix module and is denied on the fixed one.

## The reading

Both arms on the pinned 3b8040d5ff6e QEMU, `x-capstone-supervisor-slots=32` and
`supervisor_slots=32`, because at one slot the create is refused and the test says nothing.

| module | markers | exit | fault |
|---|---|---:|---|
| `309adef6dc002a01` (pre-S-21, `b04983b244e1`) | … `S21_CHILD_LIVE S21_BEFORE_WRITE` **`S21_WRITE_SUCCEEDED`** | **0** | none |
| `cb4a8d758fd55d41` (with S-21) | … `S21_CHILD_LIVE S21_BEFORE_WRITE` | **139** | `cause=25` |

So the domain really can write its live child's frame at +1408 without the fix, and really cannot
with it. The faulting address in the denied arm is `frame + 1408` exactly.

## What it does, and why each piece is there

It replicates `thread.c`'s frame setup rather than calling the public thread API, because the public
API does not hand the frame pointer back and the pointer is the whole point. Every stage prints a
marker before it acts, so a failure names itself.

**Three iterations, each of which produced a VOID arm — kept here because each is a trap worth
recognising.**

1. **Child stack inside the frame page** (`frame + 2048`). The child ran on a stack in the very page
   the module pins and retires: `cause 12 pc 0`, and nothing about S-21. The stack must be separate
   memory.
2. **Child returning with x1 unset.** `child_entry` returned, `ra` was 0, so it jumped to 0 —
   `cause 12 pc 0` again, which looks exactly like a module fault and was a missing return
   trampoline. `thread.c` installs `thread_exit_entry` in x1; this control has the child exit
   directly instead.
3. **Child exiting before the write.** The decisive one. Both arms then read `cause 25` at
   `frame + 1408` — the right address for the wrong reason, because the ordinary teardown had
   already made the page stale. **The instrument fired and could not separate the hypotheses.** The
   child now spins on a shared flag the parent sets after the attempt, so the write targets a LIVE
   child's frame, and the arms separate.

A fourth trap was in the RUNNER, not the program: its marker grep listed markers explicitly, missed
the ones added in iteration 1, and reported "no markers" for a log that plainly had five — which
sent me hunting a startup crash that did not exist. It now matches any `S21_` marker.

## Rebuild and run

    <kit>/out/sdk/capstone-cc -O1 -I<repo>/capstone/runtime/include \
        -I<repo>/capstone/runtime/virtual s21-escalate.c -o s21-escalate.dom

Run it under a module and machine both at more than one slot. `S21_CHILD_LIVE` must appear, or the
run says nothing: without it the write did not target a live child's frame and the arms converge.
