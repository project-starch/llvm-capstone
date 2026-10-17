# Pre-registration — boot 4 on the serialized module (one supervisor slot)

**Written and pushed BEFORE the boot.** Cite this commit from whatever reports the result. Nothing
here may be edited afterwards; corrections go in the result.

Sibling documents: `PREREGISTRATION-boot2-module-179686bc.md`,
`PREREGISTRATION-ceiling-row.md`, and `RESULT-q21-two-sided-on-strict-csr-pin.md` whose Tier 1 and
Tier 2 readings this one does not repeat.

## The change being predicted

Q-26. vp2 has ONE saved-context slot; a CSRUNV action 0 is refused with cause 29 whenever the slot
is used and the request is not a resume of the same frame, and in S that refusal took the kernel
with it on boot 2. The module now knows the limit instead of discovering it:

* a frame occupies the slot from its first entry until it terminates — `started && !terminal`, the
  same predicate `ensure_nodes()` already uses to find a collection owner;
* `CV_THREAD_CREATE` is **refused with `-EBUSY`** when admitting the frame would exceed
  `supervisor_slots` (default 1). The caller is itself a live frame, having escaped to make the
  ioctl, so on one slot any in-domain thread creation is refused;
* a not-yet-started frame of a **different context** WAITS on a global queue instead, because two
  vexec processes never join each other;
* the same-context case is rejected rather than queued, since a live sibling is the thing that
  would have to finish first — that is the join deadlock, and refusing at CREATE is what avoids it.

**Refusing at CREATE and not at the join is the load-bearing choice.** If the frame is admitted and
cannot run, `contract.dom`'s `threads` busy-waits 100,000,000 iterations on a value the child will
never set and then fails an assertion that never mentions slots, and `heap-threads` blocks in `join`
for ever. Both are illegible. `-EBUSY` fails at the domain's own
`require(tid > 0, "thread create")`.

## Which steps need a second live context — THREE, measured

`mode-thread-scan.py` greps each of `contract.c`'s mode bodies, resolving one level of callee, with
`threads` as the control that proves it fires:

| mode | thread creation |
|---|---|
| `threads` | direct |
| `heap-threads` | direct |
| **`vm`** | **via `vm_contract()`** — an RX-mapping test that enters a thread |

An earlier classification of mine named only the first two, from the mode NAMES and run.py's
expected exit codes. It was wrong by one step, and the measured scan is why this table is here
rather than a sentence.

## The predicted reading, measured on QEMU first

run.py on the one-slot pin, module `4e68302973484bd1`, images `125705371837ea92` /
`a5942a1add489597` / `a902e740fc5d3b0f`, QEMU `809d1fc107e64515` (the pinned 3b8040d5ff6e copy):

```
34 / 41
false: threads, vm, heap-threads,
       threads_output, vm_contract, heap_threads, shared_heap_pages_released
cause 29: 0      Scalar(0x1d): 0      kernel panic / Oops: 0
```

* the three refused steps exit **1** — the domain's own `require`, not a trap, not a hang;
* four derived checks go false with them (`threads_output`, `vm_contract`, `heap_threads`,
  `shared_heap_pages_released`), which is the cascade the board lane warned about;
* **`parallel_one:0` and `parallel_two:0`** — the cross-context wait serialises two processes and
  both still print their results, so they are NOT "unavailable on one slot";
* `preempt:143`, every deliberate fault still `139`, `sqlite_write`/`sqlite_read`/`mruby` `0`, and
  the run **reaches `cleanup:0`** — where before the queue `threads` ended the guest at step 2.

**34/41 is numerically the same figure the board lane derived for the unserialized fallback, and it
is a different 34.** Theirs came from a run truncated by the panic, with `cleanup` absent; this one
completes and names seven specific false checks. Do not read one as confirming the other.

## What would refute the design

* **any cause 29** — the module still issued an entry the hardware refuses, which is the one thing
  this change exists to make impossible;
* a **hang** in `threads`, `vm` or `heap-threads` — refused too late, at the join rather than the
  create;
* **`parallel_one` or `parallel_two` failing** — the cross-context wait is wrong and processes are
  being refused where they should queue;
* a kernel oops or panic anywhere.

## Not predicted here

The board ko for boot 4 is `d33435143f6cf754` (FPGA kernel, same source). It has not run on
silicon, and the QEMU reading above cannot settle the switcher's resume semantics, which only the
board can: discard-and-re-enter is not a resume (FP state and `mstatus` are reset and the pc is not
advanced), and this design never relies on it — it queues rather than discarding.
