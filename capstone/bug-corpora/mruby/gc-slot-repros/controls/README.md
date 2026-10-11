# mruby GC object slots: parent, sibling and successor controls

The mruby row of the pre-registration
[`docs/plans/2026-10-11-nested-lifetime-controls.md`](../../../../docs/plans/2026-10-11-nested-lifetime-controls.md)
(`53325bba0d5c`). The security table shows that patch 0008 rejects stale accesses to mruby's GC
object slots. These controls ask the other half: does it leave the permitted accesses alone? The
nested allocator is mruby 4.0.0-rc2's GC. `mrb_obj_alloc` hands out an object slot of a heap page,
and the sweep releases the slot. Under patch 0008 (`MRB_CAPSTONE_SUBLET`) each slot goes out as
a CDERIVE child of the page's lifetime `page->life`. The sweep revokes that child (CREVOKE) once
`obj_free` has run, and `page->issued[i]` records the child.

| mode | what it does | predicted, protected | predicted, release |
|---|---|---|---|
| `parent` | allocate A, drop it, full GC; confirm A was freed (`tt` = `MRB_TT_FREE` read through the page, and under 0008 `issued[i]` = NULL); read and write A's slot through the page pointer the GC uses (`mrb->gc.heaps` → `page->objects[i]`), and under 0008 also through `page->life` | completes | completes |
| `sibling` | B (kept reachable) and A on the same page; free A by GC; read B through B, check its contents, write B, read it back | completes, B intact | completes, B intact |
| `successor` | free A; allocate until `mrb_obj_alloc` hands out A's slot address as C; require C = A; read and write through C | completes | completes |
| `stale` | as `successor`, then read through A's old pointer at the probe `stale-read-A` (a separate process) | fault, cause 25, at the probe | completes and reads C's data |

The driver is `controls.c`, one image per arm with all four modes. It embeds mruby, links the
arm's own `libmruby.a`, and drives the GC through its API. `mrb_str_new` makes the object: a
32-byte string is embedded in its 80-byte slot, so it is one `mrb_obj_alloc`. `mrb_full_gc` runs
the sweep.

## A is unreachable, and that is checked

mruby's collector is precise. `root_scan_phase` (`gc.c`) marks:

- the global variables and the GC arena;
- the built-in classes, `top_self` and `mrb->exc`;
- the VM stacks of `mrb->c` and `mrb->root_c`, and the task queues.

It never scans the C stack, C registers or C globals, so there is no conservative scan for a stray
copy to feed. A is allocated between `mrb_gc_arena_save` and `mrb_gc_arena_restore`, so its arena
entry is dropped. It is never stored in a Ruby variable, an object, a global or the VM stack. The
only copies of its pointer are the driver's C variables; `stale_A` is a C global that no root scan
reads.

That A was freed is then read, not assumed. Through the page, A's slot holds a string before the
collection (the layout check) and `MRB_TT_FREE` after it. Under 0008 `issued[i]` is NULL after
it. The driver runs up to three full collections and stops with exit 75 if the slot is not free
by then. Every run needed one.

B and C are kept by the arena, which is a root. During the reuse search (`successor`, `stale`)
`mrb->gc.disabled` is set, so no allocation of the search can run a collection. Every candidate
stays allocated until A's slot comes back. It came back at the first allocation in every run.

## The page reference, and the layout mirror

The GC reaches a slot as `page->objects[i]`, from the page pointers it keeps in `mrb->gc.heaps`.
That is what the sweep, `gc_page_of` and `free_heap` use, and the driver walks the same list. The
heap page and `RVALUE` layouts are private to `gc.c`, so the driver mirrors them verbatim from the
pin (with 0008's two fields under `MRB_CAPSTONE_SUBLET`). `build.sh` compiles the driver with that
define exactly when libmruby has it. At run time the mirror is checked against the object
`mrb_obj_alloc` returned:

- `page->objects[i]` has the object's address;
- its type is `MRB_TT_STRING`;
- under 0008, `issued[i]` is that object.

A mismatch stops the run with exit 75. A's page must also hold `Object`, which is a root, so the
sweep can never free the page and the page reference stays the allocator's own.

## Probes and the transcript

Each access the controls make is one line in a `ctl_probe_*` function (noinline). That line prints
`PROBE <name> controls.c:<line>` right before the access. A run therefore records where each
access happened, whether it completed or faulted. The launcher's fault pc is attributed to a
function from the image's symbol table, and to a file:line from its DWARF (`-g`).

Each output line is a single `write(2)`. The last line gives the number of lines printed before it
and the number of failed writes. `run.py` records whether the transcript arrived whole, and
`collect.py` gives no reading for a mode whose transcript did not. This check exists because
stdio output lost whole lines in the pilot boots (`results/2026-10-11-controls/README.md`,
deviations).

Exit status: 0 means every check held; 1 means a data check failed; 75 means the setup could not
be made, which is a precondition failure and not a reading.

## Files

- `controls.c`: the driver.
- `build.sh <mruby tree> <SDK> release|protected <OUT.dom>`: builds the driver against one
  `build-mruby-domain.sh` build. It uses `probe/build-capi.sh`'s flags, plus `-g`, plus
  `-DMRB_CAPSTONE_SUBLET` for the protected arm.
- `run.py`: one boot through `runtime/virtual/run-staged.py`. It runs the four modes, each as its
  own `capstone-vexec` process, and writes `rep.json`.
- `collect.py --meta META.json --out results.json REP.json...`: gathers the boots and judges each
  prediction. A prediction holds only when every repetition shows it; a missing or incomplete
  reading is "no reading".

Results: [`../results/2026-10-11-controls/`](../results/2026-10-11-controls/README.md).
