# mruby's GC object heap under Sublet

*Design record, 2026-09-26, against mruby 4.0.0-rc2 (`9d523e2f74f2`), `src/gc.c`.
The port it plans is the third arm of `ports/mruby/musl` (`MRBD_HEAP`): level0,
then `sublet` (mruby's bodies on the runtime's revoking heap, done, mrbtest
unchanged), then this one, where every GC object slot is issued and revoked on
its own. The method is `slab-sublet-port-plan.md`'s: the three rules, and where
the allocator breaks each, function by function. The interface is
`capstone/runtime/include/sublet/sublet.h`.*

## Why the second arm is not enough

On `MRBD_HEAP=sublet` a GC heap page is one block of the Sublet heap: 1024
object slots and a header, 81,984 bytes on capstone64 (`RVALUE` is 80 bytes,
16-aligned; measured with the port's compiler). The page is revoked only when the
sweep finds it entirely dead and frees it. Until then a collected object's slot
goes on the page's free list and the next `mrb_obj_alloc` hands the same bytes to
a new object, while every stale `mrb_value` naming the old one still carries a
valid capability to them. That is the defect class most of the 4.0.0-rc2
inventory's GC-slot rows belong to (11 of 24), and the one ASan cannot see. The
third arm revokes the slot when the sweep frees it, so a stale reference dies with
its object and does not live on into the next one.

## The three rules, and where `gc.c` breaks each

### Rule 3 -- a revoked slot cannot be read or written

The GC reads free slots in five places, all of which must stop:

| Site | Reads | Replacement |
|---|---|---|
| `init_heap_page` | writes `as.free.tt`, `as.free.next` into every slot | sidecar state = FREE, sidecar free list |
| `mrb_obj_alloc` | pops `freelist`, reads `p->as.free.next` | pop a sidecar index |
| `incremental_sweep_phase` | `is_dead()` reads `gc_color`/`tt` of every slot, free ones included; pushes the freed slot with `as.free.next` | skip sidecar-FREE slots before touching them; push the index |
| `gc_gray_rescan`, `free_heap`, `gc_each_objects` | read `tt` of every slot | skip sidecar-FREE slots (the callbacks of `gc_each_objects` -- objectspace, memsize -- already skip `MRB_TT_FREE`, so they see the same set) |
| `obj_free` under `MRB_GC_STRESS && MRB_DEBUG` | `memset(obj, -1)` after freeing | fill before the revoke, through the slot's own alias |

`mrb_object_dead_p` and `heap_p` can be handed a pointer to an object that is
already dead (that is their purpose); `is_dead` then reads it. They answer from
the sidecar: the page and index come from the object's address as an integer,
and the state from the sidecar, without a load through the pointer.

### Rule 1 -- a capability cannot reach out of the object it was given for

`mrb_obj_alloc` returns `&p->as.basic`, derived from the page's capability, so
every object's capability spans the whole page. Each slot's capability must be
its own: the alias `sublet_take` returns for that slot, 80 bytes. The GC keeps a
copy of each live slot's alias in the sidecar for its own traversals (sweep,
rescan, `free_heap`), and that copy dies with the revoke like every other.

`heap_p` compares addresses; it keeps doing so on integer addresses (the page
base as a `ptraddr`), which derives nothing.

### Rule 2 -- a revocation cannot spare part of what it covers

The page header (`freelist`, `next`, `free_next`, `old`, `region`) sits in front
of the slots in the same allocation. Slot regions are carved from the slot array
only, so a slot's revoke covers 80 bytes and never the header; the page's own
handle, senior to every slot's, is what frees the page whole.

## Where the page comes from

A slot can be split off and revoked only by an allocator holding the page
linearly. The Sublet heap's `malloc` hands out aliases, not linear blocks, so
the GC heap needs a source of its own: the runtime's second program region
(`hostcall.c` parks two under `CAPSTONE_PROGRAM_REGIONS`; the FFmpeg pool port
uses region 1 the same way). The GC takes it with `__capstone_region(1)` and
lends itself pages from it; the libc-test host helper needs a second grant beside
`LT_HEAP_REGION_BYTES`.

## Function by function

- `add_heap`: a page from the GC region; the slot array is split into 1024
  80-byte regions under a handle senior to them all (`sublet_handle` first, then
  `sublet_split`), recorded in the sidecar; all slots FREE.
- `mrb_obj_alloc`: pop an index, `sublet_take` its slot (mrev + delin), zero it
  through the alias, record the alias, state LIVE.
- `incremental_sweep_phase`: for LIVE slots as now; a dead one is `obj_free`d
  through its alias, then `sublet_give` (revoke), state FREE, index pushed. A
  wholly dead page is given back through its senior handle.
- `gc_gray_rescan`, `free_heap`, `gc_each_objects`, `heap_p`,
  `mrb_object_dead_p`: consult the sidecar as above.
- `mrb_gc_add_region` (the embedder's own buffer): not under Sublet. The arm
  refuses it rather than mixing revoked and unrevoked slots in one heap.

## Cost, measured before it is designed around

The second arm already spends 259,253 revocation nodes on mrbtest, four times
silicon's 65,532 per boot; per-slot issue adds a split per slot and an mrev per
allocation. QEMU's node pool is sized by `CAPSTONE_REV_NODES`. The arm is for
the defects' matched pairs first; whether it fits a board run is a separate
question, answered by counting, not assumed.

## Validation this port needs

- mrbtest on the third arm with the same result as on level0 and on `sublet`
  (the GC's own traversals are where a missed free-slot read would fault).
- Each GC-slot defect of the 4.0.0-rc2 inventory as a matched pair: the arm
  with and without that defect's upstream fix, the second arm as the control
  that must not see it.
