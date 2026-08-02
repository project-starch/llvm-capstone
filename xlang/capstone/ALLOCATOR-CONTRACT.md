# The capability-allocator contract for the xlang Capstone column

What `xlang_alloc` / `xlang_realloc` / `xlang_free` must do, in Capstone terms.
Carried forward from `plans/xlang-phase2-seam-TODO.md` §2, rewritten against the
shim route and against the allocator that actually exists
(`capstone/benchmarks/sqlite/revoke_on_free_alloc.h`).

This is the smallest document in the column and the one that decides what it
measures. Read it before touching `mock_mruby_capstone.c`.

---

## 1. `alloc(n)` — mint a capability bounded to exactly the request

`rof_malloc` already does this. Every allocation is a fresh `cssplit` off the
linear arena, so it gets its **own revocation-tree node** and bounds of exactly
`rof_roundup(n)` — not a `cincoffset` into a shared pool, which would inherit the
pool's node and make a revoke sweep the whole heap.

Consequence for us: the 16-byte rounding means a `STACK_BYTES` of 1024 is carved
exactly. No row in the corpus has a geometry that rounding perturbs — every
`STACK_BYTES` in `xlang/cheri/shims/` is already a multiple of 16. **Check this
holds if a row is ever added.**

## 2. `realloc(p, n)` that MOVES — derive new, copy, revoke old

**This is the case the corpus is built on and the one `revoke_on_free_alloc.h`
does not provide.** Six of the fifteen rows (4, 5, 8, 10, 13, 15) are exactly
"the VM register stack was reallocated out from under a cached interior pointer".
There is no `rof_realloc`; it must be written as:

```
new = rof_malloc(n)             /* fresh node, exact bounds */
rof_copy_caps(new, p, min(old_size, n))   /* TAG-PRESERVING — see below */
rof_free(p)                     /* REVOKES p's node */
return new
```

Three things this must guarantee, each a way to get a wrong result:

- **It must always move.** `mock_mruby.c`'s `mrb_stack_extend` aborts with rc 66 if
  realloc grows in place, because then no stale pointer exists and the row would
  report a MISS that says nothing about the mechanism. `rof_malloc` always carves
  a *fresh* region from the top of the arena and never coalesces, so it can never
  grow in place. The row is structurally valid on Capstone — state that rather
  than relying on it silently.
- **The copy must be tag-preserving.** `rof_copy_caps` uses `void *` word moves
  (`ldc`/`stc`), which carry a capability *with its tag*. A byte loop would strip
  the tag off any capability inside the block, so a pointer stored in the moved
  region would come back untagged and fault later — a fault that looks like a
  catch but is an artifact of the copy. The shims' stacks hold `uint64_t` slots,
  not capabilities, so this is currently latent; it stops being latent the moment
  a row stores a pointer in the stack.
- **The returned capability's bounds are the NEW size, not the old.** Growing does
  not extend the old bounds; it returns a different capability with different
  bounds and a different node.

## 3. `free(p)` — revoke

`rof_free` revokes `p`'s node. The arena is **not** reclaimed (one-way `cssplit`,
no merge op), so the slot is marked reusable but the space is not. A
realloc-heavy workload fragments; the shims are short-lived so the arena will
hold, but a long row must check `arena_left`.

---

## 4. The part that decides the benchmark

**What does revocation do to a pointer already cached in a register or on the
stack?**

Row 10 is the worked example. The shim does:

```c
mrb_value *regs = mrb->c->stack;   /* cached */
mrb_funcall_cb(mrb, deep_callback); /* reallocs+moves the stack; old is revoked */
*(volatile uint64_t *)((unsigned char *)regs + 16) = 0xdeadbeef;  /* the defect */
```

If a revoked capability still held in a register faults on use, the row is
BLOCKED-SYNC. If revocation only invalidates memory-resident copies, it is a
MISS. **That distinction is the whole benchmark**, and it is not a property of
the shim — it is a property of the mechanism, which is why it is worth stating
before any number is produced.

### At `-O0` this question is partly dodged, and we must say so

At `-O0` `regs` is spilled to the domain's stack and **reloaded** before the stale
access, so the fault arrives on the reload rather than from a live register. That
is still a real catch — the program did dereference a revoked capability — but it
is *not* evidence that a register-resident capability faults. The honest claim is
"a revoked capability is not dereferenceable", not "revocation reaches the
register file".

We build `-O0` anyway, and must, because at `-O1`+ the compiler hoists the load
before the free or elides the dangling access entirely, so the access the
mechanism must police is never emitted. Both columns build `-O0`; the CHERI
column already does. The limitation is symmetric and belongs in the writeup, not
in a footnote.

### The control that makes a verdict trustworthy

`rof_no_revoke = 1` runs the identical program, allocator and free path **minus
the one revoke**. A row is only BLOCKED-SYNC if:

- the normal build **faults**, and
- the no-revoke control **completes and reports** `MOCK <row> use-after-free-survived`.

Without the control, an `-O0` fault cannot be distinguished from a plain spill
reload of something else. Run both for every row. This mirrors what
`run-sqlite-row9.sh` does and it is not optional.

---

## 5. What this contract does NOT cover

- **Async/quarantine revocation.** Capstone has no analogue of CHERI's deferred
  sweep. `rof_free` revokes synchronously or not at all. That is a fact about the
  mechanisms, and the table reports it rather than inventing a middle config.
- **Hierarchical revocation** (`revoke_on_free_hier_alloc.h`), where freeing a
  parent revokes a subtree. No CHERI counterpart, so it is a Capstone capability
  to report, not a comparison cell. Out of scope for the 15 rows.
- **Row 2.** Stack-use-after-return: no allocator is involved at all, so no
  allocator contract can catch it. Predicted MISS on both columns.
