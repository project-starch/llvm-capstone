# Is an STC ever the FIRST access to a fresh page? Enumerated from source, with the counter-examples

This gates the record boots. Answer, in one line each:

* **first access to a fresh PAGE: YES — `memcpy` is a confirmed counter-example.**
* **first access to a fresh SLOT (and so to a fresh arena): NO — `enframe` reads the slot prefix
  first, with a plain byte load.**

Those two together say the **narrow design is sufficient** and eager pinning across the board is
not needed. Detail and evidence below.

## 1. `memcpy` — CONFIRMED counter-example

`memcpy` is at `0x11c84` in `contract.dom` (`125705371837ea92…`). The path to its destination `stc`:

```
11ca0: xor  a3, a4, a3      ; src ^ dst
11ca4: andi a3, a3, 0xf
11ca8: beqz a3, 0x11cb4     ; same 16-byte residue -> continue
11cc0: andi a3, a3, 0xf     ; dst & 15
11cc4: beqz a3, 0x11d08     ; dst 16-ALIGNED -> skip the byte-align loop entirely
11d10: bltu a2, a4, 0x11d30 ; len >= 16 -> fall through
11d18: ldc  a5, 0x0(a1)
11d20: stc  a5, 0x0(a3)     <-- FIRST store to the destination
```

With a 16-aligned destination, a source of the same residue, and length ≥ 16, **no `sb` or `sd`
touches the destination before that `stc`.** (The `sd ra` and `stc s0` in the prologue go to the
*stack*, not the destination.) A freshly `malloc`'d, granule-aligned buffer is exactly that case.

## 2. `memset` — NOT a counter-example

`memset` at `0x11d68` contains one `stc`, and it is `stc s0, 0x0(sp)` — the prologue frame-pointer
spill to the **stack**. Its fill loop is `sb a1, 0x0(a3)`, a plain byte store. **A corrected count:
an earlier pass of mine reported "memset: 1 stc" as if it were a counter-example; reading the
order shows it is not.**

## 3. The allocator does **not** pre-touch a granted arena

* `__capstone_malloc_map` returns `(void *)sublet_base(&pending_mapping)` and writes **nothing**
  into the mapping.
* `__capstone_malloc_group_attach` → `bind_slots`, and `bind_slots` writes only
  `slots[i].parent` / `slots[i].issued` — the **external** slot array, not the mapping. It
  *computes* `base + hdr + i*stride` and skips the group header rather than writing it.

So the arena reaches mallocng untouched; nothing on the module or glue side gives it a plain first
touch at grant time.

## 4. But mallocng's first access to a slot is a plain READ

`enframe` (`meta.h:321-333`):

```c
int off = (cap_prefix(p)[1] ? *(uint16_t *)(cap_prefix(p)+2) + 1 : ctr) & 255;
assert(!cap_prefix(p)[0]);
```

The first access to a fresh slot is a **plain byte load** of the slot prefix. That is a cause-13
fault, and `resolve()` always pins with `FOLL_WRITE`
(`FOLL_WRITE | ((v->vm_flags & VM_WRITE) ? 0 : FOLL_FORCE)`), so an **absent** page is faulted in
*as a write* by GUP's populate and gets **D set**. A first touch therefore never leaves a page in
the D-clear state.

**This is also the boundary of the S-24 retraction**, and worth stating precisely because the two
look alike:

| page state | what GUP does | D afterwards |
|---|---|---|
| **absent** | calls `handle_mm_fault` with `FAULT_FLAG_WRITE` | **set** |
| **present, D clear** (e.g. after `mprotect`) | returns the page; `follow_page_pte` touches only the `struct page` | **still clear** → the S-24 loop |

So S-24 is specifically the *already-present* case — consistent with the board lane's "the store
after `mprotect`" — and not a first-touch problem at all.

## 5. Paths that DO give a plain first touch, for completeness

* the growth path writes the header with plain stores:
  `*(unsigned long *)header = (unsigned long)r->group; ((unsigned char *)header)[8] = 0;`
* calloc-style zeroing: `memset(backend_pointer(r), 0, r->nominal)` — plain `sb` loop (§2).

## 6. What this decides

**The narrow design is sufficient: keep grants lazy, and have the first kind-4 fault in an arena
pin the WHOLE arena.** The arena's first fault is `enframe`'s plain prefix read, so the pin happens
on a plain access; every later page in that arena is already pinned when a `memcpy` `stc` reaches
it, and §1's counter-example cannot fire.

**Eager pinning at grant time is NOT required**, which avoids the 201.7 MiB / 242-grant cost and
keeps the sparse steps' semantics — see
`2026-10-18-eager-pin-so-no-stc-faults-on-vp2.md`.

The one residual, stated rather than hidden: an arena whose **very first** access is a `memcpy`
`stc` — i.e. a caller that `memcpy`s into a slot without the allocator having read its prefix —
would still expose §1. `enframe` runs on the allocation path for every slot it hands out, so this
requires a pointer obtained some other way; I have not found such a path, and I have not proved
there is none.

## 6b. Why the QEMU kit cannot show S-24 — it is the PTE's STATE, not trap delivery

One wrong explanation is recorded here so it is not rediscovered. I argued in correspondence that
on QEMU a D-clear store fault inside a context is delivered to the **guest kernel's** handler,
which sets D transparently. **That is refuted**, at the source, and the refutation is the RTL
lane's: `cpu_helper.c:1813` calls `capstone_supervisor_trap()` **before any delegation** and
returns with `exception_index = RISCV_EXCP_NONE` when it consumes the trap, and
`capstone_supervisor.c:466-476` maps causes 12/13/15 to kind 4 under options bit 2 and then
`escape(...); return true;`. So a trap raised while a context is active never reaches the guest
kernel, on QEMU exactly as on the RTL (`commit_stage.sv:263-268`). QEMU's `svadu=false` walker also
does fault a store on a clear D (§6a / `cpu_helper.c:1124-1131`).

**So the difference between the two targets is the PTE's state at the fault, not how the fault is
delivered** — which is what §4's table already says, and what the measurement supports: with
`fault_repeat_max=1`, which trips the bound on the *first* repeat, the suite gave **34/41 unchanged
and ZERO `-ELOOP` lines**. (I had pre-registered that the pass count would *drop* under that bound.
It did not, and the non-drop is additional evidence the repeat branch is never reached on QEMU.)

The pre-registered readings for `resolve_pte`, which must now be read on **both** targets:

* **board:** `D = 0` at the loop, with V, R, W and A set;
* **QEMU:** either *not present* at the first store, or `D` already **1**.

The likely mechanism for the difference, to be confirmed by those prints rather than asserted: on
QEMU the page is created by the write itself, so GUP's `faultin_page` → `handle_mm_fault` →
`do_anonymous_page` makes the PTE dirty; on the board it was read-touched first (the zero page or a
clean anon page under `PROT_READ`, then `mprotect` RW: present, writable, D clear) and
`follow_page_pte` sets nothing. The launcher's touch order and the two kernels' populate paths
decide it.

The fix — `fixup_user_fault` with `FAULT_FLAG_WRITE` for a present page — is correct on both
targets regardless of which way that falls.

## 7. Not a verdict on the RTL candidate

This says what the software does. Whether a faulting STC retires with no exception and leaves a
ghost store-buffer entry is the RTL arm's reading, and nothing here should be taken as evidence
for or against it.
