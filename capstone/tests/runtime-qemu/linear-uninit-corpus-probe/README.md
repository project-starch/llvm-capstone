# LINEAR (row 11) and UNINIT (row 14) corpus probes

The last two deferred rows of the SQLite Stage-2 defect corpus, as mechanism probes
on RTL. With these, **all 17 in-scope rows are validated**
(`benchmarks/sqlite/cve-repros/stage2-mapping.md`).

```
bash ../run-linear-uninit-corpus-probe.sh      # 7 probes x -O0/-O1/-O2, one boot each
```

Requires the `rootfs.ext2` write lock — announce it in `CLAUDE.md (serialize QEMU suites)`
before booting.

## What the rows are

| Row | Class | Stage-1 defect | Stage-2 shape |
|---|---|---|---|
| 14 `cpython_uninit_connection` | null-deref | `*(unsigned char *)connection->db` with `db == NULL` | the `db` capability is **UNINIT**: it names the storage but has no read authority until `sqlite3_open` (`csinit`) |
| 11 `go_double_finalize` | double-free | `sqlite3_finalize(stmt)` twice | `stmt` is a **LINEAR** capability, consumed by the first finalize (`csdrop`); linearity leaves no second copy |

Both were deferred on 2026-07-08 as blocked intra-domain. Both blockers are gone —
`csdrop` is implemented, and a domain can now hold a real LINEAR capability. Neither
row needed a monitor op or a `start.S` change. Trail:
`agent-handoff/history/10-07-2026_02-30-00_linear-uninit-rows-closed.md`.

## Where the capability comes from

Unchanged from `../intra-domain-mrev-revoke-probe/`, whose `probe_domain.h` this
directory includes rather than copies. The controller creates a region and hands it
over with `shared_region_annotated(dom, region, PERM_INOUT, REV_TRANSFERRED)`; the
monitor enters the domain with the linear region capability in the capability-argument
register; `my_first_domain/start.S` reloads it as `domain_main`'s **first argument**.
There is no entry stub and no `lcc` index.

`PERM_INOUT` (RW) is forced: `cstighten` silently delinearises a read-only linear
capability, and `csmrev` then asserts. `REV_TRANSFERRED` is forced: the monitor must
keep no revocation handle, because the whole lifecycle is intra-domain.

## Row 14 — how a domain mints an UNINIT capability

`cap_rev_tree_revoke()` returns `retain_data` — true iff **every** node it invalidated
had already been delinearised. `helper_csrevoke` (op_helper.c:709) turns that into the
retained handle's type:

| the revoked junior run was… | retained handle | cursor |
|---|---|---|
| all non-linear (data retained) | `CAP_TYPE_LIN` | at `base` |
| still linear (data **not** retained) | `CAP_TYPE_UNINIT` | at `end` |

So the only thing separating an UNINIT capability from the LIN one the held-cap probe
gets is a single `csdelin`. Omit it:

```c
void *arena = probe_arena;                          /* LINEAR grant, node N */
void *rev   = __builtin_capstone_cap_mrev(arena);   /* node M, senior to N  */
void *db    = __builtin_capstone_cap_revoke(rev);   /* N was linear -> UNINIT */
```

### The emulator fix this needed

`_helper_access_with_cap` never consulted the capability **type**. An UNINIT capability
was denied only incidentally, by bounds: its cursor sits at `end`, so `*db` is out of
range (cause 5). **A load at a negative offset was in bounds and returned the stale
byte** — the disclosure UNINIT exists to prevent.

The spec permits type 3 as the address operand of the *store* forms (that is how an
owner initialises memory) but of **no load form**: `lb…ld` and `LDC` both list types
`0`, `1`, `5` and raise `Unexpected capability type (26)` otherwise. `csrevoke` +
`csinit` are the reclaim path. So loads through an UNINIT capability now raise cause
26, and `uninit_negative_offset_fault` is the regression test — it returns instead of
faulting without the check.

## Row 11 — `csdrop` consumes a handle, it does not free memory

This distinction is the whole row. `csdrop` clears one register's tag. It does not
revoke a lineage, does not invalidate other capabilities over the same arena, and does
not touch the bytes. Row 11 is not "the allocator refuses a second free"; it is
"linearity leaves no second capability, so the second finalize has nothing to hand
back". `linear_drop_sibling_ok` pins that: the connection the statement was carved out
of keeps working after the drop, and the host reads back the byte written through it.

The statement handle is carved with `cssplit`, which gives it a revocation node of its
own. `SHRINK`/`SHRINKTO` copy `rev_node_id` and are not a substitute (task-005, Q3).

## The probes

| Probe | Expected | Why it exists |
|---|---|---|
| `uninit_use_before_init_fault` | FAULT, cause **26** | row 14's mechanism |
| `uninit_negative_offset_fault` | FAULT, cause **26** | the denial is by TYPE, not by where the cursor sits |
| `uninit_init_then_use_ok` | `0x1412005e` | `csinit` (sqlite3_open) reclaims the same bytes through the same handle |
| `linear_drop_use_fault` | FAULT, cause **24** | row 11's mechanism: use after consume |
| `linear_double_drop_fault` | FAULT, cause **24** | the literal shape: the second `finalize` itself traps |
| `linear_no_drop_ok` | `0x11120033` | control for both cause-24 expectations |
| `linear_drop_sibling_ok` | `0x11130044` | drop ≠ free; host mmap sees the post-drop write |

## Reading the causes

**Cause 26** (`RISCV_EXCP_UNEXP_CAP_TYPE`, "Cap mem load through uninitialised
capability") is self-proving: nothing else raises it. **Cause 24**
(`RISCV_EXCP_UNEXP_OP_TYPE`) only says the register held no capability, which a
consumed linear capability *and* a reloaded revoked capability both produce — hence
`linear_no_drop_ok`. The two row-11 faults are told apart by their log line: the deref
prints "Cap mem access requires capability", the double drop prints "DROP requires
capability" (from `helper_csdrop`).

Unlike `../intra-domain-mrev-revoke-probe/`, **no expected cause moves with the
optimisation level**. Row 14's UNINIT handle keeps its tag and its own valid
revocation node, so a spill and reload changes nothing; row 11's dropped handle is
untagged and stays untagged through a spill and reload.

## Gap that still applies

After a domain revokes a `REV_TRANSFERRED` region, the **host must not touch it**:
`swap_cpmp()` reloads the monitor's stale `regions[]` duplicate untagged and
`cap_base()`'s `lcc` aborts QEMU. Every row-14 probe revokes the arena by
construction, so the controller's post-call read of its Linux mapping is opt-in
(`read-arena`) and used only by `linear_drop_sibling_ok`, which revokes nothing. See
`../intra-domain-mrev-revoke-probe/README.md`, "Gap found".
