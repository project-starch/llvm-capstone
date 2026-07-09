# Single-domain held-cap BORROW-REVOKE probe (row3 Option B)

The faithful, literal form of the row3 "after" case. **One** domain receives a
**real monitor-granted linear capability** over a region, mints its own `MREV`
handle, uses an alias derived from it, `REVOKE`s at a lifecycle point, and the
cached alias **faults**.

This is stronger than the two artifacts that came before it:

| | arena comes from | who revokes | entities |
|---|---|---|---|
| `borrow-revoke-uaf-probe` (Option A shape) | `REGION_QUERY` → a raw base address, ambient deref | the **lender** (monitor-mediated) | 2 |
| `capstone-mrev-codegen` (task-005 codegen spike) | `csdebuggencap` hand-mint | the domain | 1, firmware-free |
| **this probe** | the real `REGION_SHARE` delivery path | the domain | **1** |

## The receive protocol

The plan asked where the delivered capability lands on the domain side and what
`lcc` index the entry stub needs. The answer is that **no entry-stub glue is
needed at all**, and there is no index to guess:

* `sbi_capstone.c`'s `shared_region_annotated()` ends with
  `d = __domcallsaves(d, CAPSTONE_DPI_REGION_SHARE, r)`, which *enters* the domain
  with the linear region capability `r` in the capability-argument register.
* `capstone/my_first_domain/start.S` already saves it (`stc a1, sp, 80`) and
  reloads it as `domain_main`'s **first argument** (`ldc a0, sp, 80`). The DPI
  function code arrives as the second, scalar argument.

So the delivered capability **is** `domain_main`'s `arg` on the entry whose
`func == CAPSTONE_DPI_REGION_SHARE (1)`. See `probe_domain.h`.

Existing runtime probes discard it only because they are **`.smode` payloads**
under the `sbi.dom` scaffold. There, `handle_dpi()` → `dpi_share_region()` parks
the capability in the *scaffold's* `regions[]`, and the S-mode payload reaches the
bytes through ambient `cpmp` authority after a `swap_cpmp()` miss. A `.smode`
payload never sees a capability, and it is built with Buildroot gcc, which has no
capability builtins. A `domain_main`-style `.dom` — Capstone clang,
`my_first_domain/start.S` — needs none of that. Hence:

* domain payloads here are `.dom` images built by `build-domain.sh`;
* the controller loads them with `create_dom(path, NULL)` — **no `.smode`**.

The arena is parked in a `.bss` capability slot between the `REGION_SHARE` entry
and the `CALL` entry (two separate domain invocations). That is safe: `stc`/`ldc`
duplicate rather than move, and `cap_compress`/`cap_uncompress` round-trip the
type field, so the capability comes back `LIN` — which `helper_csmrev` asserts on.

## Grant annotations, and why they are not free choices

`REV_TRANSFERRED` + `PERM_INOUT`:

* **`REV_TRANSFERRED`** hands the region over linear with **no monitor-retained
  revocation handle**. The domain owns full authority — that is what makes the
  revoke intra-domain. (`REV_BORROWED` also delivers `LIN`, but the monitor keeps
  an `__mrev` senior to it.)
* **`PERM_INOUT` (RW), never `PERM_IN` (RO)**: `helper_cstighten` silently
  **delinearises** a `LIN` capability whose permissions do not allow write
  ("immutable linear capability can be safely invalidated without scrubbing the
  data"). A read-only grant is therefore not `MREV`-able, and since
  `helper_csmrev` *asserts* `CAP_TYPE_LIN` it would abort the emulator rather
  than fault cleanly.

## Probes

| Probe | Expects | Why |
|---|---|---|
| `held_revoke_fault` | **FAULT**, cause 25 at `-O1/-O2`, cause 24 at `-O0` | the mechanism: cached alias of a revoked, monitor-granted capability |
| `held_mem_alias_fault` | FAULT, cause 24 | alias parked in memory; the reload observes the revoked node and drops the tag |
| `held_no_revoke_ok` | `0x2230005e` | control: same sequence, **no revoke** — pins the cause-24 faults on the REVOKE |
| `held_unrelated_ok` | `0x22310033` | the sweep is not over-broad; the domain survives revoking its own arena |
| `held_ambient_miss` | FAULT, cause 24 | provenance: the arena has **no** ambient second path |
| `held_split_sibling_ok` | `0x22350044` | `SPLIT` sub-caps are independently revocable — the memsys5 property |
| `held_protected_value_lifecycle` | **FAULT**, cause 25 / 24 | step-3 scaffold: carve → use → revoke at "finalize" → cached handle faults |
| `held_arena_survives_revoke` | `0x22370077` | step-3 scaffold: revoking one carved value spares the others and the remainder |

### The cause is asserted, not just reported

Cause **25** (`Cap mem access on revoked capability`) means the tag was **intact**
and the revoked rev-tree node stopped the deref — self-proving. Cause **24**
(`Cap mem access requires capability`) only means the tag was gone, which a
*consumed linear capability* also produces (task-005, finding C3). Every cause-24
expectation therefore carries `held_no_revoke_ok` as its control.

### Why the primary probe's cause depends on `-O`

At `-O0` clang spills the alias, so the post-revoke deref reloads it:

```
mrev a3, a2 ; delin a3 ; sb a3, 8(a4) ; revoke a2 ; ldc a0, 0(a0) ; lbu a2, 8(a0)
                                                    ^^^ reload -> tag cleared -> cause 24
```

At `-O1`/`-O2` the alias stays in a register across the revoke:

```
mrev a4, a2 ; delin a2 ; sb a3, 8(a2) ; revoke a4 ; lbu a2, 8(a2)
                                                    ^^^ no reload -> tag intact -> cause 25
```

Both are real faults; only `-O1+` is self-proving. The `-O1/-O2` payloads need the
`CC_Capstone_FastCC` capability-argument fix (task-006, C1).

`held_no_revoke_ok` at `-O2` also shows task-006's **C2** fix carrying weight: its
`MREV` result is never used, and the `mrev` instruction still survives (it would
have been DCE'd when the intrinsic was modelled as `IntrNoMem`), with `rd != x0`.

## Gap found: a revoked region is a landmine for the host mapping

`REV_TRANSFERRED` leaves a **stale duplicate** of the region capability in the
monitor's `regions[]` (`sbi_capstone.c` even says `// TODO: regions[region_id]
should be added to a free list`) and drops the region's `cpmp` entry. So after the
domain revokes that lineage, the next **host** access to the region takes a `cpmp`
miss, and the monitor's `swap_cpmp()` calls `cap_base(regions[id])` on a
capability that now reloads **untagged**:

```
qemu-system-riscv64: ../target/riscv/op_helper.c:666:
    helper_cslcc: Assertion `rs1_v->tag' failed.
```

QEMU aborts. This is a **robustness gap, not a mechanism failure** — nothing about
the revoke is wrong — but it has a direct consequence for the SQLite integration:
**after a domain revokes a granted region, the host must not touch that region**,
and `regions[]` should be cleared (or `swap_cpmp` should skip untagged entries).
Both fixes live outside this lane (`swap_cpmp` is the monitor; the `lcc` assert is
the emulator's).

The probes work around it: the controller's post-call read of its Linux mapping is
opt-in (`read-arena`) and used only by the two probes that leave the arena's
revocation node live. `held_unrelated_ok` reads its sentinel back **through the
capability, before revoking**, and reports the verdict in its return value.

## Step 3: memsys5 feasibility (spike, for A to finish)

**Can the domain's SQLite build point `memsys5` at a monitor-granted linear
arena? Yes, trivially. Can SQLite's own heap allocations then be `MREV`'d — the
literal Option B end-state? No.**

`sqlite_capstone_domain.c` today does

```c
static unsigned char sqlite_heap[1 MiB] __attribute__((aligned(16)));
sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap, sizeof(sqlite_heap), 64);
```

Substituting a granted arena for that `.bss` array is a one-liner plus a `delin`,
and memsys5 tolerates it: it never does `inttoptr`. Every allocation is
`&mem5.zPool[i * szAtom]` and every free/realloc recovers the index with a
*pointer difference* `((u8 *)p - mem5.zPool) / szAtom`. The 64-byte `szAtom`
already exceeds the 16-byte alignment `store_capregval` needs. `mem5.aCtrl` lives
inside the arena, past the block area, so the grant must cover both.

What does **not** work is revoking an allocation:

1. `&zPool[k]` lowers to `cincoffset`, and `helper_cscincoffset` does
   `*rd_v = *rs1_v` then bumps only `bounds.cursor`. The derived capability
   inherits the pool's `rev_node_id`, its type, **and its bounds**. An allocation
   is not a distinct capability as far as the revocation tree is concerned — and
   it is not even bounds-narrowed to its own block.
2. So `MREV` of an allocation would mint a node senior to the **pool's** node, and
   the revoke would sweep the whole heap.
3. It cannot get that far anyway. `mem5.zPool = zByte` is a `movc`, and a LINEAR
   capability is consumed by copy (task-005, C3), so the pool must be `delin`'d
   before SQLite may touch it. Allocations then come out `NONLIN`, and
   `helper_csmrev` **asserts** `CAP_TYPE_LIN` — an emulator abort, not a fault.
4. `SPLIT` is the only derivation that mints a fresh revocation node
   (`cap_rev_tree_split`); `SHRINK`/`SHRINKTO` copy `rev_node_id` unchanged. But
   the emulator has **no merge/unsplit op**, so splitting is one-way. A buddy
   allocator that coalesces freed neighbours, which is exactly what memsys5 is,
   cannot be built on it.

### The scaffold

`probe_linear_arena.h` is what A should build the integration on: a separate,
small linear arena out of which the domain carves the values it actually wants to
protect, one `SPLIT` sub-capability each.

```c
probe_arena_init(grant);                         // sqlite3_config(SQLITE_CONFIG_HEAP)
probe_protected_buf b = probe_arena_carve(256);  // this statement's column-name buffer
… hand b.alias to the caller, it is an ordinary pointer …
probe_arena_revoke(&b);                          // sqlite3_finalize()
… every cached copy of b.alias now faults …
```

`held_protected_value_lifecycle` proves the fault; `held_arena_survives_revoke`
proves a second live value and the un-carved remainder are untouched. Carving is
one-way (the arena only shrinks), so this suits a bounded number of protected
values per grant — one per live statement — not a general heap.

**What A still needs to decide:** whether the row3 corpus copies the protected
value into the carved buffer (pragmatic Option B, works today) or the PI's bar
demands `MREV` of SQLite's own `memsys5` pointer, which needs either
per-allocation `SPLIT` plus a new emulator merge op, or an allocator that never
coalesces. That is a corpus-fidelity call, not a mechanism one — hence the stop
here.

## Running

Needs the `rootfs.ext2` write lock (one guest boot at a time across agents);
announce in `agent-handoff/COORDINATION.md` first.

```bash
bash capstone/tests/runtime-qemu/run-intra-domain-mrev-revoke-probe.sh          # -O0 -O1 -O2
OPT_LEVELS=-O0 bash capstone/tests/runtime-qemu/run-intra-domain-mrev-revoke-probe.sh
```

Each probe gets its own boot: a faulted domain poisons later domain creation in the
same guest session. A `domain_main` `.dom` runs in `PRV_C`, whose capability faults
have no delivery path, so QEMU prints `domain halted by capability fault: cause = N`
and exits — the harness run exits non-zero **by design** for the fault probes, and
the driver classifies on the serial log rather than the exit code.
