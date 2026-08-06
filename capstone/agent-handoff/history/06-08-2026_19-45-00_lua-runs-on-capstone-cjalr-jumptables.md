# Real Lua 5.4.7 runs on Capstone (cjalr) — root cause was jump tables, not cap-init

**One line.** Real reference Lua now runs a chunk end-to-end on the **cjalr** ABI
(`LUA-OK result=400`), and the block was never the thing the earlier bring-up named:
`.capstone_cap_init` tags Lua's static tables fine; the two real faults were
**absolute-addressed jump tables** (a compiler one, then Lua's own VM one), each fixed
by a build flag the working SQLite build already carried.

## The result (QEMU, cjalr, `my_first_domain/start.S`)

Staged build (`LUA_STAGE=5`, markers survive a wedge), one boot:

```
S1 newstate ok → S2 base ok → S3 load rc=0 → S4 pcall rc=0 → LUA-OK result=400 expected=400
RESULT: PASS (Lua ran; realloc-moved table returned 400)
```

`luaL_newstate` + `luaopen_base` + `luaL_loadbufferx` + `lua_pcall` of
`local t={} for i=1,20 do t[i]=i*i end return t[20]` → 400. Base library registered,
the VM executed, the table's array part was realloc'd and MOVED, the tag-preserving
realloc kept its contents. No gp-captable, no kernel descriptor delivery, no module swap.

## The two fixes (both in `build-lua-domain.sh` LUA_FLAGS)

1. **`-fno-jump-tables`** (+ `-fno-optimize-sibling-calls`). A C `switch` otherwise
   lowers to an **absolute-addressed jump table** read with a plain `lw` through a raw
   (non-capability) address → **cause 24 "requires capability"**. Hit first in
   `lua_gc`'s `switch(what)` and `singlestep`'s `switch(g->gcstate)`.
2. **`-DLUA_USE_JUMPTABLE=0`**. Separate mechanism: Lua's VM loop (`lvm.c`) has its OWN
   application-level computed-goto dispatch (`ljumptab.h`: `goto *disptab[op]` over a
   table of `&&label` addresses), enabled under `__GNUC__` (clang defines it). Those are
   **absolute code addresses with no PCC bounds** → **cause 1** instruction-access fault
   at a raw static pc (`0x79084`, a `&&L_OP_*` label inside `luaV_execute`).
   `-fno-jump-tables` does NOT touch it — it is application data, not a compiler switch.

**Both flags are in the WORKING SQLite cjalr build** (`build-sqlite-capstone.sh`
COMMON_FLAGS has `-fno-jump-tables`; SQLite has no computed-goto VM, so it never needed
`LUA_USE_JUMPTABLE`). Lua's build was simply missing them. This is the SQLite-template
lesson made concrete: the substrate is fine, the port just wasn't carrying SQLite's
codegen flags.

## Retraction — the bring-up diagnosis was wrong

`05-08-2026_06-00-00_gp-captable-lua-bringup.md` states: *"cjalr ABI is a dead end for
Lua. The small-domain `.capstone_cap_init` walk under-tags Lua-scale static capability
tables (base_funcs[], luaX_tokens[]) … the mechanism is wrong for the scale."* This is
**false and was never verified with the diagnostic knob**. Compiling with
`-mllvm -capstone-cap-init-print` shows `.capstone_cap_init` emits a store for **every**
leaf of both tables:
- `luaX_tokens[]` — 37/37 string leaves tagged.
- `base_funcs[]` — 48/48 leaves tagged, **including every function pointer**
  (`luaB_assert`, `luaB_collectgarbage`, …). Arrays-of-structs and merged string
  constants are both handled (`CapstoneCapGlobalInit.cpp`, `collectCapInits` +
  `needsMaterialization` GEP-peel).

The cause-24 fault the bring-up saw at `luaopen_base` was fix #1 (a jump table in the GC
step luaopen triggers), not under-tagging. cjalr is **not** a dead end for Lua; gp-captable
is not required for this.

## How to reproduce

```bash
CAPSTONE_REPO_ROOT=<repo> DOMAIN_EXTRA_DEFS="-DLUA_STAGE=5 -DLUA_DBG_STAGE" \
  bash xlang/lua-cdp/capstone-lua/build-lua-domain.sh
CAPSTONE_REPO_ROOT=<repo> ATTEMPTS=3 bash xlang/lua-cdp/capstone-lua/run-lua-domain.sh
# expect: LUA-OK result=400 expected=400
```
The default (non-staged) build runs the same path via `run_lua()`.

## Method notes (for the next session)

* **pc→static mapping:** `LUA_DBG_BASE` prints the code cap at `domain_main`
  (`Cap(1,0x7,cursor,base,end)`); `load_off = cursor − nm(domain_main)`, `static = pc −
  load_off`. Verified self-consistent (base ↔ segment vaddr 0x10000).
* **QEMU fault message caveat:** the `rs1=xN, imm=M` fields are reliable (they named the
  `lw …, 0(a0)` jump-table load: `rs1=x10, imm=0`); the reported `pc` was a few
  instructions off from the faulting load each time. Trust the operand fields over the pc
  when they disagree.
* **Batch-variants infra already exists** in `lua_domain.c`: `LUA_STAGE=k` (staged,
  RETURN+marker per step), `LUA_CHUNK_LADDER`, `LUA_GCPROBE` (added this session),
  `LUA_SKIP_BASE`, all behind `LUA_DBG_STAGE` csdebugprint markers. `build-lua-domain.sh`
  now takes `DOMAIN_EXTRA_DEFS` (mirrors SQLite).
* Two boots this session produced only the OpenSBI banner / a buildroot-Linux boot stall
  (no HOST output) — an intermittent boot flake, not a domain fault. `ATTEMPTS=3` rides
  over it; a "no marker" run with no `HOST: create_dom` line means the domain never ran.

## Update — the official binary-trees benchmark now runs (N=6, check=4398)

The trivial chunk was milestone 1; the **official CLBG `binary-trees`** now runs to
completion on Capstone: `BT-OK N=6 check=4398` (matches the hand-computed expected
value: stretch 255 + inner 1984+2032 + longlived 127). It is built base-library-only on
purpose — the measured cost is the tree build/traversal + revoke-on-free GC churn, which
is byte-identical to the reference; only the OS-bound reporting differs (`io.write` →
returned integer check, `2^k` → `1<<k`, `math.max` inlined). Mode `LUA_BINTREES` in
`lua_domain.c`, `N` via `-DLUA_BT_N` (default 6, matching the CHERI reproduce script).

### The real blocker was NOT arena bytes — it was revocation NODES

First attempt asserted **inside QEMU**: `cap_rev_tree.c:20 _cap_rev_tree_dup_node_before:
Assertion 'new_node != CAP_REV_NODE_ID_NULL'` — the revocation-node pool exhausted. Root
cause, from reading `cap_rev_tree.c`:
* Every `rof_malloc` does a `cssplit` (+`mrev`), each allocating a **fresh** revocation
  node. rof **never reclaims arena bytes** (one-way SPLIT).
* A node returns to the pool `free_list` only via `cap_rev_tree_release`, which requires
  `refcount == 0 && !valid`. `revoke` sets `valid=false` but does NOT drop `refcount`.
* `refcount` only drops when the capability copies in MEMORY are overwritten. Because rof
  never re-uses freed arena bytes, a collected tree's child-pointer capabilities sit in
  un-reused heap forever → `refcount` stays > 0 → the node is **never released**.
* Net: node consumption is **O(total allocations)**, not O(live set). binary-trees N=6
  makes ~thousands of allocations and exhausted the (default) **10000**-node pool.

This is a genuine architectural datum for the CHERI-vs-Capstone comparison: CHERI's
quarantine+sweep revocation has bounded metadata (amortised over a sweep), whereas
Capstone's per-allocation revocation-node model has metadata that grows with total
allocations and, composed with a non-reclaiming allocator + tracing GC, is not released.

### Fix to get a completing run: match QEMU's pool to the SILICON

`capstone-qemu` `CAP_REV_TREE_SIZE` was **10000**; the deployed silicon bitstream is
`caplifive_65536_nodes.bit` (**65536** nodes). QEMU was under-provisioned vs the silicon
it models. Bumped to **65536** (`target/riscv/cap_rev_tree.h`, rebuilt) — this changes
NOTHING about the guest instruction count (pure QEMU metadata) and makes QEMU predict the
deployed silicon. binary-trees N=6 then completes. A workload needing > 65536 nodes would
also exceed the silicon pool — so 65536 is the honest ceiling, not an arbitrary bump.

**Uncommitted working-tree changes this created** (commit on a branch when asked; run
`precommit-scan.sh`): `capstone/capstone-qemu/target/riscv/cap_rev_tree.h` (pool 65536),
`xlang/lua-cdp/capstone-lua/build-lua-domain.sh` (the two codegen flags + DOMAIN_EXTRA_DEFS),
`xlang/lua-cdp/capstone-lua/lua_domain.c` (LUA_BINTREES + LUA_GCPROBE modes, rd_icount),
`xlang/lua-cdp/capstone-lua/measure-bintrees-cost.sh` (new).

## Measured — temporal-safety overhead on binary-trees (N=6, `-icount shift=0`)

`measure-bintrees-cost.sh`, two byte-identical domains differing only in whether `free()`
revokes; only the benchmark `pcall` is bracketed (newstate/base/load excluded):

| mode | icount | check |
|---|---:|---|
| norevoke (per-object revocable caps, no revoke) | 464,313,752 | 4398 |
| revoke (full temporal safety) | 464,393,352 | 4398 |
| **revoke − norevoke (the revoke cost)** | **+79,600 instr (1.0002×)** | ✓ equal |

Both checks equal 4398 → the revoke does not corrupt the result. The free-time revoke over
the WHOLE benchmark costs **+79,600 instr = 0.017%**. This cross-validates the BST probe's
**+10 instr/op O(1)** revoke cost: +79,600 ÷ ~10 ≈ ~7,960 revokes, consistent with
binary-trees N=6's allocation count. The O(1) per-free cost scales linearly to a real
benchmark.

**The apples-to-apples comparison.** Capstone `norevoke` (per-object revocable caps, no
temporal) is the analogue of CHERI `spatial` (per-object bounds, no temporal); Capstone
`revoke` is the analogue of CHERI `temporal`. So `revoke − norevoke` (Capstone) vs
`temporal − spatial` (CHERI) both measure *the cost of adding the temporal mechanism on top
of per-object spatial safety*:
* **Capstone revoke/norevoke = 1.0002× (+0.017%)** — an O(1) tag-tree splice per free.
* **CHERI temporal/spatial ≈ 1.26× (this session's baseline, N=6)** — the async quarantine
  sweep, O(reachable heap).

Honest caveats: functional-model proxy (deterministic `-icount`, not silicon timing);
different ISAs (the *ratio* is the comparable quantity, not absolute counts); and Capstone
buys the cheap revoke at the cost of revocation-node METADATA that grows with total
allocations (the pool finding above) — a space cost CHERI's quarantine does not have. So
the honest headline is a TIME-for-SPACE trade, not a free lunch.

## Measured — memory (N=6), and the tree depth

Tree depth: `maxdepth = max(mindepth+2, N) = 6` at N=6; the **stretch tree** is built at
`maxdepth+1 = 7`, so **max depth = 7** (2⁸−1 = **255 nodes** in the deepest tree). longlived
is depth 6; the inner loop builds depths 4 and 6.

Instrumented run (rof getters in `revoke_arena_domain.c`; QEMU `helper_csdebugcountprint`
now also prints `REV-NODES alloced_n`):

| dimension | value | meaning |
|---|---:|---|
| peak live objects | **1,030** | working set (high-water of concurrent allocations) |
| live bytes at end | 92,064 (~90 KB) | longlived tree + interpreter state |
| total carved bytes | 732,256 (~715 KB) | rof footprint — never reclaimed, so = sum of all allocations; fit the 4 MB arena, which is why **arena bytes never exhausted** |
| **revocation nodes (`alloced_n`)** | **17,849** / 65,536 pool | temporal-safety metadata high-water |

**The node metadata is the binding cost, and it LEAKS.** 17,849 ≈ **2.2 nodes per
allocation** (cssplit + mrev per malloc), and ≈ 2.2× *total* allocations — NOT ~2× the 1,030
working set. If freed nodes were reclaimed, ~2,000 would suffice; the 8× blow-up is the
rof×GC composition proven above (freed arena bytes keep stale child-capabilities →
`refcount` stays > 0 → node never released). So the temporal-safety metadata (~17.8k nodes)
is ~17× the live working set (~1k objects). That is the concrete "space" in the
time-for-space trade — **but it is a prototype-allocator artifact**: a reclaiming allocator
would bound nodes to ~the working set. On the same axis CHERI pays ZERO extra per-object
metadata (bounds live inside the capability) and a bounded, sweep-released quarantine.

(Note: the `icount` printed by a run WITHOUT `-icount` is `rdcycle` = cycles, not the
deterministic instruction count. The valid instruction counts are the 464M/+79,600 from the
`-icount` measurement above; ignore the ~9.4B from the memory run.)

## CHERI side — perf AND memory (binary-trees N=6, one CHERI-QEMU boot)

`runbench.c` now also captures the benchmark child's peak RSS (`getrusage(RUSAGE_CHILDREN)`,
KB on CheriBSD); `parse-bench.py` reports both axes. Calibrated (workload = RUN − CAL):

| CHERI config | workload instr | time vs spatial | RUN peak RSS | RSS vs spatial |
|---|---:|---:|---:|---:|
| spatial (revocation off) | 841 M | 1.00× | 3,375 KB | 1.00× |
| **temporal** (async quarantine — deployed default) | 1.04 B | **1.24×** | 4,121 KB | **1.22× (+747 KB)** |
| **eager** (revoke on every free) | 328 B | **389.7×** | 3,488 KB | 1.03× (+113 KB) |

Memory reading: async temporal holds freed objects in a **quarantine → +747 KB RSS**; eager
revokes immediately so nothing accumulates → **+113 KB** (but 390× time). RSS deltas are
stable across the 3 reps (temporal 4112/4120/4132, spatial 3372/3372/3380).

## The combined comparison — Capstone vs CHERI, same benchmark, both axes

**The apples-to-apples security match: Capstone `revoke` = CHERI `eager`** (both revoke on
EVERY free → every stale access caught synchronously; the security RESULTS.md shows
Capstone 13/13 and CHERI-async 0/13 of the CDP UAFs caught at the contract point).

| | Capstone revoke | CHERI eager | CHERI async (deployed) |
|---|---:|---:|---:|
| CDP UAFs caught at access | 13/13 | 13/13 | **0/13** |
| time overhead (bt N=6) | **1.0002×** (+79.6 K instr) | **389.7×** | 1.24× |
| temporal memory overhead | ~17,849 rev-nodes ≈ **0.35 MB**¹ | +113 KB RSS | +747 KB RSS |

¹ CapRevNode ≈ 20 B × 17,849. This is the rof×GC-LEAKED figure (≈2× total allocations); a
reclaiming allocator would bound it to ~2× the working set ≈ ~2,000 nodes ≈ **~40 KB**,
comparable to CHERI eager's +113 KB.

**The story the numbers tell.** Both Capstone-revoke and CHERI-eager give full synchronous
temporal safety. CHERI pays for it in **TIME** (390× — a quarantine sweep per free, so it is
undeployable, which is why CHERI actually ships the *async* mode that catches 0/13 at the
access). Capstone pays for it in **SPACE** (per-object revocation-node metadata) but the
revoke itself is an O(1) tag-tree splice, so time is ~free. So the honest headline is a
**time-for-space trade that lands on the right side for this security property**: Capstone
reaches eager-strength temporal safety at ~1× time, where CHERI's eager-strength costs 390×.

Honest caveats: functional-model proxy (deterministic `-icount` / rdinstret, not silicon
timing); different ISAs (compare *ratios*, not absolute counts); the two memory figures live
in different places (Capstone's is hardware/model revocation-tree metadata, CHERI's is
process-heap quarantine RSS) — both are "temporal-safety memory," but not the same bytes;
and Capstone's node figure is inflated by the prototype allocator's non-reclamation.

## Possible refinements (not yet run)

* **bump baseline** — add a broad-NONLIN-heap bump allocator mode to the lua domain to get
  the alloc-side cost (`norevoke − bump`) and the total temporal cost (`revoke − bump`), the
  full three-way breakdown the BST probe has.
* **exact revoke count** — emit a `rof` free counter to turn "+79,600 total" into a precise
  "+X instr/revoke over Y revokes" (expected ≈ +10).
* **larger N** — N>6 raises maxdepth and the allocation count; watch the 65536 node pool.
