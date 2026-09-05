# Q-03 fix design: a HOLE with a `region_live[]` sentinel — never renumber, never reuse

*Proposal, 2026-09-05. Audited by `claim-auditor` the same day: mechanism SAFE WITH CHANGES (eight,
all folded in below and marked **[audit]**), the pool-full item and the acceptance criteria as first
written NOT SOUND — rewritten. One omission would have stopped the monitor booting (item 0).*

## The constraint that decides the shape

Region ids are **guest-visible array indices** into the monitor's `regions[]`
(`create_region` returns `region_n - 1`, stand-in `sbi_capstone.c:546`). The kernel module
assumes it in writing (`modcapstone/module/capstone.c:33`), caches `base_paddr` keyed by that
index (`:222-231`, `:288-308`), and `mmap` resolves physical pages through that cache
(`:391 remap_pfn_range`). It has no way to learn that a slot moved or was reused: no
`REGION_POP` ioctl, and `probe_regions` only ever *extends* its table (except a full
`region_n = new_region_n` resync at `:307`).

So any fix for the exact-fit case must keep **index == id for every slot the module may hold**.
That rules out the two obvious shapes, both already refuted (Q-03 entry, audit 2026-09-05):

| shape | why not |
|---|---|
| compact (move the last slot into `i`) | retargets a live guest id; next `create_region` reuses the freed id, the module takes the "Region ID reuse detected" branch (`:215`) without recording geometry, and userspace maps the wrong pages. Silent. |
| shrink only at the tail (the landed port; the firmware's 2026-08-01 fix) | leaves the middle-slot case as a spin (`0x1235`, 3 of 24 items in manifest B). Its module hazard is **latent** (second audit 2026-09-05: M ≤ N is invariant because the module's own region is always the tail when it syncs, and a fresh `__get_free_pages` image cannot exact-fit it) — but it is **silent** (no print), so nothing measures how often it fires, and it does not fix the bug. |

## Design

**A consumed slot becomes a HOLE: it keeps its index forever, and is skipped everywhere.**

0. **[audit, fatal if omitted]** The three genesis regions are CPMP-resident and are never
   appended: `cap_env_init` sets `region_n = 3` and the CPMP maps by hand
   (`sbi_capstone_dom.c:12-21`). They must be marked live **explicitly**, mirroring the file's
   own `-1` loops at `:22-27`: `region_live[0..2] = 1; for(i = 3; ...) region_live[i] = 0;`.
   Without it the first `split_out_cap` at boot skips all three, reaches `:251-252`
   `capstone_error(CAPSTONE_NO_REGION)` and the firmware spins before Linux starts. The nested
   monitor (`sbi.dom.c:46`, `region_n = 1`) needs the same. Never rely on zero-init here.
1. `unsigned region_live[CAPSTONE_MAX_REGION_N];` alongside `regions[]` — **`unsigned`, not
   `unsigned char` [audit]**: the file has no sub-word type anywhere (`grep char|short` = 0) and
   Capstone-C carves one exactly-sized capability per global (`region_cpmp[64]` → 512 B), so do not
   introduce the first one in a file with this parse history. Set to 1 at every append
   (`:292-293`, `:309-310`, `:543-544`, `:720-725`, `:1084-1085`).
2. In `split_out_cap`, the exact-fit branch (`:262-272`) becomes, for **every** `i` including the
   tail: clear the CPMP mapping exactly as `drop_exact_fit_tail` does now, then
   `region_live[i] = 0; regions[i] = 0;`, **`region_n` untouched**. The tail-only shrink is removed
   with it. `C_PRINT(0x1236); C_PRINT(i); C_PRINT(region_n);` marks every hole — the first
   instrument that will ever say whether the tail case fires. **[audit]** If `!linear` at an exact
   fit, the fall-through at `:308-311` would re-append the same capability at a new index — the
   first index MOVE in the file. Unreachable today (`linear == 0` only at boot, `sbi_capstone_dom.c
   :33,35,44`, none an exact fit); make it loud rather than implicit: `C_PRINT(0x1238)` + spin.
3. Every consumer that takes an id from the guest rejects a hole the same way it rejects an
   out-of-range id (`return -1`): `:550`, `:660`, `:754`, `:773`, `:812`, `:851`, `:1016` —
   `if(region_id >= region_n || !region_live[region_id])`.
4. Every search over the pool skips holes: `:236` (`split_out_cap`'s containing-region search)
   and `:1008` (`swap_cpmp`'s find-by-address; a hole has `region_cpmp == -1`, so it does NOT hit
   the `continue` at `:1009` and would fall into `cap_base(regions[hole])`). **[audit]** `:182` is
   `print_regions`, dead code with no caller, not a CPMP scan — skip optional. The eviction write
   at `:1035-1036` needs nothing: it is indexed through `cpmp_region[]`, reached only when all 16
   CPMPs are occupied, and a hole's CPMP mapping is cleared, so no CPMP ever names a hole.
5. `pop_region` (`:791-806`) clears `region_live` for every popped slot. **[audit]** It does hand
   indices back to the append path, so "holes are never reused" holds only because `pop_region`
   is unreachable from the module (no ioctl; the nested payloads that issue `REGION_POP` hit the
   nested table). State it as that, not as an enforced invariant.
6. `REGION_COUNT` (`:968`) still returns `region_n` **including holes**; `query_region` on a hole
   returns `-1` as for an out-of-range id (`:851-853`). Module side verified: `sbi_res.value` is
   `long`, `len` becomes `ULONG_MAX`, which fails `len < MAP_SIZE_LIMIT` (`:25`, `:384-386`) so
   `mmap` can never select it and `pre_mmap_offset` is not advanced; `libcapstone.c:557-574`
   `map_region` returns NULL for it.
7. **Holes are never reused by the exact-fit path or by any append** (see 5 for the one caveat).
   Reuse is the compaction failure by another route.
8. **[audit] `REV_TRANSFERRED` slots become holes too** (`:608-620`): the linear capability is
   handed to the domain and the CPMP mapping cleared, but `regions[region_id]` is left holding a
   stale duplicate (`:610` says so: "should be added to a free list"). That slot is already dead,
   still reads live, and its crash path is documented in
   `intra_domain_mrev_revoke_probe_guest.c:22-28` (stale duplicate → untagged after the domain
   revokes → `cap_base` in `swap_cpmp`). Same two lines as item 2, distinct print `0x1239`. The
   five REV_TRANSFERRED probes in the nightly list are its regression check.
9. **Pool full — decided [audit]:** the stand-in has **no** `region_n >= CAPSTONE_MAX_REGION_N`
   check (the board copy has four). Today an append at slot 64 takes a capability bounds fault
   inside M-mode with an unresolved re-entry — not a clean overrun. "Return an error, never spin"
   is not implementable at `:292`/`:309` (inside `split_out_cap`, which returns a capability; a
   null propagates into `__split` and aborts QEMU) nor at `:1084` (`dpi_share_region` is void, no
   error channel). Decision: every append site prints `C_PRINT(0x1237); C_PRINT(region_n);`
   first; the three with an error channel (`:543`, `:720`, `:724`) then `return -1`; the three
   without (`:292`, `:309`, `:1084`) spin — a deliberate, printed wedge that the batch runner
   reboots from, strictly better than the silent fault it replaces. Note the module cannot see a
   `-1` as an error (M-3): `create_region` returning `-1` leaks the pages and prints
   `Failed to fetch information about the newly created region` — that dmesg line is the
   module-side observable. Budget headroom is **zero**: monitor pool 64, module array 64 (M-2).

The board firmware (`caplifive-system-dev`, `CAPSTONE_SPLIT_EXACT_FIT`) has the same site and
the same module; the same patch applies. That is a separate decision for the project lead —
it needs a firmware rebuild and, for the FPGA, a board session to validate.

## What the build must answer — predicted BEFORE it is built (rewritten after audit)

The first version predicted "exactly three `0x1236` prints at items 8, 12, 22". **Wrong, and it
would have failed a correct build:** `run-domain-batch.py:158-166` reboots QEMU after every
WEDGE, so items 9-24 of the wedging run executed in three fresh boots, which is why all three
prints read `i = 8, region_n = 10`. Once item 8 stops wedging, items 9-24 run in ONE boot with a
carve history that has never existed; positions 12 and 22 were reboot artefacts, not predictions.
Item 8's print is deterministic (identical boot prefix), but its `i`/`region_n` may read `8+k` /
`10+k`: `k` is the count of silent tail drops in items 1-7, which nothing has ever measured —
**record it as a measurement, not a failure.**

Manifest B replayed verbatim (`/tmp/capstone/sweep/b2/q03b/B.tsv`), **twice**:
1. 24/24 `RET`; zero `0x1234`; zero `0x1235`.
2. **≥ 1** `0x1236`, the **first at item 8**. Later prints: record count and positions.
3. `grep 'Print = Scalar(0xdeadbeef)'` on the batch log = **0**. Non-zero means
   `CAPSTONE_NO_REGION` (`0x1` follows — holes make "no containing region" more reachable, and
   that path is a spin, `:251-252`) or a monitor-internal capability fault (a missed consumer
   reaching `cap_base` raises `UNEXP_OP_TYPE` into the monitor's own trap vector).
4. `grep 'lcc on an UNTAGGED operand'` on QEMU stderr = **0** — the signature of a missed
   consumer. A missed consumer reaching `__split`/`__delin`/`__mrev` aborts QEMU (`assert`, live
   in the debug build) and the runner records `FAULT` — also caught.
5. **Reject `RET 18446744073709551615` and `RET 262150125`** in the results: a pool-full
   `create_domain` returning `-1` is invisible to the module (M-3), `capstone-test.user` prints
   `retval = %lu`, and the runner's `(\d+)` match writes `RET`; `0x0FA017ED` is
   `CAPSTONE_DOMAIN_FAULT_RETVAL` (`sbi_capstone.h:54`) and reads as `RET` the same way. Neither
   is a pass.
6. **Record `region_n` at the end of the boot** (REGION_COUNT from a final probe item): manifest B
   without reboots is the first run in which the pool is never reset — open item 3 goes live here.
7. Zero `0x1237` (pool full) and zero `0x1238` (the unreachable `!linear` exact fit).

Manifest F (the 6/6 determinism replay) unchanged.

**Module-desync directed test** (a small host program under `capstone/tests/runtime-qemu/` — NOT
`capstone-test.user`, whose `create_region` is commented out, `userspace/capstone-test.c:34`, which
is why manifest B/F could not exercise the module at all; the SQLite hosts do create regions,
`sqlite_host_row3_b2.c:41,54-56`): `REGION_CREATE` ×2, a `DOM_CREATE` that exact-fits (the fill
images that produced item 8), then `REGION_CREATE` and `REGION_QUERY` of the returned id: base/len
must equal what the module allocated, and no `Region ID reuse detected` in dmesg — **after open
item 0 has shown that line can fire at all.** Negative control on the hole id itself: `REGION_QUERY`
must return **`len == ULONG_MAX`** if the module has probed it, `len == 0` only if never probed
(**[audit]** "0 / -1" as first written passes either way and checks nothing); and the share
rejection must be observed through a direct `ioctl(IOCTL_REGION_SHARE)` reading `args.retval` or
from the domain side — `libcapstone.c:540-547` `share_region` is `void` and discards the monitor's
`-1`, so through the library a rejected share is indistinguishable from a successful one.

The QEMU suites the nightly runs (RV8, BEEBS, SQLite memory arm, the five REV_TRANSFERRED probes)
unchanged — the rebuilt monitor is the one every suite boots.

## Open items (must close before code)

0. **Positive-control the module's only desync detector first.** `pr_alert("Region ID reuse
   detected")` (`module/capstone.c:216`) has never been shown to fire; in a scratch build force
   `m_args.region_id` low once and confirm the line reaches the console. Without it, every clean
   dmesg in the tests above is a zero from an unproven instrument.

1. **CLOSED by audit:** no existing path moves a live capability between indices — every append
   is at `regions[region_n]`, `:1036` writes back to the slot's own index, and `:572/:592/:604/
   :716/:785/:823/:880` rewrite in place; `share_child_region`'s `:716` is a type change over the
   same address range, so the module's cached geometry stays right. The design's own `!linear`
   exact-fit fall-through (item 2) would have been the first move; it is made loud instead.
2. **CLOSED (source read 2026-09-05):** a hole answers `query_region` with `-1` exactly like an
   out-of-range id (`:851-853`), so `probe_regions` records it with `len = (unsigned)-1`
   (`:298-303`); that fails `len < MAP_SIZE_LIMIT` (`0x10000000`, module `:25`), so it advances
   no mmap offset and `map_region`'s match loop (`:384-386`) can never select it — an `mmap`
   over a hole returns `-EINVAL`, never page 0.
3. **CLOSED for the nightly, OPEN for batches — and coupled to M-2:** the module's own array is
   64 (`MAX_REGION_N`, `:30`) while the board pool is 96 (`caplifive-system` `sbi_capstone.h:78`)
   with no bound in `probe_regions` (`:292-306`); a board fix of this design must not push the
   monitor count past 64 without fixing M-2 first. the nightly suites boot **one domain per
   boot** (`run-all-beebs.sh:26,153,196` loops over per-benchmark runners; `run-all-rv8.sh:21`
   likewise), so the pool never grows past a handful there. Only `fuzz/run-domain-batch.py`
   ("ONE QEMU boot", its docstring) and the SQLite arms run many domains per boot; manifest B
   (24 items) ends near 30 of 64. A batch above ~50 items would hit the limit, which is why
   item 8's check must **return** and the batch runner's existing reboot-on-fault path must
   treat it as a per-item failure, not a batch error.
5. `device_ioctl` takes no lock (auditor, UNRESOLVED): concurrent `REGION_CREATE`s could desync
   the module by a route unrelated to holes; inert under `-smp 1` QEMU, unestablished on the board.
4. Capstone-C parse of the new array writes: bisect in `/tmp` with the unpatched source as the
   control if anything fails to parse — **not** from a remembered rule (the one recorded
   earlier today was wrong).

## Rejected on the way here

- **Compaction** — refuted, above.
- **Reclaiming holes on `create_region`** — same failure as compaction once the module has
  probed the hole as a padding entry.
- **A separate id → index table (stable ids)** — **[audit]** my first reason ("changes the
  module's contract") was wrong: the module's contract is the four observable behaviours
  (`create_region` returns an id, `REGION_COUNT` bounds the id space, `probe` enumerates,
  `query` answers per id), and a dense-id table serves all four; `REGION_COUNT` becomes a
  high-water mark and dead ids answer `-1` under **both** designs. Edit counts are ~19 (hole) vs
  ~17 (table), and the table removes the three skip sites whose omission is silent. The honest
  grounds for still choosing the hole on the stand-in: the table needs a compaction step that
  **moves a capability**, which is exactly the platform-dependent operation the first audit
  refuted (spec LDC/STC nulling on silicon vs QEMU's tagged duplicate). The hole's real cost is
  a permanently leaked slot per exact fit against a pool with zero module headroom (M-2) —
  measured for the first time by criterion 6 above. Revisit if that number is not small.
