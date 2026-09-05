# Q-03 fix design: a HOLE with a `region_live[]` sentinel — never renumber, never reuse

*Proposal for review, 2026-09-05. Not implemented. Goes through a second `claim-auditor` pass
before code, per the Q-03 entry in `ref/ISSUES.md`.*

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

1. `unsigned char region_live[CAPSTONE_MAX_REGION_N];` alongside `regions[]`. Set to 1 on
   every append (`:292-293`, `:309-310`, `:543-544`, `:720-725`, `:1084-1085`); the arrays are
   zero-initialised so the untouched tail reads dead.
2. In `split_out_cap`, the exact-fit branch (`:262-272`) becomes, for **every** `i` including
   the tail: clear the CPMP mapping exactly as `drop_exact_fit_tail` does now, then
   `region_live[i] = 0; regions[i] = 0;` and **do not touch `region_n`**. The tail-only shrink
   is removed with it: one rule, no special case, and the latent tail hazard goes with it.
   `C_PRINT(0x1236); C_PRINT(i); C_PRINT(region_n);` marks every hole creation — an
   instrument that cannot fire silently, and the positive control for the change.
3. Every consumer that takes an id from the guest rejects a hole the same way it rejects an
   out-of-range id (`return -1`): `:550`, `:660`, `:754`, `:773`, `:812`, `:851`, `:1016`
   — `if(region_id >= region_n || !region_live[region_id])`.
4. Every search over the pool skips holes: `:182` (CPMP swap scan), `:236`
   (`split_out_cap`'s containing-region search), `:1008` (find-by-address). A hole holds no
   capability, so `cap_base`/`cap_end` on it (`:1011-1012`) must never execute.
5. `pop_region` (`:791-806`) clears `region_live` for every popped slot (holes it pops are
   already dead; popping a live slot behaves as today). Unreachable from the module, kept
   correct anyway.
6. `REGION_COUNT` (`:968`) still returns `region_n` **including holes**, and `query_region`
   on a hole returns `-1` as it does for an out-of-range id — the module's `probe_regions`
   records a hole as `base = len = (unsigned)-1`... **open item 2 below**: check what the
   module does with that entry (`:295-303`) and whether `mmap` of it is refused.
7. **Holes are never reused.** Reuse is exactly the compaction failure by another route
   (the module may already hold the index, as a padding entry).
8. The stand-in has **no pool-full check** — nothing tests `region_n >= CAPSTONE_MAX_REGION_N`
   (the board firmware does, `caplifive-system` copy ~`:600`). With holes permanent the pool
   (64 slots, `sbi_capstone.h:42`) is consumed faster, so the check goes in with this change
   and must **return an error**, never spin. Budget: manifest B reached `region_n = 10` at
   item 8, ~1 slot per item plus one hole per exact fit — a 24-item boot ends near 30; a
   long nightly boot needs counting before this ships (open item 3).

The board firmware (`caplifive-system-dev`, `CAPSTONE_SPLIT_EXACT_FIT`) has the same site and
the same module; the same patch applies. That is a separate decision for the project lead —
it needs a firmware rebuild and, for the FPGA, a board session to validate.

## What the bitstream/build must answer, written before it is built

- Manifest B replayed verbatim (`/tmp/capstone/sweep/b2/q03b/B.tsv`): **24/24 RET**, zero
  `0x1234`/`0x1235`, and **exactly three `0x1236` prints at items 8, 12, 22** with the same
  `i` values as the wedges had. Three prints prove the change fired where the bug was; 24/24
  with zero prints would mean the exact fit never happened and proves nothing.
- Manifest F (the 6/6 determinism replay) unchanged.
- **Module-desync directed test** (a small host program under `capstone/tests/runtime-qemu/` —
  NOT `capstone-test.user`, whose `create_region` is commented out, `userspace/capstone-test.c:34`,
  which is why manifest B/F could not exercise the module at all; the SQLite hosts do create
  regions, `sqlite_host_row3_b2.c:41,54-56`):
  `REGION_CREATE` ×2, a `DOM_CREATE` whose image exact-fits (drive it with the fill images
  that produced position 8), then `REGION_CREATE` and `REGION_QUERY` of the returned id:
  base/len must equal what the module allocated, and no "Region ID reuse detected" in dmesg.
  Its negative control is the hole id itself: `REGION_QUERY` on it must return len 0 / `-1`,
  and `REGION_SHARE` of it must fail, not share a domain's code pages.
- The QEMU suites the nightly runs (RV8, BEEBS, SQLite memory arm) unchanged — the rebuilt
  monitor is the one every suite boots.

## Open items (must close before code)

0. **Positive-control the module's only desync detector first.** `pr_alert("Region ID reuse
   detected")` (`module/capstone.c:216`) has never been shown to fire; in a scratch build force
   `m_args.region_id` low once and confirm the line reaches the console. Without it, every clean
   dmesg in the tests above is a zero from an unproven instrument.

1. **Index stability under the share paths.** `share_child_region` (`:658-725`) appends
   `head`/`tail` and rewrites `regions[parent_id]`; `shared_region_annotated` and `REVOKE`
   rewrite `regions[region_id]` in place (`:572`, `:592`, `:604`, `:716`, `:785`). In-place
   rewrites keep the index; if any path *moves* a slot, the module is already desynced today,
   independent of this design. The 2026-09-05 audit (tail-drop reachability) was asked this.
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
- **A separate id → index table (stable ids)** — correct and cleaner, but it changes the
  module's contract (`:33` TODO) and both sides at once; the hole preserves the contract with
  ~14 monitor-side edits and no module change. Revisit if the pool budget (open item 3) fails.
