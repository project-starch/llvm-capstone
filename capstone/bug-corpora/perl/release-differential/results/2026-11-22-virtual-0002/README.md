# Landing sublet patch 0002: the corpus on stock, 0001 and 0001+0002 (result, read 2026-11-23)

Pre-registration: [`PREDICTIONS.md`](PREDICTIONS.md), lane commit `affe31cecf98`, written 2026-11-22 19:06 before the
runs. **All four predictions hold, so 0002 is landed:** `ports/perl/sublet/pending/5.36.3/0002` is now
`ports/perl/sublet/patches/5.36.3/0002` (sha256 `04540913…`, unchanged).

What 0002 changes: `S_sv_arena_of` finds a head's arena in an address-sorted array by bisection and reads no arena;
0001 walks `PL_sv_arenaroot`, one load through each arena. It changes no CDERIVE or CREVOKE site and no lifetime. On
silicon (bitstream P1, `suns.pl`) the protected interpreter costs +16.05 % in-context cycles with it and +208 % without
(`docs/ref/nested-allocator-workloads.md`, 2026-11-18).

## Runs

- Images as pre-registered, read back from each bundle's `inputs.json`: stock `2ed593006a934fd8…`, p0001
  `f2edeb54c0da05cf…`, p0002 `140c747371415b07…`.
- Platform: kit-devq (QEMU `59f553ef2968dc2a…`, module `af7484e5e39019a6…`), the same in all three bundles. One VM
  per arm, 2026-11-23 07:47-08:24; each runner exited 0.
- **Correction (2026-11-23, after this README was first pushed in 408ffbcd52de): the guest had no exact bounds.**
  PREDICTIONS.md has the VMs brought up with `--exact-bounds`, and every `inputs.json` here records
  `"exact_bounds": true`. But kit-devq's QEMU is kit-profq's wrapper, which strips `x-capstone-exact-bounds=true`
  on purpose: that twin is RTL-pinned, and the silicon has no exact bounds. Its `applied.log` shows the applied
  line at this run's three boots (07:47:02, 08:02:09, 08:12:18) as
  `-cpu rv64,sstc=false,h=false,sv48=false,sv57=false,x-capstone-u-mode=true,svadu=false`. The record copies the
  requested `-cpu`, not the applied one (the runtime lane's kit, which it is fixing). Read the field as false. All
  three arms ran on that same platform, so the landing criterion, p0001 against p0002, is unaffected. Predictions
  3 and 4 compare with 2026-11-11, whose platform record also says true; whether that run's kit applied it is not
  checked here.
- Per arm: `<arm>/` is `run-virtual.py`'s bundle (11 cases), `<arm>-qualify/` is `qualify.py`'s (47 core test files).
  Raw guest output is not committed.

## Readings against the predictions

1. **Holds.** Every runner exited 0, both controls passed on every arm (`eval ok, shim ok`), and all 11 rows exist.
2. **Holds: p0002 equals p0001 in every cell.** That is 11 of 11 cases on outcome, cause, function, in_fault_sites
   and the harness verdict word with its counts. It is also 47 of 47 qualify files on outcome, cause, function and the
   `ok` / `not ok` counts. The comparison script was checked against stock vs p0001, where it reports 5 differing
   cases (01, 02, 05, 06, 09).
3. **Holds.** stock equals 2026-11-11's `virtual-malloc` rows in outcome, cause and function, 11 of 11.
4. **Holds.** p0001 equals 2026-11-11's `virtual-nested-pools` rows in outcome, cause and function, 11 of 11. That
   includes 06's fault 25 in `Perl_regfree_internal`. p0002 equals those rows too.

| arm | fault | completed | exit255 | qualify: completed / fault, ok / not ok |
|---|--:|--:|--:|---|
| stock (`virtual-malloc`) | 6 | 4 | 1 | 25 / 22, 3,285 / 22 |
| p0001 (`virtual-nested-pools`) | 9 | 2 | 0 | 25 / 22, 3,285 / 22 |
| p0002 (`virtual-nested-pools`) | 9 | 2 | 0 | 25 / 22, 3,285 / 22 |

qualify reads the same per file on all three arms, stock included: outcome, cause, function and counts.
