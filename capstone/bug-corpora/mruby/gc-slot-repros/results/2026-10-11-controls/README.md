# mruby GC object slots: parent, sibling and successor controls, 2026-10-11

This is the mruby row of the pre-registration
[`docs/plans/2026-10-11-nested-lifetime-controls.md`](../../../../../docs/plans/2026-10-11-nested-lifetime-controls.md)
(`53325bba0d5c`, pushed before any harness was built). The nested allocator is mruby 4.0.0-rc2's
GC object slots: `mrb_obj_alloc` hands out a slot and the sweep releases it. The protected arm
uses patch 0008 (`MRB_CAPSTONE_SUBLET`), the configuration in the security table. The release
arm is the same source without the define. The driver, its build, its runner and the judge are
in [`../../controls/`](../../controls/README.md). `results.json` holds every repetition: the
lines the driver printed, its probes, the addresses and the fault records.

**All eight predictions hold, in 3 of 3 repetitions per arm.** The positive control fired as
registered:

- the stale read through A completes on release and returns C's data;
- on protected it faults with cause 25, at the probe's load.

## What ran

**Source.** A `git archive` of `53325bba0d5c` (`capstone/`, `compiler-rt/lib/builtins`) was
committed in a scratch repository on p13. The uncommitted `controls/controls.c`, `build.sh` and
`run.py` were copied into it; their sha256 are in `results.json` `inputs.llvm_capstone_revision`.
`collect.py` and the two READMEs were written after the runs. The branch has since moved to
`74a7b08f4e9a`. Its two new commits touch only `bug-corpora/sqlite` and
`bug-corpora/wireshark`.

**Builds.** One virtual application SDK was built (musl mallocng, arena 64 MiB) and copied per
arm. `ports/mruby/app/build-mruby-domain.sh` then ran with `MRBD_PIN=4.0.0-rc2` (mruby
`9d523e2f74f2`) for both arms in parallel: 110 s each, 116 s with the SDK.

| arm | `MRBD_SUBLET` | mruby image | `libmruby.a` | driver `controls.dom` | CDERIVE / CREVOKE words in `.text` |
|---|---|---|---|---|---|
| release | 0 | `7b5872bc7141cbad` | `53158de081b2ca21` | `f0d9697ac29000bf` | 0 / 0 |
| protected | 1 (patch 0008) | `321dc9b98691cc0a` | `57ecc53f2c552459` | `da44c7cf1ece407e` | 8 / 2 |

Both mruby images are byte-identical to the security-table and same-answers images
(`release-differential/results/2026-10-11-virtual`). The driver adds no lifetime instruction of
its own: its counts are libmruby's. Driver flags are those of `probe/build-capi.sh`, plus `-g`,
plus `-DMRB_CAPSTONE_SUBLET -I capstone/runtime/include` on the protected arm. Compiler:
`clang-22` `4c6135fe...`. Full hashes are in `results.json` `inputs.builds`.

**Platform.** These are the registered hashes, recorded per boot in `results.json` `boots`:

- QEMU `af37cc32...`;
- launcher `capstone-vexec` `3975314a...`;
- module `capstone_vm.ko` `a1c6cb6b...`;
- kernel `e833dfb1...`;
- firmware `44cd6cbc...`;
- rootfs `62049800...`.

Configuration: virtual profile, one hart, exact bounds, 2 GiB. Each boot is its own ephemeral VM
through `runtime/virtual/run-staged.py`. In the guest, `CAPSTONE_EXEC_DIAGNOSTICS=1` lets the
launcher print its domain fault line. Each mode is its own `capstone-vexec ./controls.dom MODE`
process, in the order parent, sibling, successor, stale. Six boots ran in parallel on p13, three
per arm. Boot wall times: release 20.1-20.3 s, protected 20.6-21.1 s; each `run.py` took 26-27 s.

## Results

Each cell held in all three repetitions. Each repetition recorded `CTL ... start ... pages=1
live=720 generational=1` and `FREED A after 1 full gc`, and each transcript arrived whole.

| control | arm | outcome | what was read |
|---|---|---|---|
| parent | protected | completed | A's slot through the page: `tt` = `MRB_TT_FREE`, `issued[i]` = NULL, `next` = the freelist link, slot = the page's freelist head; wrote `0x5a5a1234c0ffee11` at slot+64 and read it back. The same through `page->life` (`tt` 4 = `MRB_TT_FREE`, write read back). |
| parent | release | completed | the same through the page; there is no `page->life` |
| sibling | protected | completed | B intact (`tt` 18, payload `B-SIBLING-PAYLOAD-0123456789abcd`); written and read back |
| sibling | release | completed | the same |
| successor | protected | completed | C = A after 1 allocation; C read (`C-SUCCESSOR-PAYLOAD-...`), written and read back |
| successor | release | completed | the same |
| stale | protected | **fault, cause 25** | in `ctl_probe_stale_read_A`+164, `controls.c:180` (`lbu` of A's first payload byte; fault address = A + 0x14) |
| stale | release | completed | the read through A returned `C-SUCCESSOR-PAYLOAD-0123456789ab`, C's data |

**Probe locations** are all in `capstone/bug-corpora/mruby/gc-slot-repros/controls/controls.c`:

| probe | line |
|---|---|
| `page-read-tt` | 158 |
| `page-read-next` | 160 |
| `page-write-tail` | 162 |
| `page-read-tail` | 164 |
| `page-read-issued` (protected only) | 167 |
| `life-read-tt` | 169 |
| `life-write-tail` | 171 |
| `life-read-tail` | 173 |
| `obj-read` (B, C) | 176 |
| `obj-write` (B, C) | 178 |
| `stale-read-A` | 180 |

**Addresses, repetition 1.** These are the slot addresses: each process places the domain
elsewhere, so they differ between processes.

Within every run of every repetition:

- C equals A;
- B sits four slots above A (indexes 955 and 959 of 1,024);
- A and B are on the page that holds `Object`.

| control | arm | A | B | C | page | A's capability base |
|---|---|---|---|---|---|---|
| parent | protected | `0x3f9281dc60` | -- | -- | `0x3f92807060` | `0x3f9281dc60` (the slot) |
| parent | release | `0x3fba16dc50` | -- | -- | `0x3fba15b060` | `0x3fba15b060` (the page) |
| sibling | protected | `0x3f8a47db20` | `0x3f8a47dc60` | -- | `0x3f8a467060` | `0x3f8a47db20` |
| sibling | release | `0x3fa166db10` | `0x3fa166dc50` | -- | `0x3fa165b060` | `0x3fa165b060` |
| successor | protected | `0x3fa8f6dc60` | -- | `0x3fa8f6dc60` | `0x3fa8f57060` | `0x3fa8f6dc60` |
| successor | release | `0x3f84d45c50` | -- | `0x3f84d45c50` | `0x3f84d33060` | `0x3f84d33060` |
| stale | protected | `0x3fbdd45c60` | -- | `0x3fbdd45c60` | `0x3fbdd2f060` | `0x3fbdd45c60` |
| stale | release | `0x3f99015c50` | -- | `0x3f99015c50` | `0x3f99003060` | `0x3f99003060` |

The protected stale faults, with the code range from each launcher line:

| repetition | pc | fault address | code range |
|---|---|---|---|
| 1 | `0x3fbd412bfc` | `0x3fbdd45c74` | `0x3fbd400000-` |
| 2 | `0x3fa3812bfc` | `0x3fa3f45c74` | `0x3fa3800000-` |
| 3 | `0x3fac412bfc` | `0x3facafdc74` | `0x3fac400000-` |

Each pc is image VA `0x22bfc`, which the disassembly shows is the `lbu a3, 0(a3)` loading through
A + 0x14 in `ctl_probe_stale_read_A`. Under 0008, A's capability is bounded to its slot (its
base is the slot). On release it is the GC's own pointer, derived from the page (its base is the
page).

## Predictions

| arm / control | prediction | verdict (n = 3) |
|---|---|---|
| protected / parent | completes; the read returns what the allocator left there | holds |
| protected / sibling | completes, B intact | holds |
| protected / successor | C's address equals A's; access through C completes | holds |
| protected / stale | rejected with cause 25 at the labelled probe | holds |
| release / parent | completes | holds |
| release / sibling | completes, B intact | holds |
| release / successor | C = A, completes | holds |
| release / stale | completes and reads C's data | holds |

The judge (`collect.py`) was checked against inputs that must fail it:

- the pilot-2 boots, whose transcripts lost lines, give "no reading" on every mode;
- the final boots with their arm labels swapped give "does not hold" on protected/parent,
  protected/stale and release/stale.

## Deviations and notes

- **The corpus had no scaffolding.** `gc-slot-repros` held only a planned `corpus.json`, with no
  README, harness, runner or arms. The driver follows `release-differential`'s C-API cases
  (`probe/build-capi.sh`'s flags), and the boot flow follows the memory-behaviour runs
  (`runtime/virtual/run-staged.py`, one ephemeral VM per boot).
- **Two pilot boots per arm were void, and are not counted.**
  - Pilot 1 (`_IONBF` stdio into a guest file) lost and truncated output lines. The launcher
    printed no fault line, because `CAPSTONE_EXEC_DIAGNOSTICS` was unset.
  - Pilot 2 (line-buffered stdio, `O_APPEND`, diagnostics on) still lost whole lines. The new
    line-count check caught every loss.
  - The driver now hands each line to one `write(2)`. A third pilot and all six final boots
    arrived whole. The accesses did not change across the pilots; only the output path and,
    with it, the probe line numbers did.
  - Where lines did arrive, pilots 2 and 3 agree with the final runs: protected stale was cause
    25 in `ctl_probe_stale_read_A`, and everything else completed. Their records stay on p13
    under `~/ctl-mruby/runs/pilot*`.
  - Not root-caused: why stdio output written through the delegated I/O path loses whole lines.
    The 2026-10-11 corpus README reports the same loss on the console.
- **N = 3 per arm.** The pre-registration does not fix N.
- **Beyond the registration.**
  - On the protected arm, control 1 also reads and writes the slot through `page->life`, the
    CDERIVE parent that the slot's child came from.
  - Every control first confirms that A was freed, by reading through the page.
- **Allocator detail.** B and C stay reachable through the GC arena. The GC is disabled only
  during the reuse search, so that the search's own allocations cannot run a collection.
- **p13's clock** runs about 26 minutes behind the build host. No result depends on a
  timestamp; the wall times are durations.
