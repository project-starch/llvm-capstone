# Landing sublet patch 0002: the corpus on stock, 0001 and 0001+0002 (pre-registration, written 2026-11-22 19:06, before the runs)

`ports/perl/sublet/pending/5.36.3/0002-sv-arena-lookup-by-index.patch` (04540913…) changes how `S_sv_arena_of`
finds a head's arena: a sorted array searched by bisection instead of a walk of `PL_sv_arenaroot`. It changes no
CDERIVE or CREVOKE site and no lifetime. `pending/README.md` lets it move to `patches/` only after this port's own
suite has run on an image built with it. This is that run.

## Images

All three are built by `ports/common/application/build-virtual.sh perl` from one detached worktree at origin/dev
deb4ebf0b06b, with `CAPSTONE_SDK_CORPUS=1` (each SDK's codegen ends in `-fno-builtin`). The compiler is
`llvm-agp`: "clang version 22.0.0git (… d9fa11a409d2…)", dev 77bf93aef483 plus auipcgp, which is off by default
(the runtime lane's choice). The SDK runtime archives hash the same in all three builds (b0256e730c46…,
25e82f64b12f…).

| arm | selector | patches | perl sha256 | CDERIVE / CREVOKE words |
|---|---|---|---|---|
| stock | `CAPSTONE_VIRTUAL_LIFETIMES=0` | port 0001-0012 | 2ed593006a934fd8… | 7 / 3 (mallocng) |
| p0001 | `=1` | + sublet 0001 | f2edeb54c0da05cf… | 49 / 6 |
| p0002 | `=1` | + sublet 0001, 0002 | 140c747371415b07… | 39 / 4 |

- Between p0001 and p0002, the only source difference is `sv.c`. Encode's `def_t` and the two Compress
  `constants.h` files also differ, but only in the build path they embed.
- p0002 has fewer lifetime words than p0001, at the same sites. Read as inlining moved by the new lookup; that is
  inferred, not read from the disassembly.
- 0002 is copied into `patches/` only in the scratch worktree. It stays in `pending/` in every committed tree
  until this run is read.

## Platform

- Kit `perf/devheap/kit-devq`, the runtime lane's dev kit: its QEMU, Image, fw_jump, rootfs, launcher, job helper
  and module. The ssh server is `dlpre-val/ssh_server`, because the kit's rootfs has no dropbearkey.
- `capstone_vm up --profile virtual --exact-bounds`, with no module parameters and no CPU property. 2026-11-11's
  `x-capstone-test-csrs=true` sets up the test CSRs for timing, and no verdict here reads time.
- One VM per arm, booted in the order stock, p0001, p0002, each waiting for the shared QEMU lock.
- Per arm, `run-virtual.py` runs the 11 cases with the eval and shim controls (600 s per case). Then `qualify.py`
  runs the 47 core test files of 2026-10-11's `qualify-v2.txt` that this release has (op/gc.t is absent), 600 s
  each.
- A stubbed dry run of the chain (every VM and runner call replaced by echo) reached its last rung. The lock-retry
  branch and the failure branch of its `up` test were each exercised on a recorded message.

## Predictions

1. Each arm's runner exits 0: both controls pass and all 11 rows exist.
2. **p0002 equals p0001 in every cell**: outcome, cause, function, in_fault_sites, and the harness verdict word with
   its counts. Neither image's own addresses, which appear in exit texts, are compared. qualify reads the same per
   file on p0001 and p0002: outcome, cause and function, and the `ok` / `not ok` counts. **This is the landing
   criterion.**
3. stock equals 2026-11-11's virtual-malloc rows in outcome, cause and function.
4. p0001 equals 2026-11-11's virtual-nested-pools rows in outcome, cause and function. That includes 06's fault 25 in
   `Perl_regfree_internal`, the platform drift 2026-11-11 recorded; 06 stays outside both counts. 2026-11-11 built
   the one-operand conversion of 0001, and its `control-dev/` two-operand arm read the same on 05 and 06.

## What each outcome means

- **2 holds** (and 1): 0002 moves to `patches/5.36.3/` in the lane, `pending/README.md` loses its row, and the
  commit cites these matrices. The images recorded under earlier results name the patches they were built with and
  are not rebuilt.
- **2 fails in any cell**: 0002 is not landed, and the cell is read first.
- **3 or 4 fails while 2 holds**: platform drift from 2026-11-11. It is written down, and it does not block the
  landing, whose criterion is the pair p0001 / p0002 on one platform.
- **1 fails** (exit 75: a control failed; 78: a builtins SDK): that arm is void and is rerun once.
