# SQLite runs again, and 15 of its 23 source patches retire

Two separate things, found together because the second could not be tested until
the first was fixed.

## SQLite was not running, and no suite would have said so

`run-sqlite-memory.sh` failed. The README says green; the state doc records 3/3
on silicon. SQLite is in NO nightly-tier suite, so the 14 green suites of the
same night said nothing about it.

The failure is not codegen:

    Loadable size = 3336736
    SQ: obs=18446744073709551615      <- create_dom returned -1
    create_dom failed

Without a `.capstone_domreq` section the kernel module sizes a domain's headroom
as `max(2 * code_len, 512 KiB)`, so a 3.3 MB image asks the buddy allocator for
over 10 MB in one block and the DOM_CREATE ioctl fails before a single
instruction runs. `capstone/tests/runtime-qemu/domreq.S` exists for exactly this
and names the case in its own header: "that 3x-on-the-image rule is what pushed
an interpreter image past the buddy allocator's maximum order".

The two-region path is already in the module in the current rootfs. Only the
mruby probe had adopted the declaration; the SQLite build had not. Adding it
(declared dom_data 1 MiB, all of it stack -- SQLite's heap is the in-image
`sqlite_heap[]` array driving memsys5, not dom_data) makes the domain load and
the workload pass, both markers, correct rows.

The declaration is non-alloc, and the build now VERIFIES that no loaded byte
moved, the way the mruby build does. This project has a documented layout
sensitivity where four added instructions flipped a passing run.

## Two wrong readings on the way, both mine

**"SQLite is broken, probably a compiler regression."** Wrong. Nothing in the
compiler was involved; the image simply outgrew a sizing rule.

**"It hangs until the timeout."** Wrong. That run ended `__CAPSTONE_INFRA_FLAKE__
phase=boot-login`, a boot flake. I read the log before the run had finished and
turned an unfinished file into a symptom. The lesson is small and cheap: read the
exit status, not the tail of a log that is still being written.

## 15 of 23 patches retire

With the baseline green, the patches could finally be tested. Each class
corresponds to a compiler defect that has since been fixed.

| class | n | retires |
|---|---|---|
| ternary between two string literals | 8 | yes |
| pointer difference `pReadr - aReadr` | 2 | yes |
| `pMem - (u < nField)` | 1 | yes |
| `SQLITE_INT_TO_PTR` in an initializer, plus the runtime loop that replaced it | 4 | yes |
| `memsys5Methods` built at runtime instead of statically | 1 | **NO, still needed** |

Retired: 15. Remaining: 8, of which seven were never compiler workarounds
(SQLITE_TRANSIENT sentinel, the Atoi64 scope fix, the THREADSAFE guard,
YYDYNSTACK, and the three 16-byte alignment patches from gaps 6 and 8) and one
is `memsys5`.

The `memsys5` verdict is a MATCHED PAIR, not a ladder: removing all 16 candidates
fails (twice), removing 15 passes, and the two arms differ by exactly that one
patch. So it is the static `sqlite3_mem_methods` initializer that still cannot be
built, even though `static-cap-globals` passes and a minimal probe of the same
shape compiles and links. Worth its own investigation; a reduced reproducer does
NOT fall out of the shape alone.

The eight injected helper functions (`sqlite3CapstoneViewOrTable` and friends)
are gone with their callers, re-verified green afterwards because removing them
changes the image.

## What a probe could and could not tell

All five patch classes compile as minimal probes at -O0 and -O2, and for the
ternary class the generated code was checked rather than assumed: it now branches
and materialises a distinct string capability in each arm instead of turning the
select into arithmetic, which is what `convertSelectOfConstantsToMath` returning
`VT.isInteger()` buys.

But `memsys5` also compiled as a probe and is still needed. **A minimal probe that
compiles is evidence that a crash is gone, not that the workload works.** The
only thing that settled the question was the end-to-end run.

## Left undone

SQLite still belongs in the nightly tier. It has now been broken and fixed once
without any suite noticing, and the rule that broke it (proportional headroom)
was a reasonable change made elsewhere.
