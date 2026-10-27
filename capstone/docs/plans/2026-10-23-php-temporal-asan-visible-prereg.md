# Pre-registration — the temporal cases ASAN already sees

Filed BEFORE the run, per the root `CLAUDE.md` rule. The result commit cites this commit's hash.

## The population, and why it is interesting

`population(corpus, "temporal-asan-visible")`: the **29** `CWE-416` cases in the corpus's
default-build subset that ASAN detects **without** PHP's allocator being patched — the exact
complement, on the temporal axis, of the 31 this project has already measured.

They are not ASAN-blind for one reason: `efree` reaches `free()`. The block genuinely leaves
PHP's size-class cache, so ASAN's poisoning works. That single fact is what makes them a useful
control rather than more of the same, because it is also the fact our seam does **not** depend
on — the seam retires at `_efree`, above the cache, so it should not care either way.

**19 are out of the denominator**: native PHP dies on a signal, so there is no pair to read.
That is the CRASH-057 rule, and it is now applied on the corrected predicate (below).
**10 are scoreable**:

```
CRASH-062  CRASH-068  CRASH-070  CRASH-075  CRASH-078
CRASH-081  CRASH-083  CRASH-113  CRASH-117  CRASH-155
```

## A correction to the apparatus, made before the run and not after

The old test for "the oracle crashes, so there is no pair" was `pristine_outcome != "OK"`. That
conflates a crash with PHP ending itself: PHP exits **255** on its own fatal error — a parse
error, an uncaught exception, a failed include — after printing a deterministic diagnostic. Such
a case has a perfectly good oracle.

The old test never misfired on the ASAN-blind 31, where the only non-`OK` outcomes are two
`SIGNAL:11`. It was a condition satisfied by its input rather than by its logic. On **this**
population it misfires twice: `CRASH-062` and `CRASH-070` are `EXIT:255` from a PHP fatal error
and would have been discarded as "native PHP crashes on it".

The predicate is now `pristine_outcome.startswith("SIGNAL:")`, cross-checked per case against
the corpus's own `crashes_on_pristine_build`; a disagreement stops the run rather than being
resolved in our favour. The gate is negative-tested in the offline suite. The change is inert
for the ASAN-blind 31: that population is still 31, and `ran` still compares against exit 0 for
every case whose oracle is `OK`.

## Predictions, per arm

| | `stock` (0) | `block` (4) | `object` (6) | `exact` (7) |
|---|---|---|---|---|
| all 10 | MISSED | see below | **CAUGHT** | **CAUGHT** |

- **`stock` — MISSED for all 10.** Mode 0 takes one capability per block and never retires it.
  If any case reads CAUGHT here it is a case that faults on any build, as CRASH-022 and
  CRASH-102 do among the 31, and it says nothing about the protections.
  (CORRECTED 2026-10-27: CRASH-022 "faults on any build" because PHP's `_zend_qsort_swap`
  strips the tags of the array it sorts, a port artefact fixed by patch 0002, not because of
  its defect; CRASH-078 in this population is the same artefact. See the results README's
  "Correction".)
- **`object` and `exact` — CAUGHT for all 10.** The seam retires at `_efree`, which these
  cases reach by construction. This is the prediction most likely to be wrong in an
  interesting way: a fault that lands outside the case's own ASAN site reads `unattributed`,
  which is what happened to CRASH-140 at `block` in the previous run.
- **`block` — the real question, and the one cell named in advance as most likely to move.**
  Mode 4 performs **no revocations** (measured: 0 / 0 / 8937 / 8937 in the previous run), so
  every catch it produces comes from a freed block being reused and overwritten with scalar
  data, which clears the tag. Because these blocks genuinely return to the heap and can be
  reused by *unrelated* allocations — rather than being parked for the same size class, as a
  cache-hidden block is — I predict **block catches a strictly higher fraction here than the
  6 of 21 it managed on the cache-hidden cases**. I am not predicting a number.

## Two cases carrying their own caveat

- **CRASH-070.** Its recorded pristine run prints `failed to open stream: soap12-test.inc`, so
  the oracle itself ran without the include the trigger needs. Whatever it reads, the reading
  is of a degraded oracle, and it is reported with that attached rather than quietly counted.
- **CRASH-062.** `EXIT:255`, PHP's own `Fatal error: Class declarations may not be nested`.
  The control arm must reproduce exit 255 **and** that text, which no run of this harness has
  ever been asked to do before.

## What would make this measurement void

The cross-arm gates are unchanged and all must pass: revoke instructions present in the image,
distinct mode lines per arm, strictly more revokes on the retiring arms, the bound probe
faulting as declared, and the benign workload completing with revocation on.

## Vehicle

The **identical image** as the ASAN-blind measurement — `php-real-seam.dom`, sha256
`b36d053690136b03f92dcb8ebb67bde2759168fd3167d17c6f3fdaeebebf8836` — on the merged QEMU at
submodule `1a6dd20732`, so the two tables are comparable without a second build to argue about.

## Amendment, after the first run and before the second

The first run of this population voided a whole column and the reason was the instrument, not
the port. Recorded here, before the re-run, so the result cites the amendment rather than
explaining itself afterwards.

**`--run-timeout` was 40 s, and the exact arm's `revocation-sanity` crossed it.** The arm-level
control therefore read `did-not-complete`, and the judge correctly refused all nine of that
arm's scoreable cells as `control-failed`. Measured afterwards with a 240 s ceiling, the same
control on the same image completes in **14.5 s** (stock 11.7, block 12.6, object 14.1), so 40 s
was never a deterministic shortfall -- that boot ran 3x its nominal cost and crossed a ceiling
sitting just above it. The same 40 s also killed two `object` cells outright (CRASH-083 and
CRASH-155, `infra`, watchdog). The re-run uses `--run-timeout 240`, roughly 16x nominal.

Nothing about the predictions changes. The arms, the image, the population and the ten
scoreable cases are exactly as registered above; the image sha stays
`b36d053690136b03f92dcb8ebb67bde2759168fd3167d17c6f3fdaeebebf8836`, byte-identical to the
published 31-case run, because an arena rebuild would cost that comparability.

**Two judge defects the run exposed, both fixed before the re-run, both inert for the published
31.** Each one silently cost a reading that the population's own cases are the first to need:

1. `observe()` set `completed = (exit_code == 0 and fault is None)`. For a case whose pristine
   record is an EXIT rather than a signal -- PHP ending itself with a diagnostic and status 255
   -- that can never hold, so every arm decayed to `inconclusive`, the control arm included,
   which takes the case's whole column with it. `completed` now compares against the oracle's
   own rc, which is the generalisation already applied to `ran`. CRASH-062 and CRASH-070.
2. The per-case oracle comparison compared stdout literally, while the stock-arm comparison
   normalised the script path. PHP prints the script's absolute path inside its own warnings and
   fatal errors, and the corpus's host path and the guest's `/mnt/vm` are both correct, so the
   literal form reported a divergence on every such case -- "2 lines vs 2", differing only in
   the path. Both sites now normalise.

**One prediction above is already falsified and is not revised here.** CRASH-078 was predicted
MISSED on stock; stock caught it, `cause=24` at `ext/standard/array.c:162`, ASAN's own site,
with `exact=0 revoke=0` and `retires=0 revokes=0` confirming no revocation took place. The
mechanism is PHP's own `AG(cache)`: `_efree` parks the block, `_emalloc` hands the same payload
capability back, the new tenant overwrites what was a pointer field, and the stale read loads an
untagged value. That is a tag-clobber catch needing no revocation at all -- the mechanism named
above as the source of `block`'s catches, which turns out to be live in `stock` too. The stock
column is therefore not a capability-free baseline, and the re-run is not expected to change
this cell.

**Three cases are expected to stay NO-READING for reasons the timeout does not touch**, and
raising the ceiling must not be read as expected to recover them: CRASH-081 faults inside the
seam's own source (`zend-alloc/sublet/zend-cap-blocks.c:425`) on every arm, a port artefact;
CRASH-068 faults at an unsymbolised pc before the case's marker on every arm; and CRASH-075
exhausts the seam's block provider (`code=1 the heap would not lend a block`, at ~8,000 live
blocks) on every arm, an instrument capacity limit that needs a larger arena and therefore a
rebuild this run declines to make.

## Second amendment — three instrument fixes, filed before the run that tests them

The first readable run of this population produced five non-catches. Each was investigated to a
mechanism, and **none of the five was the capability mechanism failing to notice the bug**. Three
had a cause fixable by a build or launch parameter; those three are fixed here, and this
amendment records what each is predicted to do BEFORE the run that tests it.

### What was wrong, and what changed

1. **CRASH-081 — the guest stack, not the allocator.** The fault was `cause=28` with `rs1=x2`
   (the stack pointer), address `0x3fb1affff0`, bounds `0x3fb1b00000..0x3fb1c00000`: a 16-byte
   store 16 bytes BELOW a stack of exactly 1 MiB, at a function prologue. The trigger recurses
   deeply on purpose, so PHP exhausted the stack before reaching the use-after-realloc; Linux
   gives ASAN 8 MiB and it survives. The symbolised line (`zend-cap-blocks.c:425`, the opening
   brace of the exit-reporting function) was whichever frame was being entered, not a seam bug.
   **Fix: `CAPSTONE_APPLICATION_STACK_BYTES` 1 MiB -> 8 MiB** (and `DATA_BYTES` -> 16 MiB, since
   `Application.cmake:17` requires data to cover stack). Verified in the new image:
   `.capstone_domreq` now reads `DATA=16843008 STACK=8388608`.

2. **CRASH-046 and CRASH-075 — revocation node IDs, not memory.** Both reported `code=1 the heap
   would not lend a block`, at 12,646 and 8,038 live blocks. `heap.c`'s
   `__capstone_sublet_malloc_linear` opens with `ensure_ids(256)`, which returns `-ENOMEM` when
   the module cannot grow the node table; the heap then returns NULL. The module defaults to
   `node_initial_pages=4` growing in batches of 16, one page carries 256 nodes, and this seam
   mints ~4 nodes per block with `collections=0 reclaimed=0` -- retired IDs are never swept, so
   demand only grows. **Fix: the guest now `insmod`s with `node_initial_pages=1024`** = 262,144
   nodes, the value and the reason the retarget plan already prescribed. The insmod line is
   pinned by three offline checks, because losing the parameter would silently restore the
   failure under a message that says nothing about sizes.

3. **CRASH-068 — a catch inside a function with no debug info.** `cause=24` at a pc that
   symbolised to `??:0`. Resolved by symbol: **`strlen + 0x14`**. ASAN reports the
   use-after-free at `spprintf.c:518`, and the trigger's own comment says the error path does
   `free(c->name)` and then formats that name -- so `spprintf` calls `strlen` on the freed
   string and our tag check fires on exactly that read. **Fix: the SDK's C files are now built
   with `-g`**, so `string_bounds_safe.c` carries DWARF. Verified: that pc now resolves to
   `string_bounds_safe.c:40:9`, where the old image gave `??:0`.

   NOTE: the flag must be `CMAKE_C_FLAGS_RELEASE`, not `CMAKE_C_FLAGS`. The toolchain assembles
   its `-ffreestanding -nostdinc -isystem<musl>` prologue into `CMAKE_C_FLAGS_INIT`, which CMake
   honours only while `CMAKE_C_FLAGS` is absent from the cache; setting the latter discards the
   prologue and clang finds host glibc headers. The first attempt failed exactly that way.

### Predictions

| case | now | predicted after | why |
|---|---|---|---|
| CRASH-046 | setup-fault (all arms) | a reading on all arms | 262,144 nodes vs ~50,000 needed |
| CRASH-075 | setup-fault (all arms) | a reading on all arms | same |
| CRASH-081 | setup-fault (all arms) | a reading on all arms | 8 MiB stack for a deep recursion |
| CRASH-068 | setup-fault, site `??:0` | **`unattributed`, site `string_bounds_safe.c`** | `-g` makes the fault legible, NOT attributed: the rule wants ASAN's own file and this is a libc primitive called from it |
| CRASH-113 | unattributed | unattributed | untouched by all three fixes |
| the 6 caught | CAUGHT on 3 protected arms | unchanged | untouched by all three fixes |

**CRASH-068 is stated as still-NO-READING on purpose.** `-g` buys a named site, not a verdict.
Promoting it needs the attribution question settled -- a fault inside a string primitive
attributed to the caller ASAN names -- which is the same open question as CRASH-008 and
CRASH-073, now three cases rather than two. It is not settled here and is not being settled by
a build flag.

### The images changed, so the published comparison must be re-made

```
php-real-seam.dom  b36d0536...8836  ->  3cff5a15b9fa737b60ffb519470c3fab889c3ad02430f6127216913611183cc8
probe-seam.dom     5cca4723...a3d0  ->  419ec120a0ac9af115ba591fb746fd1951ce4fa329011980fceffb4995207174
```

The old pair is kept byte-for-byte so the published 31-case numbers remain reproducible. But a
bigger stack and a bigger node table are changes the ASAN-blind 31 were ALSO measured without,
so that population must be re-run on this image before the two tables are set beside each other.
Until that re-run exists, the published 25-of-28 and anything measured here are on different
instruments and are not comparable.

---

# Third amendment (2026-10-26): a new instrument, and BOTH populations on it

Filed before the run, as the rule requires, and the result commit cites this commit's hash.

## Why the earlier predictions cannot be carried over

Everything between the trigger and the hardware changed since the second amendment:

| layer | change |
|---|---|
| upstream | merged `origin/dev`, 879 commits (`8001cabd3e05`) |
| arena ceiling | `CV_MAX_ARENAS` 512 -> 8192 -- the reason for the merge |
| firmware + monitor | rebuilt at the recorded buildroot pin (`d04bd83` -> `8fd1ea1`, opensbi `49b0f93` -> `cf344cf`, monitor `2c49c41` -> `4674ab6`). The KERNEL SOURCE is unchanged (`830b3c68` both sides) |
| musl | `0009` rewritten (+344), `0010` (+42), `string_bounds_safe.c` (+108), `sublet_heap.c` deleted |
| allocator seam | block provider re-based from the deleted Sublet loan onto CDERIVE/CREVOKE (`fa8902138ab0`) |
| platform properties | `x-capstone-test-csrs=true`, `supervisor-slots=32` -- without which NOTHING runs |
| bounds | Option (iii): exact below 4 KiB, `2^(msb(len)-9)` granule at or above |

## The baseline of record, and an honest gap in it

The run after the second amendment (the fourth measurement of this population) was **never
committed**: no bundle, no table, no record of any kind. Its numbers exist only in a working
session, so they are not evidence and are not cited here. The baseline of record therefore
remains the state the second amendment describes, and this run is the **first recorded
measurement of this population on a platform whose own gate passes** (`run.py` 42/42,
`ports/php/app/results/platform-qualification-2026-10-26/`).

That gap is itself a finding about the apparatus: `run-realphp.py` writes a bundle only when it
is asked to, so a run driven by hand leaves nothing behind. Treat an uncommitted run as not
having happened.

## Predictions, 29 `temporal-asan-visible`

Per arm, unchanged in shape from the second amendment:

| | `stock` (0) | `block` (4) | `object` (6) | `exact` (7) |
|---|---|---|---|---|
| cases reaching `_efree` | MISSED | see below | **CAUGHT** | **CAUGHT** |

And these specific cells, each falsifiable:

1. **CRASH-046, CRASH-075, CRASH-081 become readings on all four arms.** The measured cause of
   their `setup-fault` was `errno 28` (ENOSPC) with `arenas=512` exactly, at every failure on
   every arm, against passing controls at `arenas=17`. The ceiling is now 8192 and `vm-mappings`
   -- which holds 1200 mappings and fails on the old module -- passes.

   **This is the prediction most likely to fail, and it has failed before.** The second
   amendment predicted the same outcome for these same three cases, attributing them to node
   exhaustion, and was wrong: capacity was 262,144 against a peak of 64,587-90,218 with
   `node_growths=0`. If they stay `setup-fault` now, the arena ceiling was not the whole cause
   either, and the next step is to read `b=` on the failure rather than to theorise again.

2. **CRASH-068 moves its fault site but stays `NO-READING unattributed`.** It faulted inside a
   libc string primitive (`string_bounds_safe.c:40`, `strlen`), and that file changed by +108
   lines along with musl's `memcpy`/`memmove`/`memset` word paths. So the site should move. It
   should NOT become CAUGHT: attribution wants ASAN's own file and line, and a libc primitive
   called from PHP is not that, however the bytes moved.

3. **CRASH-113 stays `unattributed`.** It faults at `zend_execute_API.c:626` where ASAN names
   `zend_hash.c:850`, and nothing in this work touches it. A change here would mean the
   discrepancy was an artefact of the old instrument and would need explaining before use.

4. **The six cases previously CAUGHT on the three protected arms stay CAUGHT.** The boring
   prediction, and the one that matters most: the seam's block provider was rewritten. If any of
   the six regresses, that is a re-base defect and not a finding about PHP. `probe-seam.dom`
   reproduces all ten of its documented stages, so I expect no regression -- which is exactly
   why a regression would be informative.

5. **`block` catches a strictly higher fraction than the 6 of 21 it managed on the
   cache-hidden cases.** Carried forward unchanged. No number predicted.

6. **No case is CAUGHT on `exact` by an overflow inside the Option (iii) granule.** New, and it
   is a limit rather than a hope: at or above 4 KiB the bound is rounded up to
   `2^(msb(len)-9)`, so a small overflow past a large object is NOT catchable. Any CAUGHT whose
   fault address lies within that slack is an artefact and must be refused.

## Predictions, 31 `asan-blind`

Re-run on the same instrument, which the second amendment already said was required before the
two tables may be set side by side.

- **24 reachable cases: CAUGHT on the protected arms**, as recorded.
- **The 7 OS-dependent cases stay `not-reached`**: CRASH-008, 084, 086, 099, 112, 092, 100.
- **CRASH-100 specifically stays out of the denominator**, and for a reason that is a property
  of this build rather than of the platform: it needs `stack-use-after-return`, and the image
  still reports `shrink-stack: off (the comparison build)`. Turning it on is its own variable
  and its own pre-registration; it is NOT part of this run.
- **CRASH-008 and CRASH-073 move their fault sites but stay out of the denominator**, same
  reasoning as CRASH-068: `stpcpy` and `memcpy` are libc primitives, and the merge rewrote them.

## What would make this measurement void

The cross-arm gates are unchanged and all must pass, with one replaced because the old form
silently stopped meaning anything:

- the image must contain BOTH capability instructions it needs -- `derive=16 revoke=6` measured
  by `count-cap-insns.py`, by ENCODING, because `objdump` prints CDERIVE and CREVOKE as
  `<unknown>` and the old `grep -c '\brevoke\b'` would pass a correct image for absent;
- distinct mode lines per arm;
- strictly more revokes on the retiring arms than on the control;
- the bound probe faulting as declared;
- the benign workload completing with revocation on;
- and, newly: any run whose node high-water comes within `CV_NODE_RESERVE` of capacity is
  refused, because a cause-30 resource fault lands at an arbitrary pc and is otherwise
  indistinguishable from a catch. Capacity is 262,144 on the paged table; the flat table's
  65,536 would NOT clear this population's measured 64,587-90,218, which is why the paged
  table is used.

## Vehicle

    clang          9bdb06e96beaedb9805ec63d8eb494bfb052e09e8b9904ff4b998443752b62e6  (f3b69c7af1c4)
    qemu           d9de56fc74c77f8a2fbfed79624c3ff73baa7110192a7e07b7829989e5957d6d
    fw_jump.elf    b1663eb62a192741055589bafd60aea8121d464efb3afd7da2b98508316639ef
    Image          108d811705e71414d022748d21fb040b4dc18ca15df87faf637704091f73bfe5
    rootfs.ext2    aa811268f1a69f7f79bfc09d778267d6370d306414f707cda1908eecc591626a
    capstone-vexec 8489f201ae0447f938a9fa8f6856b6719a54810d0c0a433873a037e731bf68d1
    capstone_vm.ko 5878060e59e1b114a99d3e04128284508cfb13ffd2356204b943856aa78837be
    php-real-seam  f8a15344e9fe31942566ef2a44011972672390660ebaef76db362c57f595372f
    probe-seam     9481b7cea04809ae4621f56d4decc596e1e7b9071afa51fa4c1522baabbbc744
    -cpu           rv64,sstc=false,h=false,sv48=false,sv57=false,x-capstone-u-mode=true,
                   x-capstone-exact-bounds=true,x-capstone-test-csrs=true,
                   x-capstone-supervisor-slots=32
    insmod         node_initial_pages=1024 supervisor_slots=32

The image is byte-identical across two builds. The `-cpu` string and the module arguments are
recorded in every bundle's `inputs.json` from this run forward: they decide whether the platform
runs at all, and two runs differing only in them are not the same measurement.
