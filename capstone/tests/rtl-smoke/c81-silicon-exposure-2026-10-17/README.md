# C-81 exposure of the silicon-verdict images: compile-only libcall twins (2026-10-17)

**The question (the runtime lane's routing, after the compiler lane's C-81).** A constant-size copy of unprovable alignment is
expanded inline into scalar chunks, and a scalar chunk never carries a capability's tag. The silicon corpus images were
built with `-ffreestanding -fno-builtin`, so explicit `memcpy` calls already go to the runtime's copy. What remains is
`__builtin_memcpy` and the copies the compiler synthesises (struct assignment, by-value aggregates, array init).

The instrument is a twin in which every inline-expanded copy is a call: rebuild the image with LTO codegen option
`-capstone-lower-memops-via-libcall`, which turns every mem* expansion into a call, and compare it with the verdict
image. **The scanner diff below does not enumerate those copies; see the correction that follows.**
- The twin is compile-only and NEVER RUN. That flag's "does not work yet" status (`CapstoneISelLowering.cpp`) is about
  running a domain under it, and does not apply.
- The diff is the compiler lane's `scan-memcpy-expansion.py --diff`: runs present only in the verdict image (C-81 folder,
  `capstone/tests/compiler-repros/C81-memcpy-unprovable-alignment-strips-tags/`, lane/compiler-exact-bounds-iii).

## Correction (2026-10-17, later the same day): the scanner diff sees a fraction of the inline copies

**Withdrawn:** "the instrument enumerates every inline-expanded copy". This section and the SQLite section first rested
on that claim. claim-auditor, checking the mruby result, counted the mem* calls the twin gains instead of the scanner's
runs.

**The count.** `memcalls2.py` counts every `auipc`/`addi` reference to `memcpy`, `memset` and `memmove`, per function,
in both images. The references only the twin has are the expansions the verdict image carries inline:

| image | inline mem* expansions the twin diverts | the scanner's plain-only runs |
|---|---:|---:|
| SQLite 02 backupattach | 48 | 2 |
| SQLite 05 detachtrig | 47 | 2 |
| set 1 controls | 12 | 8 |
| set 1 memcached-heap-00 | 13 | 9 |
| mruby | 2,957 | 89 (shape key) |

**Why the scanner misses them.**
- It reports only scalar runs of 16 bytes or more.
- `ldc`/`stc` end a run, which is intended: a capability-width copy keeps its tags. But so does every instruction whose
  name starts with `c`, so `cincoffset`/`cincoffsetimm` between the chunks of a scalar copy split it below the threshold.
- Its diff keys runs by shape, not by function.

**The checks that replace it,** run on the same images:
1. `rescan-nosplit.py`: the scanner's run detector, with only control flow and `ldc`/`stc` ending a run. It lists the
   plain-only runs that carry a load and store 16 bytes or more.
2. `granule-check.py`: over those runs, it flags one whose copy-stores cover 16 contiguous bytes from one base. A
   copy-store stores a register that was loaded earlier in the same run. Sixteen contiguous bytes are the least a scalar
   copy needs to move a whole capability. Its control: the `copy32` shape is flagged, and a zero fill is not.

| image | plain-only runs | flagged | what the flagged run copies |
|---|---:|---:|---|
| SQLite 02, 05 | 2 each | 1 | `cacheEntryClear`'s `yColCache` (`int, i16, u8, int, int, int`) |
| set 1 controls, heap-00 | 2, 3 | 1 | `dl_round`'s `struct capstone_signal_event` (integers and 128 raw siginfo bytes) |
| mruby | 65 | 3 | `dl_round` as above; `capstone_spawn_pack`'s `struct capstone_spawn_action a = actions[i]` (six `uint32_t`; `path` is an offset); `mrb_ary_delete+0x358`, the `tt` + padding half (0x10-0x1f) of an unboxed `mrb_value` |

SQLite's second run is in the soft-float builtin `__floatdidf`. It stores one register three times and then loads,
so it copies nothing. So every expansion the twin diverts is one of:
- capability-width (`ldc`/`stc`), which keeps tags;
- shorter than a capability;
- a fill;
- a scalar copy of a pointer-free object.

**The SQLite 02/05 verdicts stand on this evidence.** The two-run diff alone did not support them.

**How the twin reaches the code that matters.** The B0 pipeline is full LTO: the case code, the runtime and the musl
archive (`/tmp/capstone/b0/musl-gpct-lto/libc-capstone.a`) are all bitcode, and lowering happens at the link. So the
option goes to the LTO plugin, through `build-b0-hello.sh`'s new `B0_PLUGIN_EXTRA` hook (empty by default). A compile-time
`-mllvm` would have produced a byte-identical twin.

**The controls, all three required:**
1. **The plain rebuild must reproduce the board's image hashes**, so the hook is inert when empty and the inputs are the
   board's. Recipe: the compiler lane's frozen dev147 toolchain (clang-22 sha256 504dca34…), the musl archive above, and
   `run-silicon-cases.py build`.
2. **Every twin's hash must differ from its plain image**, so the option reached LTO codegen.
3. **The scanner's own positive control must pass** (repro.c through the kit's clang): `copy32_unknown` and
   `copy32_unknown_builtin` flagged, `copy32_known` clean, and only `copy32_unknown_builtin` still flagged with
   `-fno-builtin`. Without it the scanner refuses to report.

## Set 1: memcached plain-heap, `silicon-level0` (`twin-set1.sh`)

- **Control 1:** the controls image and memcached-heap-00..08, all 10, equal `build-silicon-level0.log` of 10-16 04:39 hash
  for hash.
- **Control 2:** all 10 twins differ.
- **Control 3:** PASS, 6 of 6 lines.

| image | expansions only in the verdict image | by unit (bytes per scalar move) | unit 1 |
|---|---|---|---|
| controls | 8 | 2, 4, 8 | 0 |
| memcached-heap-00..08 | 9 each | 2: 1, 4: 1, 8: 7 (heap-00; the same counts in each) | 0 |

Every run reported is at least 16 bytes. The largest are a 162-byte run at unit 2, then 72, 64 (twice), 48, 40 and 16
(twice) bytes at unit 8. The controls image carries 8 of the 9, so they are in shared runtime or libc code, not in the
cases.

**What this establishes:**
- each set-1 image carries 8-9 scalar inline copies of 16 bytes or more that the libcall lowering would turn into calls;
- none is byte-granular.
**A scalar copy of any width drops a capability's tag.** So each of these is a CANDIDATE, and the finding would be one
over a region that holds a capability.

**Attribution** (`attribute.py`, which maps each twin-only run to its enclosing non-`.L` symbol and prints its
instructions). All nine cases carry the same nine runs; only `main`'s offset moves (+0xec or +0xf0). The controls image
carries the same set without `main`'s, with `printf+0x7c` in place of `vfprintf+0x44`.

| run | what it stores | can it drop a tag? |
|---|---|---|
| `abort+0x50`, 48 B | `sd zero` x6: a zero-filled stack local | no, it loads nothing |
| `dl_ioctl+0x14c`, 64 B | `sd zero` x8: a stack local at `s0-0x180` | no |
| `dl_prctl+0x4c`, 72 B | `sd zero` x8 over two locals, then an integer spill | no |
| `dl_prctl+0x244`, 16 B | an integer spill, then `li a0, 0` and `sd a0` | no |
| `dl_prctl+0x2e4`, 16 B | the constants 0x20 and 0x10 | no |
| `vfprintf+0x44`, 40 B | `sd zero` x5: 40 bytes, the size of musl's `nl_type[NL_ARGMAX+1]` | no |
| `main+0xec`, 64 B | `sd zero` x8: a stack local at `s0-0xe0` | no |
| `run_event+0x500`, 16 B | `sw s1` and `sw zero` x3, the integer fields of a struct whose capability field is stored by its own `stc` | no |
| `dl_round+0x35c`, 162 B | the ONE copy: `ld`/`sd` over 160 bytes, then `sh`. It is `events[slot].ev = block->events[i]` in `__capstone_signals_take` (signals.c), inlined by LTO, followed by the 2-byte `state`/`on_alt` store. | no: `struct capstone_signal_event` (`runtime/include/capstone/delegate.h:62`) is `seq`, `signo`, `flags`, `mask`, `generation` and `info[128]`, the raw siginfo bytes the launcher wrote. It holds no pointer. |

**So none of set 1's candidates is a C-81 instance.** Eight load nothing, and the one copy moves a pointer-free wire
record.
**The verdicts they bear on:** set 1's rows are spatial overreads (cause 28), not cause-24 temporal rows, so these
candidates touch no CAUGHT-by-24 reading in this set.

## SQLite stock: the two CAUGHT-24 rows, 02 backupattach and 05 detachtrig (`twin-sqlite.sh`)

These are the only rows in the silicon totals where a cause-24 verdict could in principle be a C-81 artefact (C-81 strips
tags, so it can only fake a cause-24 reading). The records: 02 wedges at `sqlite3BtreeUpdateMeta+0x48`
(`ldc a1, 0x80(a1)`), 05 at `sqlite3DropTriggerPtr+0x154` (`ldc a3, 0x0(a1)`), both cause 24
(`repro322-silicon-2026-10-16/RESULTS.md:18-19`).

**How the twin reaches the code.** `build-sqlite-silicon.sh` is not LTO. `EXTRA_MLLVM` goes into `SILICON`, which every C
object in the link is compiled with: the amalgamation, `libc.o`, `beebs_string.o` and the soft-float builtins.
`start.o` and `domreq.o` are assembly.

**The flag diverts every inline mem* expansion this build can emit.** Read in dev147's own source (612b3ec514c0):
- `CapstoneMemOpsViaLibcall` makes `findOptimalMemOpLowering` refuse (`CapstoneISelLowering.cpp:25793`);
- the target's own `EmitTargetCodeForMemcpy` emits only under `-capstone-memcpy-high-half-fixup`, default off and not in
  this build's flags;
- `EmitTargetCodeForMemset` needs Xqcilsm;
- the only `getMemcpy` in the call lowering (byval arguments) passes `AlwaysInline=false` (`:24411`);
- the sources contain no `__builtin_memcpy_inline`.

**Controls:**
1. The plain rebuilds equal `repro322-silicon-2026-10-16/SHA256SUMS`: backupattach 8fe5a153c1229ac3, detachtrig
   02ca19e2f6f5affe.
2. The twins differ: b22ee4e5579c4ae6 and b781ab8d54735c2c. Object by object, `amalgam.o` and `floatdidf_ng.o` change;
   `libc.o`, `beebs_string.o` and the builtins come out byte-identical, so they hold no expansion for the flag to divert.
3. The scanner's self-test passes in both diffs, 6 of 6.

**Result:** 2 runs present only in the verdict image, the same two in each. These are the scanner's runs; the twin
diverts 48 and 47 expansions, which the correction above covers.

| run | where | what it is | can it drop a tag? |
|---|---|---|---|
| 48 B of `sd zero`, then `sb 9` at offset 0 (the scanner counts 49 B, unit 1) | `yy_reduce+0x1fc4` in both images | `SelectDest dest = {SRT_Output, 0, 0, 0, 0, 0};` (the build's adapted amalgamation `sqlite3-capstone.c:140841`), a fresh 48-byte stack local at `s0-0x100`. `SRT_Output` is 9. | No. It copies nothing: every store writes a constant, and the twin's `memset` writes the same zeros. |
| 20 B, five `lw`/`sw` pairs | `cacheEntryClear` | `pParse->aColCache[i] = pParse->aColCache[pParse->nColCache]`, stride 20 | No. `struct yColCache` is `int, i16, u8, int, int, int` and holds no pointer. |

So neither run can produce an untagged capability, on any path. The register each faulting `ldc` read cannot have come
from them, and the question of whether they lie on its path does not arise.

**Both rows stay CAUGHT with respect to C-81.**

The 98 runs present in both images survive the libcall lowering, so they are not mem* expansions. They are ordinary
field-typed stores, of the kind the scanner header calls a candidate and not a finding. A pointer field is moved by
`ldc`/`stc` there.

**Over every scalar run in the image, not only the diverted ones** (`allruns.py` with `granule-strict.py`, which also
requires the 16 copied bytes to come from one source base in the same layout): 171 and 168 runs, 2 flagged in each.
- One is `cacheEntryClear`, above.
- The other is an 8-byte spill chain in `wherePathSolver+0xbc`: `ld a0, -0x198(s0); sd a0, -0x190(s0); ld a0,
  -0x190(s0); sd a0, -0x188(s0)`.

So no scalar copy of a 16-byte object exists anywhere in these images except the pointer-free `yColCache`.

**What this does not cover:** a scalar copy of a capability written as such in the source (type punning through an
integer type). That is a source property, not C-81.

## mruby: B5's `mruby.dom`, the image of the silicon mruby-bounds rows (`twin-mruby.sh`)

The rows are outside the silicon totals, so this section completes the record; no verdict depends on it. The image is
full LTO through `build-b0-hello.sh`, so the twin takes set 1's `B0_PLUGIN_EXTRA` hook.

**Controls:**
1. **The plain rebuild equals the record, a7ac283d7050a5d8.** It is built from the tree that built the board image
   (7049b13d969e, B5's landing) plus the inert hook, with dev147 and the 10-04 musl archive (`silicon-cflags.txt`).
   - The mruby source is the board build's pinned tree, copied.
   - `build-b5-mruby.sh`'s patch-set stamp hashes `sha256sum` output, which carries the patches' absolute path. So the
     copy is re-stamped for the new tree's path, after the patch contents are compared.
2. The twin differs: f87f9cbf0fa8e5a6.
3. The scanner's self-test passes, 6 of 6.

**Expansions:**
- The twin diverts 2,957 inline mem* operations in 893 functions. Most are `mrb_value` copies at `ldc`/`stc` width.
- `rescan-nosplit.py` leaves 65 load-carrying plain-only scalar runs, and `granule-check.py` flags 3. All three copy
  pointer-free objects (see the correction).
- 34 of the 65 have one shape: four `sw` at offsets 0x10-0x1c. That is the `tt` + padding half of an unboxed `mrb_value`.
  The build sets `MRB_NO_BOXING` (`boxing_no.h`: a 16-byte union, then the type tag at 0x10, a 32-byte stride), and the
  capability half is stored by its own `stc` just before.

**The cause-24 rows.** A C-81 copy strips the tag and keeps the address bits, so a later use would fault with the
address of something in the domain.

Two premises carry the argument, both read from primary sources:
- **"address" is the operand's integer view.**
  - The fault line prints the trap value, which the monitor logs as `SUPT`.
  - In the RTL at d93a8d66b, a cause-24 load reports the full effective address (`core/load_store_unit.sv:1327-1329`
    and `:1384`).
  - A capability-ALU fault reports `operand_a`, the rs1 cursor without the immediate (`core/ex_stage.sv:489-498`).
- **This domain's addresses are all far from the fault values.**
  - Base: `DBAS:AC100000`. Code: 0xac100000-0xac290000. The lowest shared region is `BASE:AC040000`.
  - Heap faults land at 0xae0a61e0-0xae0ce080.
  - QEMU lays the domain out exactly 0x34200000 higher. Every pc, and every cause-28 heap address, differs by that
    much between the two platforms.

| row | fault | address, silicon / QEMU | what the value is |
|---|---|---|---|
| 01, 07 | `mrb_vformat+0x5e8` `lbu a1, 0x10(a0)` | 0x10 / 0x10 | a base of 0, a NULL. Several paths reach the load, two with `a0` zeroed by the code itself (`mrb_class`'s `default: return NULL`) |
| 02 | `mrb_vm_exec+0x718` `lbu a3, 0(a0)`, after `ldc a0, 0x40(a1)` | 0x1604 / 0x1604 | the VM's instruction fetch through `ci->pc`. It reads the same on both platforms, so it is not a pointer |
| 17 | `mrb_func_basic_p+0x60` `ldc` | 0x30274e0b65a71c20 / the same | bit 61 set, beyond any physical address, and the same on both platforms |
| 18-21 | `mt_put+0x4c` `lw` | 0 / 0 | a NULL method table, as the four case names say. `mt_put`'s own inline copies are `ldc`/`stc` |
| 22 | `mrb_proc_parameters+0xcc` `cincoffsetimm a1, a4, 0x50`, after `ldc a4, 0x20(a1)` | 0x7d8000 / 0 | mruby's inline-packed symbol `:f0`: 'f' gives 7<<20 and '0' gives 54<<14 (`src/symbol.c:132-150`). `src/class.c:3323` stores the alias's method id in the proc's body, where an irep pointer belongs, with a scalar store and no copy. QEMU leaves the trap value 0 for this instruction |

**What is established:** no faulting operand of the nine cause-24 rows is a tag-stripped capability.
- Eight of the nine addresses are byte-identical on both platforms, although the platforms place the domain 0x34200000
  apart.
- Row 22's value is identified.
- The three fault functions that have diverted expansions (`mrb_vformat`, `mrb_vm_exec`, `mrb_proc_parameters`) do not
  change that, because the argument is about the value, not about where it was written.

**What is not established:** that C-81 has no indirect effect, where a tag stripped elsewhere changes control flow so
that a different, non-pointer value reaches the fault. No such path has been found or proposed.
- The experiment that would settle it is to run the twin (f87f9cbf) under QEMU's silicon-mimic profile on the nine rows.
  `smoke.rb` must pass first, because the flag's status for running is "does not work yet".
- QEMU already reproduces 8 of the 9 silicon addresses exactly.

**Sets 2-9 are not twinned.** None of their in-scope rows reads cause 24, the only reading C-81 can produce. Their
twins would list candidates and touch no verdict.
