# C-81 exposure of the silicon-verdict images: compile-only libcall twins (2026-10-17)

**The question (the runtime lane's routing, after the compiler lane's C-81).** A constant-size copy of unprovable alignment is
expanded inline into scalar chunks, and a scalar chunk never carries a capability's tag. The silicon corpus images were
built with `-ffreestanding -fno-builtin`, so explicit `memcpy` calls already go to the runtime's copy. What remains is
`__builtin_memcpy` and the copies the compiler synthesises (struct assignment, by-value aggregates, array init).

The instrument enumerates every inline-expanded copy: rebuild the image with LTO codegen option
`-capstone-lower-memops-via-libcall`, which turns every mem* expansion into a call, and diff it against the verdict image.
- The twin is compile-only and NEVER RUN. That flag's "does not work yet" status (`CapstoneISelLowering.cpp`) is about
  running a domain under it, and does not apply.
- The diff is the compiler lane's `scan-memcpy-expansion.py --diff`: runs present only in the verdict image (C-81 folder,
  `capstone/tests/compiler-repros/C81-memcpy-unprovable-alignment-strips-tags/`, lane/compiler-exact-bounds-iii).

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
over a region that holds a capability. That is not yet established for any of them. They are not attributed to
functions here.
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

**Result:** 2 runs present only in the verdict image, the same two in each.

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

**What this does not cover:** a scalar copy of a capability written as such in the source (type punning through an
integer type). That is a source property, not C-81.
