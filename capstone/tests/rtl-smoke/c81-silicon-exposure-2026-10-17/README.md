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
