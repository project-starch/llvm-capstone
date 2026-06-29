# Capstone authority / provenance test suite

A reproducible, runtime-grounded evidence base for the paper's **provenance**
(C2) and **granularity** (C1) claims. Each test is a tiny domain whose
*source*, *generated assembly*, and *actual QEMU runtime outcome* are all
captured and checked against a recorded oracle (`oracle.tsv`).

Unlike a codegen-only inspection, this suite **runs** each case in the Capstone
QEMU guest and observes whether the access traps — which already surfaced a
result that the codegen-only analysis got wrong (see "Findings" below).

## Layout

```
domains/                  one .c per test (domain_main(unsigned *res, unsigned func))
oracle.tsv                expected runtime outcome per domain
opt-policy.tsv            probes whose tested operation disappears above -O0
build-authority-suite.sh  compile each domain to .s (asm) and link a .dom
run-authority-suite.py    boot QEMU once per domain, classify vs the oracle
run-authority-suite.sh    thin wrapper (BEEBS/RV8-style)
run-authority-opt-matrix.* additive -O1/-O2/-O3 classification sweep
```

The canonical suite is built at **-O0 on purpose** so source-level operations
under test remain visible. The additive optimization sweep runs eligible probes
at `-O1`, `-O2`, and `-O3`. `opt-policy.tsv` records probes that are skipped
because generated-assembly inspection proves optimization removed the operation
being measured; a successful constant-folded program is not counted as evidence
for the original property.

## Running

```bash
source capstone/tests/capstone-test-env.sh
bash capstone/tests/capstone-authority/run-authority-suite.sh
# -> per-domain PASS/FAIL table + __CAPSTONE_AUTHORITY_SUITE_PASSED__

bash capstone/tests/capstone-authority/run-authority-opt-matrix.sh
# -> domain x {-O1,-O2,-O3} PASS/FAIL/SKIP table
```

Each domain gets its **own boot**: a capability fault inside a domain currently
aborts QEMU (a domain-mode fault hits a `riscv_cpu_do_interrupt` assertion in
the QEMU model), so a single trapping domain would otherwise kill the run for
all that follow. The diagnostic is emitted before the abort, so the trap is
still observed. Set `AUTHORITY_ONLY=name` to run one domain; `AUTHORITY_NO_BUILD=1`
to skip the rebuild. Matrix logs and machine-readable TSV results are written
under `$CAPSTONE_TMP_ROOT/capstone-authority-opt-matrix/`, not the repository.

## Oracle vocabulary

| expect | meaning | QEMU signal |
|--------|---------|-------------|
| `tag-fault` | deref of an untagged value | `Cap mem access requires capability`, no retval |
| `bounds-fault` | deref outside the capability bounds | `Cap mem access OOB`, no retval |
| `ok` | completes with the exact `retval` in `detail` | `retval = <decimal>`, no fault |
| `no-trap-today` | known granularity gap: completes with the exact `retval` in `detail` and no fault | `retval = <decimal>`, no fault |

## Tests and what each proves

| domain | expect | proves |
|--------|--------|--------|
| `forge_inttoptr` | tag-fault | an integer cast to a pointer is untagged; deref traps — integers cannot become authority (provenance, C2) |
| `ptr_int_ptr_roundtrip` | tag-fault | laundering a pointer through a (volatile) integer drops the tag; the reconstructed pointer faults — round-trips are fail-safe |
| `pointer_diff` | ok | a positive pointer difference scales the cursor delta correctly and produces an integer, not authority |
| `pointer_diff_neg` | ok | a negative pointer difference uses signed scaling and returns `-7`, guarding against the former `srli` miscompile |
| `global_inbounds` | ok | ordinary in-bounds global access works (positive control) |
| `global_last_byte` | ok | byte 63 of a narrowed 64-byte global remains accessible |
| `global_one_past` | bounds-fault | dereferencing byte 64 crosses the global's exclusive upper bound |
| `global_unsigned_wrap_index` | bounds-fault | an all-ones unsigned byte index lands before the narrowed global and traps at -O0 |
| `global_signed_negative_index` | bounds-fault | signed index -1 lands before the narrowed global and traps at -O0 |
| `global_oob` | bounds-fault | **the granularity result (C1)**: a[100] in `char a[64]` reaches an adjacent in-segment global; with `-capstone-shrink-globals` (default on) `a`'s capability is narrowed to its object so it traps (was no-trap before SHRINK — set `-mllvm -capstone-shrink-globals=false` to see the before) |
| `global_oob_cross_segment` | bounds-fault | the coarse protection that already exists: an over-read past the *segment* traps even without object SHRINK, because capabilities are segment-bounded |
| `heap_inbounds` | ok | in-bounds access to a `cap_shrink`-narrowed heap allocation works (positive control) |
| `heap_oob` | bounds-fault | **heap granularity (C1)**: `p[100]` past a 64-byte allocation that malloc narrowed with `cap_shrink` traps, even though it stays inside the backing arena (heap analogue of `global_oob`) |
| `stack_inbounds` | ok | in-bounds access to a `-capstone-shrink-stack`-narrowed local works (positive control; built with the flag on) |
| `stack_last_byte` | ok | byte 63 of a narrowed 64-byte stack object remains accessible |
| `stack_one_past` | bounds-fault | dereferencing byte 64 crosses the stack object's exclusive upper bound |
| `stack_oob` | bounds-fault | **stack granularity (C1)**: `buf[100]` past a 64-byte local traps when `&buf` is narrowed to its object by `-capstone-shrink-stack` (built with the flag on; stack analogue of `global_oob`) |
| `subobject_overread` | no-trap-today | `first[8]` reaches the adjacent field because global SHRINK narrows the whole struct, not its fields |
| `many_pointer_args` | ok | the 9th/10th pointer args (stack-passed) are delivered tagged and deref correctly — regression guard for the fixed stack-arg tag-loss bug |
| `spill_reachability` | ok | capabilities survive a register spill (stc/ldc) across a call with tag and bounds intact (PI Q1) |

## Findings (established by running, not inferring)

1. **Provenance is enforced at runtime.** `forge_inttoptr` and
   `ptr_int_ptr_roundtrip` both trap with `Cap mem access requires capability` —
   the codegen prediction (untagged scalar `ld`/`lw`) is confirmed to actually
   fault on the hardware model.

2. **Bounds are SEGMENT-granular today, not object-granular** — and the domain
   image is a *single* `PT_LOAD` segment (see `my_first_domain/link.ld`), so the
   bound is effectively the **whole domain image**. Evidence: the `Cap mem access
   OOB` diagnostic reports `bounds = (base, end)` spanning hundreds/thousands of
   bytes (the segment), never `sizeof(obj)`. Consequences, both observed:
   - **Cross-segment** over-read (`global_oob_cross_segment`, victim at the image
     end) → **traps today**. There is real, coarse spatial protection.
   - **Cross-object within the segment** (`global_oob`) → **does not trap today**.
     This is the T3 granularity gap the C1 contribution closes.

   > This corrects the earlier codegen-only claim that global over-reads "run
   > without faulting" unconditionally — the truth is finer: they run silently
   > only while they stay inside the (whole-image) segment bound.

3. **Object-granularity SHRINK is now implemented (Step 3, C1).** The backend
   narrows every sized data global to `[&g, &g + sizeof(g))` at materialization
   (`CapstoneISelDAGToDAG.cpp`, `selectLGA`), gated by `-capstone-shrink-globals`
   (default on). With it on, `global_oob` flips from no-trap to **bounds-fault**
   (recorded in the oracle); `global_inbounds` and the functional tests still
   pass, i.e. legitimate in-object access is unaffected. Run with
   `-mllvm -capstone-shrink-globals=false` to reproduce the pre-SHRINK
   (segment-granular) behaviour for the before/after comparison.

   **Correctness fallout (measured):** CoreMark ✓, all 7 RV8 benchmarks ✓, and
   the full BEEBS suite 82/82 ✓ with SHRINK on — including the large
   indexed-global-table cases (sha512/aes/norx S-boxes, miniz buffers), the
   patterns most at risk from object bounds. The one case that surfaced
   (`rijndael`) is **a real bug it found, not a false positive.** `aesxam.c`
   declares
   `static char r[4]` and then executes `*(unsigned long*)r = RAND(...)`, an
   8-byte store through a 4-byte object (it assumes `sizeof(unsigned long)==4`,
   false on rv64). Broad/segment bounds silently clobbered 4 adjacent bytes;
   object SHRINK traps the OOB write. This is the canonical "tight bounds expose
   a latent spatial-safety violation" result. (The rijndael build script now
   patches the genuine bug — `r[4]` → `r[8]` — so the benchmark is spatially
   correct and the gate stays green; the finding stands.)

4. **Object-granularity HEAP bounds (C1, malloc side).** The bump allocator
   (`benchmarks/rv8/adapted/rv8_malloc.c`) now narrows every `malloc` return to
   the requested size with `__builtin_capstone_cap_shrink` (recovering its own
   size header through the wide arena capability so `realloc` still works).
   `heap_oob` traps; `heap_inbounds` passes; RV8 (which uses this allocator,
   incl. `dhrystone` records that hold capability fields, and `realloc`) still
   passes. Same precision story as globals: exact < 4 KiB, representable grain
   above.

5. **Object-granularity STACK bounds — feasible (C1, spike, opt-in).** An
   address-taken whole stack object (a bare `FrameIndex`) is narrowed to its
   frame-object size in the backend (`CapstoneISelDAGToDAG.cpp`, `ISD::FrameIndex`),
   gated by `-capstone-shrink-stack` (**default off**). `stack_oob` traps,
   `stack_inbounds` passes, and the flag is exercised end-to-end:
   - **CoreMark ✓** and a stack-heavy BEEBS subset **9/9 ✓** (`fasta`,
     `huffbench`, `levenshtein`, `st`, `recursion`, `dijkstra`, `fir`, `crc32`,
     `nsichneu`) built with `-capstone-shrink-stack=true`.

   Scope of the slice (why it stays opt-in): only **whole-object** addresses are
   narrowed — interior pointers (`&s.field`, i.e. `ADD(FI,offset)`), load/store
   bases, varargs save areas, and dynamic `alloca` are left at the broad stack
   bounds. So this is object- not subobject-granularity, and wider stack
   patterns (especially `-O2` register/frame churn) need broader validation
   before it can be on by default. The lit guard is `cap-shrink-stack.ll`.

6. **A domain-mode capability fault aborts the QEMU model** (`riscv_cpu_do_interrupt`
   assertion `env->priv < PRV_C`). The fault is correctly detected and diagnosed
   first; graceful in-domain fault delivery is a separate runtime gap, noted here
   for the record.

7. **Subobject bounds remain a measured gap.** `subobject_overread` reads from
   one array field into the next field of the same struct and returns the exact
   adjacent-field marker without a fault. Whole-object global SHRINK therefore
   does not provide field-level spatial isolation.

8. **Optimization coverage is explicit.** All 12 eligible probes preserve their
   oracle class at `-O1`, `-O2`, and `-O3`. Eight probes are O0-only because the
   operation under test is removed: two pointer differences, three global
   edge/index loads, the cross-segment load, stack-passed pointer arguments, and
   forced capability spills. Exact reasons are recorded in `opt-policy.tsv`.
