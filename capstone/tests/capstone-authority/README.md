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
build-authority-suite.sh  compile each domain to .s (asm) and link a .dom
run-authority-suite.py    boot QEMU once per domain, classify vs the oracle
run-authority-suite.sh    thin wrapper (BEEBS/RV8-style)
```

Built at **-O0 on purpose**: this is an authority / ISA-behaviour suite, so the
source-level operations under test (forged derefs, OOB loads, 9th/10th
stack-passed pointer args, spills) must be preserved 1:1 rather than optimised
away — at -O2 the compiler merges globals, does IPCP, constant-folds and DCEs,
all of which defeat these probes. The Step-3 SHRINK narrowing is inserted in the
backend, so it applies at every optimisation level regardless.

## Running

```bash
source capstone/tests/capstone-test-env.sh
bash capstone/tests/capstone-authority/run-authority-suite.sh
# -> per-domain PASS/FAIL table + __CAPSTONE_AUTHORITY_SUITE_PASSED__
```

Each domain gets its **own boot**: a capability fault inside a domain currently
aborts QEMU (a domain-mode fault hits a `riscv_cpu_do_interrupt` assertion in
the QEMU model), so a single trapping domain would otherwise kill the run for
all that follow. The diagnostic is emitted before the abort, so the trap is
still observed. Set `AUTHORITY_ONLY=name` to run one domain; `AUTHORITY_NO_BUILD=1`
to skip the rebuild.

## Oracle vocabulary

| expect | meaning | QEMU signal |
|--------|---------|-------------|
| `tag-fault` | deref of an untagged value | `Cap mem access requires capability`, no retval |
| `bounds-fault` | deref outside the capability bounds | `Cap mem access OOB`, no retval |
| `ok` | completes with the exact `retval` in `detail` | `retval = <decimal>`, no fault |
| `no-trap-today` | completes with no fault **today**; flips to `bounds-fault` after the Step-3 global SHRINK | `retval`, no fault |

## Tests and what each proves

| domain | expect | proves |
|--------|--------|--------|
| `forge_inttoptr` | tag-fault | an integer cast to a pointer is untagged; deref traps — integers cannot become authority (provenance, C2) |
| `ptr_int_ptr_roundtrip` | tag-fault | laundering a pointer through a (volatile) integer drops the tag; the reconstructed pointer faults — round-trips are fail-safe |
| `pointer_diff` | ok | pointer subtraction is a pure integer computation; no trap, no authority |
| `global_inbounds` | ok | ordinary in-bounds global access works (positive control) |
| `global_oob` | no-trap-today | **the granularity gap**: a[100] in `char a[64]` reaches an adjacent in-segment global and is **not** caught today; flips to a trap once `a`'s capability is SHRINK'd to its object (C1 before/after) |
| `global_oob_cross_segment` | bounds-fault | the coarse protection that already exists: an over-read past the *segment* traps today, because capabilities are segment-bounded |
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

3. **The Step-3 before/after is now a one-line oracle flip:** `global_oob` is
   recorded `no-trap-today`; after global-object SHRINK lands it should be
   changed to `bounds-fault` and must trap. `global_oob_cross_segment` must keep
   trapping. That is the measurable granularity result.

4. **A domain-mode capability fault aborts the QEMU model** (`riscv_cpu_do_interrupt`
   assertion `env->priv < PRV_C`). The fault is correctly detected and diagnosed
   first; graceful in-domain fault delivery is a separate runtime gap, noted here
   for the record.
