# Stale capability metadata does NOT leak through integer ops — REFUTED, 2026-08-08

**Verdict: the leakage hypothesis is dead.** Ordinary `lui`+`addi` over a register that
previously held a real capability **does** clear its metadata tag. This closes a suspected
R-14 root cause and kills a candidate explanation for the SQLite blocker.

## The result

`verif/tests/custom/capstone/cincoffset-stale-metadata.S`, Verilator, ~13 s:

    *** SUCCESS *** (tohost = 0) after 525 cycles

**The PASS is meaningful because the test carries its own positive control and the control
fired.** It first feeds `CINCOFFSET` a genuinely-tagged capability as rs2 — which
`capstone_flu_unit.anvil:30` must reject — and bails to `selfcheck_fail` if that does not
trap exactly once with `mcause == 25`. The trace shows **exactly one exception and one
`mret`, with mcause 25 (`0x19`)**. So the check can fire, and the trap/mcause plumbing
works. With that established, the real test — rebuild the capability, overwrite the register
with `lui`+`addi`, then use it as `CINCOFFSET`'s rs2 — did **not** trap.

Without that self-check this would have been another clean result from an instrument never
shown able to produce the opposite, which is the most expensive mistake on this project. The
test's author built the control in; running it is what made the negative worth anything.

## What it refutes

The chain reasoned out in the test's own header — `commit_stage.sv:279/325` gates the
metadata regfile's write-enable while the integer regfile writes unconditionally, and
`ariane_regfile_ff.sv` merely *holds* the old word when write-enable is low, so the tag
should survive — **does not produce an observable stale tag on this RTL.** Whatever the
mechanism, the tag is gone by the time `CINCOFFSET` reads rs2.

## What it cost, and why the reasoning was seductive

A real structural difference had been found behind the SQLite blocker:

| image | `__capstone_cap_init` | `cincoffset` | `cincoffsetimm` |
|---|---|---|---|
| `sqlite_silicon.dom` (wedges) | 1522 insns | **254** | 96 |
| `fdp0.dom` / fdreg model (runs clean) | 47 insns | **0** | 0 |

**The "zero" is only true INSIDE cap-init.** Whole-image, `fdp0.dom` contains **18**
`cincoffset` and 111 `cincoffsetimm`, and it *executes* them: `fdreg_len30`'s inner loop runs
`cincoffsetimm a1, s0, -0x34; lwu a1, 0x0(a1); cincoffset a0, a0, a1` — a capability register
redefined by an integer load and used as rs2 **one instruction later**, a tighter version of the
shape claimed to be fatal, thousands of times per boot, on this bitstream, returning every time.
"Immunity by absence" had no basis.

Combined with `capstone_flu_unit.anvil:29-34` raising `UNEXPECTED_OPERAND` when rs2 carries
capability metadata, and R-5's "illegal capability ops wedge rather than trap", this was a
coherent story tying the blocker to the same ungated-metadata path as R-18/R-19
(`issue_read_operands.sv:1140`).

**The structural difference is real. The proposed reason it would matter is not.** The
mechanism required an integer offset register carrying a stale tag, and that register is
clean.

## Scope — narrow, and stated so it is not over-read

The run exercises the **`lui`+`addi` producer only**, which is the R-14 codegen signature. A
register whose integer value arrives by a **load**, an **`add`**, or a **register copy** is
NOT covered. If the stale-tag idea is ever revived it must be revived for one of those
producers, with the same self-check discipline.

## Reusable invocation

Reconstructing this was itself a task; it is now recorded in the test header. ~13 s against
the prebuilt model at `work-ver/Variane_testharness` — no rebuild.

    cd capstone/capstone-ariane
    docker run --rm -v "$(pwd)":/workdir --user "$(id -u):$(id -g)" --entrypoint bash \
      -e HOME=/tmp -e RISCV=/toolchain -e CVA6_REPO_DIR=/workdir cva6-build-rv -c '
        cd /workdir; source verif/sim/setup-env.sh >/dev/null 2>&1; cd verif/sim
        python3 cva6.py --testlist=../tests/testlist_capstone.yaml \
          --test <TESTNAME> -o out_X --iss_yaml cva6.yaml \
          --target capstone_cv64a6_imafdc_sv39 --iss=veri-testharness \
          --issrun_opts=+debug_disable=1+UVM_VERBOSITY=UVM_NONE'

To pass a define to a test, add `--gcc_opts="-DFOO=1 "` — **the trailing space is required**,
the concatenation at `cva6.py:1252-1253` has no separator.

## Still open on SQLite

**CORRECTION to this note, same day.** The clamp series I cited here (`m1/m3/a0/a1/n1/n2`,
"`n1` returns, `n2` does not") is **SUPERSEDED** — `SILICON-BLOCKER.md:1218` marks it as
having run on a NON-STAGED image. Citing it here repeated a retracted result, which is
exactly the failure this file exists to prevent. The **live** localization is at
`SILICON-BLOCKER.md:3-22` (2026-08-06, control green, corrected classifier):
`sqlite3_config`, `sqlite3MutexInit`, `sqlite3MallocInit`, `sqlite3PCacheSetDefault` and
`sqlite3PcacheInitialize` all return distinct rc values, and **`sqlite3RegisterBuiltinFunctions()`
WEDGES**. Seven distinct values were delivered out of the domain, all of them AFTER cap-init.

**That also relocates the hypothesis this note refutes.** `__capstone_cap_init` is measured to
COMPLETE on this bitstream — `/tmp/capstone/mtv/sqpc.log` shows the `0x9E11`
`INTERP_RETURN_PRECALL` sentinel, which the glue emits *after* `RUN_CAP_INIT` and immediately
before `call domain_main`, with the control green in the same boot. So cap-init was never the
wedging phase and the 254-cincoffset argument was aimed at the wrong part of the program.

The two
walkers compile to near-identical code — same prologue, same `*144` indexing, same
`ldc a1, 0x70(a0)`, same `auipc/addi/jalr` call sequence, same `auipc` count. Two further
hypotheses died the same day: "SQLite uses `auipc` where fdreg does not" (both use two) and
"the call overruns the PCC/code window" (the target is inside the code region).

The cap-init descriptors were also checked and are **well-formed** — 179 records, no
non-power-of-2 alignments, no zero sizes, no init source outside the file, one 256 KiB record
which is the SQLite heap. A malformed descriptor is not the answer either.

The submodule commit carrying the test result is `57243ede4`; it **cannot be pushed** (the
RTL remote returns 403), which is why this note exists here.

---

## LEAD, 2026-08-08 (NOT confirmed): our patch removed `static` from the array

Found while looking for what differs between the fdreg model that runs and the SQLite image
that wedges. Recorded as a lead, not a root cause — the copy behaviour below is **inferred
from the language semantics and has not been observed in the disassembly or on hardware.**

    upstream sqlite3.c :  static FuncDef aBuiltinFunc[] = {
    our sqlite3-capstone.c (:137228) :   FuncDef capstoneBuiltinFunc[] = {

`static` is gone, and it is gone inside `sqlite3RegisterBuiltinFunctions()` — the function the
**live** localization (`SILICON-BLOCKER.md:3-22`) says wedges, with every earlier step of
`sqlite3_initialize()` returning a distinct rc.

**Why this could matter.** A non-`static` local array with an initializer is re-initialised on
every call. LLVM gave the storage a global home — `sqlite3RegisterBuiltinFunctions.aBuiltinFunc`,
**9216 bytes** (64 × 144) in `.data` at `0x15d010`, confirmed in the symbol table — but the
semantics still require the initialiser to be re-applied per call, i.e. a template copy. The
elements are `FuncDef`, whose `zName` and function-pointer fields are **capabilities** in a
pure-capability domain. A byte-wise copy destroys capability tags; the walker then does
`ldc a1, 0x70(a0)` on an untagged word.

This is also the open item recorded elsewhere as "initialized-global template copy on silicon".

**Why fdreg does not reproduce it:** `fdreg_defs` IS `static` (`fdreg_kernel.h:351`), so there is
no per-call re-initialisation, and fdreg returns.

### The minimal out-of-SQLite reproducer this implies

Take the existing 13 KB fdreg rung and **remove `static` from `fdreg_defs`** — one keyword, no
other change. Everything else (struct layout, string contents, the `zName` read, the
`fdreg_len30` call) already matches SQLite. If the non-static build wedges where the static
build returns, that is the repro, and the blocker is a **software** defect in our patch rather
than silicon.

Cheap and staged, in this order:
1. QEMU first — free, and it may already diverge.
2. Verilator, if a directed shape can be built (~13 s).
3. One board boot, batched: `static` and non-`static` at distinct link addresses, control first,
   both expected to RETURN so the run yields data either way.

**What is NOT established:** that a template copy is actually emitted; that it is byte-wise; that
tags are lost. All three are checkable in the disassembly of
`sqlite3RegisterBuiltinFunctions` before spending any board time. Note the symbol is 1036 bytes
but a naive disassembly extraction yielded only 38 instructions, so read the full range by
address rather than trusting a symbol-boundary scrape.

**Provenance settled meanwhile:** the resident bitstream comes from the RTL branch
`fpga-testing` (project lead, 2026-08-08). Both candidate heads there — local `458982093` and
`origin/458982093`'s successor `e1b3db6ba` — carry `reg head : logic[16]` (65536 nodes) and an
ACTIVE `CINCOFFSET` rs2 check, so the corrections above hold regardless of which was synthesised.
`fpga-testing-old-anvil` (`8d10c1e8f`) is the `logic[10]` + commented-check revision that the
stale `7aac52f93` assumption came from.
