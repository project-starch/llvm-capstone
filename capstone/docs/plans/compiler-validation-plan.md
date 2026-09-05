# Compiler validation and hardening plan

**Status: PROPOSED 2026-09-04 — for the project lead's review; nothing here is executed yet.**
Built from three read-only inventories (backend, frontend/ABI, tests + defects) and three design
passes (static coverage; execution oracles and fuzzing; semantics and ABI audit). Every
high-consequence claim was re-verified on the built compiler before it entered — marked ✔ below —
and each check records what it could not distinguish. Two decisions are the lead's (ABI direction,
Tier 4.1; board stacking, Tier 3) and Tier 0.0 depends on the session that holds `capstone-qemu`.

## Context

The Capstone LLVM backend (`llvm/lib/Target/Capstone/`, a 184-file copy of the RISCV target with
~5,000 divergent lines) and clang target have been validated so far by 60 lit tests, 15 clang
tests, and a nightly of runtime suites that build almost everything at `-O0`. The c128 carrier
type landed on 2026-09-04 and silently reverted four things on the way in. The project lead has
asked for a **reliable and correct** compiler with **complete** test coverage, frontend and
backend, using as many reviewers/auditors as necessary.

This plan is built on three read-only inventories (backend, frontend/ABI, tests+defects) whose
highest-consequence claims were then **verified against the built compiler** — marked ✔ below.

## Verified findings that shape the plan

| # | finding | evidence |
|---|---|---|
| 1 | ✔ `CapstoneABIInfo` is **dead code**; capstone64 ships on `DefaultABIInfo` (every aggregate byval/sret) | `createTargetCodeGenInfo` has no capstone case; 0 refs in `CodeGenModule.cpp` |
| 2 | ✔ `uintptr_t` is 64-bit while `sizeof(void*)==16`; `(void*)(uintptr_t)p` → `ptrtoint to i64` — tag lost. musl `__scc` (`syscall_arch.h:41`) assumes the opposite | `clang -dM -E`; IR |
| 3 | ✔ **No Sema for any `__builtin_capstone_*`**: `cap_tighten(p, 999)` = `report_fatal_error` + stack dump from C | `SemaChecking.cpp:2050-2110`; ran it |
| 4 | ✔ `cap_get_cursor` selects to `mv` (explorer concern refuted); ✔ all 10 icmp codes on capability `select` compile (the `llvm_unreachable` at `ISelLowering.cpp:10832` is not reached from IR) | ran llc |
| 5 | ✔ `-capstone-shrink-stack` and `-shrink-globals` both default **true**; production silicon builds pass `false` for both | `ISelDAGToDAG.cpp:47,66-70`; `build-sqlite-silicon.sh:2545` |
| 6 | ✔ byval `{void*,long}` copy is tag-preserving (`ldc`/`stc`) at -O2 — but no lit test covers byval/sret capability structs | ran clang |
| 7 | ✔ built `clang`/`llc` are current with HEAD; ✔ `llvm-mc`/`llvm-objdump` binaries predate c128 (Aug 18) | mtimes |
| 8 | **Zero `-O1` lit tests; 1 of 16 nightly rows at `-O2`; the only value oracle (`sqlite-slt`) runs at `-O0`.** C-3, C-17, S-13 are all `-O1`-specific | explorer 3 |
| 9 | **No `llvm/test/MC/Capstone/`** — assembler and disassembler untested; the board-run skill's mandatory pre-boot check depends on `llvm-objdump` | explorer 3 |
| 10 | Five lit tests carry CHECK-NOTs that cannot fail (C-26 class): `ptr-diff-signed.ll`, `cap-i128-ptr-diff-const.ll:15`, `cap-i128-or-undef.ll:16`, `static-cap-global-init-large.ll:103`, `aggregate-memcpy-align.ll` | explorer 3 |
| 11 | 10 of 23 instructions are smoke-only (one bare CHECK in `intrinsics.ll` / `cap-control-flow.ll`); 10 of 19 builtins are used by no C anywhere | explorers 2, 3 |
| 12 | **Four unfiled backend bugs** documented only in `build-coremark-capstone.sh`: tail-call lowering emits `cjalr ra` not `cjalr zero`; `rd!=rs1` LINEAR sink; mixed scalar/cap lowering; list-path copy trap. And `-O0` *causes* a tag-clearing granule-share bug | explorer 3 |
| 13 | **R-21**: `cincoffset`/`scc`/`tighten` do not consume a LINEAR source and `init` duplicates it (spec violation) — the compiler's linearity model (`InstrInfo.td:2591-2599`) may rest on wrong semantics | ISSUES.md:4369 |
| 14 | 40 residual `i128` sites in `ISelLowering.cpp` post-c128; a live `ResultVT==i128` branch in `selectLDC_STC` (`DAGToDAG.cpp:1285`); a user-facing diagnostic string that is now false (`:8470`) | explorer 1 |
| 15 | ~12 patches to shared LLVM (TableGen, ValueTypes, SelectionDAG, ValueTracking, AsmPrinter) guarded only by `lit-generic` | explorer 1 |
| 16 | No random-program oracle on the machine; native-x86 differential already practised (BEEBS host, SLT native) | checked |
| 17 | ✔ **A QEMU boot+login+run is 5–15 s**, not minutes (smoke 5 s; CoreMark 11 s; RV8 73 s for 7; BEEBS 1144 s for 82; authority 1087 s for 40 one-boot-each domains). Minutes-per-boot is the FPGA JTAG cost. Every `-O2` twin of an existing suite is therefore cheap and needs no new infrastructure | nightly `20260904_160628` durations |
| 18 | ✔ **C-2's build blocker is gone**: RV8 `qsort` and `miniz` compile at -O1 and -O2 on HEAD (rc=0) — the c128 split removed the i128 mixed-extend family. C-3 (runtime failure at -O1/-O2, 5 benches, measured 2026-07-28) is the open RV8 question and must be re-measured. `tests/compiler-repros/C2[123]*` READMEs are stale | compiled both |
| 19 | ✔ **`__builtin_ctz` crashes the compiler at -O0 AND -O2** (`LegalizeDAG` assertion `Res.getValueType() == Node->getValueType(0)`). C-20/C-24 are OPEN — their commits are repro packages, not fixes; no lit test mentions cttz | ran clang |
| 20 | ✔ `llvm-stress` and `llvm-reduce` **are built in-tree** (`cmake-build-debug/bin`, dynamically linked against the current `libLLVMCapstoneCodeGen.so`); `capstone/tests/reduce.sh` already wraps `llvm-reduce` | ls |
| 21 | SQLite compiles and links at -O1 (`build-sqlite-silicon.sh:2691-2701`); the -O0 default is "not validated end to end", a decision awaiting exactly the validation this plan does | read |
| 22 | A capability fault inside a domain aborts QEMU (`run-authority-suite.py`), so a multi-domain boot needs resume-after-fault; `hostcall.c` is "written but unexercised" — fuzz output must go through the 32-bit `*res` retval channel | agent B, to confirm at implementation |
| 23 | ✔ **Tail calls are miscompiled at -O1+**: `long g(long x){return f(x+1);}` emits `cjalr ra, 0(a1)` and then *nothing* — no epilogue, no return; control falls off the end. Cause: `ISelDAGToDAG.cpp:4029-4030` routes `CapstoneISD::TAIL` into `selectCall`, which always builds `PseudoCALLIndirect`; `PseudoTAILIndirect` (→ `cjalr zero`) is never selected. Masked since June by `-fno-optimize-sibling-calls` in CoreMark and SQLite. No ID, no test | ran clang; A, C |
| 24 | ✔ **The S-12 workaround pass is inert on c128 code**: it matches `Capstone::X0` as the MOVC source (`S12MovcLdcHazard.cpp:120`) but a null-capability move is `MOVC $c0`, and its rename candidates are `X16..X31` (`:169`), which the verifier rejects on a GPCR operand | grep; A |
| 25 | ✔ **`call a0, a1` does not assemble** ("invalid operand") — `CAP_CALL`'s mnemonic collides with `PseudoCALL`'s `call $func`; the disassembler prints exactly that text, so one of 23 MC round-trips is broken by construction | ran llvm-mc; A |
| 26 | ✔ **The c128 QEMU merge (`b6e65e99`) broke the QEMU build** — `op_helper.c` brace depth 0 at the pre-merge parent `cb23bf201b`, 2 at the merge — and the binary in use was dated 2026-08-27. **Fixed the same day** by the docs/board session (`capstone-qemu f5972c364f`, three defects, via superproject `db079043`): builds, `run-smoke.sh` passes, the SLT negative control fires on the rebuilt binary. The third defect — an unconditional `UNEXP_OP_TYPE` raise in three helpers — was real (cause 24 immediately, once the first two were fixed). **Still open:** what the 08-27 binary was built from (the reflog puts HEAD at `cb23bf201b` at build time, which already carries those raise sites, yet that binary ran SQLite — so its pedigree is unknown, not merely old); and the nightly's relink branch (`run-nightly.sh:192-200`) should have fired and stayed silent. **Withdrawn:** "SLT matches native 15/15" — there is no committed per-record harness; `run-sqlite-slt.sh` is a liveness check and the comparison was ad hoc (Q-02) | awk; stat; C; `db079043` |
| 27 | ✔ **R-21 is stale in the direction that matters**: `capstone_flu_unit.anvil:45-46,76` now null a LINEAR `rs1` on `cincoffset`/`cincoffsetimm` (and `:109-112` on `scc`), and commit `2035df882` is an ancestor of the flashed `5097eb166` bitstream. Spec, QEMU and RTL all **consume**; the compiler's pure-use `CIncOffset` (`InstrInfo.td:2442-2450`) is the odd one out. Still non-conformant in RTL: TIGHTEN passes rs1 through, INIT duplicates it | grep; merge-base; C |
| 28 | ✔ **The default global ABI is QEMU-only by construction**: the glue `delin`s the entry capabilities and `PseudoCapGlobalBase` `delin`s every gp-derived base; RTL raises `UNEXPECTED_CAP_TYPE` on `delin` of a NONLIN cap (`capstone_dyn_unit.anvil`), QEMU silently `return`s (`op_helper.c:1179`), and QEMU re-mints gp on every `cjalr`. Only `-capstone-gp-captable` (`ldc gp[i]`, no `delin`) is silicon-real — which matches the project's own silicon history | grep; C |
| 29 | `cap_get_tag` lowers to `lcc` selector 0 (`ISelDAGToDAG.cpp:3303`), which per C's RTL/QEMU reading traps on an untagged operand — the builtin traps on the value it exists to test. `CAPENTER` funct7 in LLVM (`0b0100010`) differs from the decoder's, and `CAPEXIT` exists in neither spec nor RTL; `CAP_RETURN`'s operand roles are inverted vs spec/RTL | grep; C (semantics to be confirmed by `rtl-oracle` in Tier 4) |
| 30 | `lowerADD`, `lowerSUB`, `lowerScalarI128{Shift,Logical,And,Mul}` in `ISelLowering.cpp:7989-8693` have **no call sites** and `LowerOperation` has no ADD/SUB/SHL/MUL cases — ~35 of 43 `MVT::i128` sites are dead by construction; the "128-bit `_Atomic` reaches an unlowerable shift" comment is stale (all forms become `__atomic_*_16` libcalls at every -O) | grep; A, C |

## Decisions already taken by the project lead

- Install **csmith and yarpgen** under `$HOME` as random-program oracles.
- **QEMU and board freely** (QEMU serialized on the rootfs lock; board serialized across lanes; reflash ask-first).
- **Validate first, then retire proven compiler-debt** — every workaround classified with evidence, classification reported before removal.
- **`-O2` correct on QEMU and silicon is the bar**: execution tests run at `-O0` and `-O2` and must agree.

## Definition of COMPLETE (the plan is done when every cell is filled)

a. every `.td` capability instruction in a positive CHECK with a paired negative control;
b. every intrinsic: ISel + lit + QEMU execution test;
c. every Capstone-only custom lowering / `report_fatal_error` / `llvm_unreachable`: a test that reaches it, or a proof it is unreachable;
d. every `cl::opt` tested at both values;
e. every production flag set (A–J) exercised at each `-O` level in use; execution suites at `-O0` **and** `-O2`;
f. MC round-trip for every instruction;
g. every OPEN C-nn has a regression test or a recorded reason it cannot;
h. every workaround classified compiler-debt vs silicon-debt with evidence.

## Tier 0 — make the instruments trustworthy (before any claim is made)

Nothing in later tiers is evidence until these are done; each has already produced a wrong
verdict on this project when skipped.

0. **The QEMU oracle — finding 26.** The c128 QEMU merge broke the build and no gate caught it;
   the docs/board session fixed it the same day (`capstone-qemu f5972c364f`, three defects — the
   third an unconditional `UNEXP_OP_TYPE` raise that was confirmed real by running it). Rebuilt
   binary: `run-smoke.sh` passes, `slt/check-negative-control.sh` fires. What is **still open**,
   and stays a hand-off to that session: (c) the pedigree of the 08-27 binary — the reflog puts
   HEAD at `cb23bf201b` when it was built, which already carries the raise sites, yet it ran
   SQLite to completion; until that is explained, every result taken on it has *unknown*
   pedigree; and (e) the nightly's relink branch (`run-nightly.sh:192-200`) should have fired on
   09-04 and stayed silent — a gate that should fire and does not outranks the breakage it missed.
   **New prerequisite for Tier 2 (from Q-02):** "SLT matches native 15/15" is **withdrawn** —
   `run-sqlite-slt.sh` is a liveness check and the per-record comparison against `slt_native` was
   never committed. So before `sqlite-slt` is used as a value oracle anywhere in this plan, Tier
   2a's first deliverable is a committed harness (`capstone/tests/twins/slt-compare.sh`) that
   compares each record's result against the native run, exits non-zero on any mismatch, and is
   positive-controlled by `slt/negative-control.test`. Every QEMU result line names the binary it
   ran on. Static tiers and rtl-sim tests never waited on any of this.
1. **Rebuild the MC tools.** `llvm-mc` and `llvm-objdump` in `llvm/cmake-build-debug/bin` are
   dated 2026-08-18 — before the c128 register-class split (`4bef5b212152`) touched both the
   AsmParser and the Disassembler. The `board-run` skill's mandatory pre-boot check disassembles
   with this binary. `ninja -C llvm/cmake-build-debug -j90 llvm-mc llvm-objdump llvm-readobj`.
2. **Confirm `llvm-stress` and `llvm-reduce` are current.** Both exist in
   `cmake-build-debug/bin` (built Aug 18, dynamically linked against the current
   `libLLVMCapstoneCodeGen.so`, so they track the backend); `ninja -j90 llvm-stress llvm-reduce`
   is the no-op check. Never use the copies under `~/dev/llvm-project` — same version, different
   tree, upstream semantics.
3. **Record a green baseline this plan owns.** `llvm-lit -sv llvm/test/CodeGen/Capstone
   clang/test/CodeGen/*capstone* clang/test/Sema/capstone*` and `capstone/tests/run-nightly.sh
   --skip-build` (serialized on the rootfs lock; ~6 h). The last nightly is 16 PASS / 1 FAIL,
   the FAIL being the known Q-01 residual (`sqlite-memory`, never had a passing baseline). Any
   later red row is attributed against this run, not against memory.
4. **Verify the built compiler matches the sources before every claim** — the mtime check that
   was needed today (`82213a4a` committed 92 s after the binary; the binary was fine, but only
   the `.o`-vs-source mtimes proved it). One line in the pre-commit tier: refuse to run lit if
   any `llvm/lib/Target/Capstone/*.{cpp,h,td}` is newer than `bin/llc`.

## Tier 1 — static coverage: lit + MC (≈ 66 h)

### 1.0 Infrastructure first
- `llvm/test/CodeGen/Capstone/lit.local.cfg` (model: `llvm/test/MC/RISCV/lit.local.cfg`): gate
  on the Capstone target and add the substitution `%llc-cap` = `llc -mtriple=capstone64
  -mattr=+m -verify-machineinstrs`. Same file in `llvm/test/MC/Capstone/` and
  `llvm/test/MC/Disassembler/Capstone/`.
- **-O1 arms via two mechanical RUN lines per file** (`%llc-cap -O0/-O1 < %s -o /dev/null`),
  not a separate directory: all 58 non-`not` inputs already pass the -O1 verifier today, so the
  arms cost nothing and catch the "Cannot select"/verifier class that every -O1 failure on this
  target has been. Targeted -O1 CHECKs only where the shape is -O1-specific. `-Os` coverage
  comes from clang RUN lines (llc has no `-Os`).
- New tests omit `target datalayout` (llc derives it from the triple).
- `report_fatal_error` → `not --crash llc`; `diagnose()` → `not llc`; through clang → `not
  %clang_cc1` matching "fatal error: error in backend".

### 1.1 New tests, by consequence (file → construct → positive CHECK / negative control → closes)

| ID | file | what it pins | closes |
|---|---|---|---|
| T1 | `tail-call.ll` (**XFAIL**), `clang/…/capstone-tail-call.c` (**XFAIL**) | after callee-save restore: `cincoffsetimm sp` then `CHECK-NEXT: cjalr zero, 0(a…)`; `CHECK-NOT: cjalr ra`; control `@not_tail` must show `cjalr ra` then `cjalr zero, 0(ra)`; gp-free arm `jalr zero`; `-Os` and `-fno-optimize-sibling-calls` arms | finding 23 (C-28) |
| T2 | `intrinsics.ll` rewrite (model: `cap-i128-and-capability-mask.ll`) | every intrinsic body pinned `# %bb.0:` / `CHECK-NEXT: <insn>` / `CHECK-NEXT: cjalr zero, 0(ra)`; tighten at 0/7/31; ccsrrw named + numeric CSR; `--implicit-check-not=lcc {{.*}}, 2` | (a),(b) for 7 smoke-only instructions |
| T3 | `intrinsics-unused-result.ll` | ASM: `drop`/`revoke`/`mrev`/`delin`/`call`/`capenter`/`ccsrrw` survive with result unused; `seal`/`tighten`/`scc`/`init`/`shrink` are DCE'd (`CHECK-NOT`); IR arm can't go red until intrinsics carry `IntrWillReturn` — record as a `TODO` for Tier 4 | hasSideEffects modelling |
| T4 | `s12-movc-ldc-rename.mir` (**XFAIL**) + `s12-movc-ldc-nofire.mir` | `-run-pass=capstone-s12-movc-ldc` ON/OFF/`-s12-window=1`; ON must rename `$c10 = MOVC $c0` → `$c16`; no-free-reg / redefinition / call-between arms unchanged | finding 24; 2 cl::opts; (h) |
| T5 | `cap-control-flow.ll` strengthen | `CHECK-NEXT` pinning for call/capenter/return/capexit; control `@plain_ret` ending in `cjalr zero` so the existing `CHECK-NOT: cjalr` is honest | 4 smoke-only instructions |
| T6 | `cap-byval-sret.ll`; `clang/…/capstone-abi-cap-struct.c` | byval `{ptr as200, i64}` copied with `ldc`/`stc` + `ld`/`sd`; sret `stc a1, 0(a0)`; `va_arg` of a by-ref struct loads `ptr addrspace(200)` (`CHECK-NOT: load ptr,`); `-O0/-O2/-Os` | finding 1 pinned as shipping behaviour; vararg AS0 fix (`beab9348`) |
| T8 | `flags-memcpy-fixup.ll`, `flags-memops-libcall.ll`, `flags-merge-string-constants.ll`, `flags-cap-init-limit.ll` (model: `ldc-retry.ll`'s arm structure) | each cl::opt at both values with the OFF arm as the control | (d): all 17 flags |
| T9 | `select-cap-condcodes.ll` | all 10 CCs on a capability select at -O1/-O2, exact branch opcode per CC; `--implicit-check-not=addi a0, a1, 0` | (c) `:10832` proven unreachable |
| T10 | `fatal-tighten-range.ll`, `fatal-ccsrrw-range.ll` (`not --crash`), `clang/…/capstone-tighten-range-diagnostic.c`, `unreachable-fatal-routes.ll` | the two reachable `report_fatal_error`s pinned; `:1360`/`:1765`/`:1913` recorded unreachable with the pinning test (immarg verifier / IR truncation) | (c) |
| T12 | `c14-copy-class-postra.mir` | `-run-pass=postrapseudos`: `$c9 = COPY $c10` → `MOVC`; `$x18 = COPY $x11` → `ADDI`; `CHECK-NOT: MOVC $x`. **Do not restore** the deleted `c14-livein-not-scalar.mir` — it tests a rule `1238c3aa` removed | C-14 |
| T13 | `c6-negative-offset-sink.ll` | CGP-sunk `gep i8, p, -8` → `lw a0, -8(a0)` in both blocks; `--implicit-check-not=lui` | C-6 |
| T14 | `clang/…/capstone-atomic-128.c` | `_Atomic __int128` load/store/RMW at -O0/-O1/-O2 → `__atomic_*_16` libcalls, no diagnostic, no `lr.d` | finding 30 |
| T15 | `o1-shapes.ll` + `c17-wide-constant-arm.ll` (`not llc`) | C-2 shape → two i64 `or`, no libcall; C-22 → `addi a1,-1; ori a0,a1,7`; C-23 both halves stored; C-17 → the clean "Cannot materialize" diagnostic | (g): C-2, C-17, C-21..C-23 |
| T16 | `c20-cttz.ll` (**XFAIL**) | `llvm.cttz.i32/i64` at -O0/-O2, `ctlz` control | finding 19 (C-20) |
| T17 | `opt-level-flag-sweep.ll` | {default, gp-captable, gp-free} × {-O0,-O1,-O2}, verifier only | (e) |
| T18 | `gp-table-linear-delin.ll` | exactly one `delin` per gp-derived base — **reshaped by the Tier 4 DELIN decision** | C-27 |
| T20 | LICM test: add a `nonnull` positive control that IS hoisted | proves the C-19 `CHECK-NOT` can fail | C-19 |
| T21 | `clang/…/builtins-capstone.c` → all 19 builtins, `--implicit-check-not=inttoptr`; delete the byte-duplicate `capstone-builtins.c` | (b) clang half |

### 1.2 The six vacuous tests — exact fix, and the mutation that proves the check fires
1. `ptr-diff-signed.ll` (C-26): keep the i64 functions, make the RUN `--implicit-check-not=__divti3
   --implicit-check-not=__moddi3`, add control `@sdiv_nonexact_var` that MUST emit `div`.
   Mutation: `sdiv exact` → `sdiv` on a 12-byte element → `mul`-by-inverse becomes `div`, red.
2. `cap-i128-ptr-diff-const.ll:15`: add `--implicit-check-not=__muloti4`, pin `ptr_diff_q1`
   `CHECK-NEXT` from `ldc` to `srai`. Mutation: 48-byte struct → `srai` becomes `mul`, red.
3. `cap-i128-or-undef.ll:16`: add the widening-mul shape that actually produces `__muloti4`
   (`mul i128` of two `zext`s, truncated) + `--implicit-check-not=__multi3`. Mutation: drop `+m`.
4. `static-cap-global-init-large.ll:103`: `ASM-COUNT-65: stc` between the label and `cjalr`
   (the header's own measured number). Mutation: one initializer element → `i64 1`, an `sd`
   appears, red.
5. `aggregate-memcpy-align.ll`: `CHECK-COUNT-4: lw`/`sw` (16 B), `COUNT-11` (44 B),
   `copy_align16` pinned `ldc`→`stc`→`cjalr`. Mutation: `align 4` → `align 16`, `ldc` appears.
6. `cap-mrev-delin-side-effects.ll`: add `-mtriple=capstone64` to the `opt` RUN; record that the
   IR arm cannot discriminate until `IntrinsicsCapstone.td` uses `DefaultAttrsIntrinsic`.

**Every new negative check carries a `; MUTATION: <change> -> <check that fires>` header line,
performed once at write time.** The coverage script (1.4) refuses a file with a CHECK-NOT and no
MUTATION line.

### 1.3 MC suite — `llvm/test/MC/Capstone/`, `MC/Disassembler/Capstone/` (model: `MC/RISCV/rv64i-valid.s`, `-invalid.s`, `insn.s`)
- M1 `cap-valid.s`: all 23 instructions, every operand form, immediates at range edges, both
  register spellings; encodings **re-derived from the rebuilt `llvm-mc`** (Tier 0.1).
- M2 `cap-invalid.s`: out-of-range immediates, wrong arity, `mrev zero` (GPCRNoC0), FP register
  as capability, `%lo` where an immediate is required — messages anchored `:[[@LINE]]:N:`.
- M3 `cap-regnames.s`: pin whether `c10` is accepted and that capabilities print as `a0`.
- M4 `cap-call-mnemonic.s` (**XFAIL**, finding 25) + `cap-call-symbol.s` (control: `call foo`
  still produces `auipc`/`jalr` + `R_Capstone_CALL_PLT`).
- M5 `Disassembler/…/cap-instructions.txt`: every M1 encoding decodes; fixed-field violations;
  `mrev` with rd=0 → invalid; unassigned funct7 under `OPC_CAP_OP` → invalid.
- M6 `pseudo-expansion-roundtrip.ll`: `PseudoTRUNC_CAP` and `PseudoCapGlobalBase` through
  llc → obj → objdump.
- M7 `obj-relocs-{cap-init,gp-table,cap-constant}.ll`: `llvm-readobj -r` on `.rela.capstone_cap_init`
  (ADD64/SUB64 pairs), `.rela.capstone_gp_table`, `.rela.capstone_gp_initdesc`
  (references the undefined `__gpfree_globals_base` — pin it), `.rela.rodata` `R_Capstone_64`;
  add `| llvm-objdump -d -r -` arms to the four existing `-filetype=obj` tests.
- M8 `cap-insn-split.s`: the glue's hand-encoded SPLIT assembles via `.insn` and disassembles as
  `<unknown>` (pins that no accidental decode exists).

### 1.4 Coverage accounting — `capstone/tests/check-lit-coverage.sh` (definition-of-complete as a gate)
Style of `precommit-scan.sh`; exit 0 complete / 1 gaps / 2 error; `--only
instr|intrinsics|flags|fatal|olevels|cnn|mc|mutations`. Sections: (1) every `.td` mnemonic has
a positive CHECK, a negative control, an M1 line and an M5 line; (2) every intrinsic has a
`call @llvm.capstone.cap.*` in CodeGen tests and a `__builtin_capstone_cap_*` in clang tests;
(3) each of the 17 Capstone `cl::opt`s has RUN lines at both values, and any `"capstone-…"`
string in the backend not on the list fails ("new flag without coverage"); (4) every
`report_fatal_error("Capstone…`/`diagnose(` message appears in a CHECK or in
`lit-coverage-unreachable.txt` with file:line + reason + pinning test; (5) every test has -O0
and -O1 RUN lines or is in the exempt list with a reason; (6) every OPEN C-nn maps to a test or
a reason in `lit-coverage-cnn.txt`; (7) MC dirs cover every mnemonic; (8) every file with a
negative check has a `MUTATION:` line. **Positive control: on the current tree it must report**
11 instructions without negative controls, 23 without MC lines, 11 of 17 flags with no RUN line,
all fatal sites uncovered, 58/58 missing -O1, C-nn all unmapped — and `--self-test` runs it
against a stub tree and requires exit 1.

### 1.5 Naming
Rename the 4 tests whose IR is now all-i64 (`cap-i128-and-capability-mask` → `cap-addr-bitmask`,
`cap-i128-or-undef` → `cap-ptrdiff-nonpow2-div`, `cap-i128-ptr-diff-const` →
`cap-ptrdiff-const-offset`, `cap-i128-select-capability` → `select-cap-vs-scalar`); keep the
other 7 `i128`-named files — they are genuine and now the only pins on the legalizer's i128
expansion. Delete `capstone-builtins.c`; fold `capstone-i128-backend-diagnostic.c` into T14.

### 1.6 Order
P0 (~14 h, exposes the bugs first): lit.local.cfg ×3, coverage script + positive control, T1
XFAIL, T4 XFAIL, MC skeleton + M4 XFAIL; rebuild MC tools in parallel. P1 (~14 h): T2, T5, T3,
T9, T10, the six vacuous fixes with mutations, T16. P2 (~10 h): M1–M8 on rebuilt binaries.
P3 (~17 h): T6, T14, T13, T12, T18, T15, the 58 mechanical -O arms, T17. P4 (~11 h): T8 flag
tests, renames, coverage script green, nightly rows `lit-mc` and `lit-coverage`. Declared XFAILs
at the end of Tier 1: 5 — each names its C-nn and must reach 0 as Tier 5 fixes land.

## Tier 2 — execution oracles (miscompiles that lit cannot see)

QEMU is cheap (finding 17), so this tier is ordered by *information*, not cost. Every oracle is
positive-controlled before its first verdict is believed.

### 2a. `-O0` vs `-O2` twins of every existing suite — day 1, no new infrastructure

The runners already honour an env var; only CoreMark needs a script edit.

| suite | `-O2` invocation | oracle | expected today |
|---|---|---|---|
| RV8 ×7 | `DOMAIN_OPT_LEVEL=-O2 OUT_DIR=… LOG_DIR=… bash capstone/benchmarks/rv8/run-all-rv8.sh` (and `-O1`), 3× each | marker | C-3: 5 failed 10/10 at -O1 on 2026-07-28 — **first retire-or-file item** (~2 min) |
| BEEBS ×82 | `DOMAIN_OPT_LEVEL=-O2 RUN_ALL_BEEBS_LOG_DIR=… bash capstone/benchmarks/beebs/run-all-beebs.sh` | `0xC171C0DE` marker | unknown; `dijkstra/edn/rijndael` carry source adaptations for **-O0** backend crashes — retest unadapted |
| CoreMark | add knobs to `build-coremark-capstone.sh`: `COREMARK_UNIFORM_OPT=1`, per-object `COREMARK_OPT_<obj>`, `COREMARK_SIBLING_CALLS=1`, `COREMARK_JUMP_TABLES=1`, `COREMARK_UPSTREAM_CRCU8=1` (defaults = today's pins); then `DOMAIN_OPT_LEVEL=-O2 COREMARK_UNIFORM_OPT=1 … bash capstone/tests/runtime-qemu/run-coremark.sh` | CRC (`Correct operation validated`) | 8 single-knob variants, ~12 s each — each pin relaxed alone isolates one of the four unfiled bugs |
| SQLite SLT | **first** write `capstone/tests/twins/slt-compare.sh` — per-record comparison of the domain's SLT output against `slt_native`, non-zero on any mismatch, positive-controlled by `slt/negative-control.test` (no such harness is committed today; Tier 0.0). Then `SQLITE_OPT_LEVEL=-O1`, `-O2` with `run-sqlite-slt.sh`; also `SLT_TEST=` the real-join files (`q_two`, `dd2_join`, `qj2`) | per-record vs `slt_native` **via the new harness** | -O1 never validated (finding 21); C-17 latent; the 15/15 claim is withdrawn until the harness re-establishes it |
| authority | `bash capstone/tests/capstone-authority/run-authority-opt-matrix.sh` (O1/O2/O3; 8 exclusions in `opt-policy.tsv`; one boot per domain because a fault aborts QEMU, ~18 min/level) | `oracle.tsv` | not in the nightly today |
| probe suites | most build through `build-domain.sh`, whose default is **-O2** already — get the `-O0` twin with `DOMAIN_OPT_LEVEL=-O0`; enumerate with `grep -l 'OPT_LEVELS\|DOMAIN_OPT_LEVEL' capstone/tests/runtime-qemu/run-*.sh` | per-probe markers | — |

**Agreement gate** — new `capstone/tests/twins/compare-twins.sh`: joins the per-benchmark result
files of the two directories into `capstone/tests/twins/results/<date>.tsv` with verdicts
`AGREE-PASS | O2-ONLY-FAIL | O0-ONLY-FAIL | BOTH-FAIL | FLAKE | BUILD-FAIL(level)`; exits
non-zero on any non-AGREE row. This is definition (e).

**Triage ladder for a disagreement** (each rung = rebuild + ~5 s boot): 3× rerun → lowest failing
level (-O1, -Os) → `-mllvm -opt-bisect-limit=N` binary search (~15 rebuilds) → per-TU/function
`optnone` bisection → backend `cl::opt` bisection (`-capstone-enable-{copyelim,copy-propagation,
dead-defs,sink-fold,machine-combiner}=false`, `-capstone-use-aa=false`, `-capstone-shrink-*`,
`-capstone-lower-memops-via-libcall=true`, `-enable-misched=false`) → extract the guilty
function's post-middle-end IR → `capstone/tests/reduce.sh` (crash) or an execution
interestingness script (miscompile) → `capstone/tests/compiler-repros/<ID>/` and a lit test.

### 2b. Workaround ON/OFF differential — day 2; definition (h)

QEMU has no silicon defects, so **any knob whose removal changes QEMU output is compiler-debt by
construction**; identical QEMU output makes it a silicon-debt candidate settled on the board.
Each experiment is a matched pair on the fixed corpus (BEEBS + RV8 + CoreMark + SLT, at -O0 and
-O2) and **must assert the two `.dom` sha256s differ** — otherwise it tested nothing.

| knob | expected class | note |
|---|---|---|
| `-capstone-s12-movc-ldc-workaround` | silicon-debt (RTL fixed) | retire after 2 more clean board draws |
| `-capstone-retry-untagged-ldc`, `-double-ldc` | instrument | keep, never default |
| `-capstone-memcpy-high-half-fixup` family, `BEEBS_LDC_HIGH_HALF_FIXUP` | silicon-debt, **S-06 fixed** — deletion already authorised by `S06-WORKAROUNDS-TO-REVERT.md` §3 | one board rung (`s06agg`=15) re-confirms on the current bitstream |
| `-capstone-lower-memops-via-libcall` | not a workaround | any difference = a bug in one of the two memop lowerings |
| `-capstone-shrink-globals/-stack` ON vs OFF | silicon-debt if OFF is needed only on the board; compiler-debt if ON faults under QEMU | authority `stack_*`/`global_*` are the bounds oracle |
| `-capstone-merge-string-constants` | optimisation choice | difference = compiler-debt |
| `BEEBS_MEMCPY_OPTNONE` (S-04) | silicon-debt, board-classified by the `sm0`/`sm` pair on 2026-08-10 | refresh on the current bitstream |
| `BEEBS_STRING_WRITERS_OPTNONE` | silicon-debt (stage 167: optnone clears bit 0 of a 7-byte memmove) | "neutral" was workload-level only |
| `BEEBS_STRING_LINEAR_SAFE` | **depends on R-21** — QEMU models linear consumption by `cincoffset`, RTL does not | hand to Tier 4 |
| `SQLITE_OPT_LEVEL=-O0`, RV8/BEEBS `-O0` defaults | compiler-debt candidates | retire after 2a passes + board confirmation |
| CoreMark's 7 pins | compiler-debt each (the script itself calls them backend bugs) | the `crcu8` byte-spill tag clear is a **-O0 miscompile reproducible under QEMU** — highest-priority fix, since -O0 ships |
| `-fno-jump-tables` (everywhere) | compiler-debt: scalar `lw` through a `.rodata` table | fix candidate: target-default `-capstone-min-jump-table-entries` or capability-load lowering |

Record: `capstone/tests/workarounds/CLASSIFICATION.tsv` — `id | knob(file:line) | default | corpus+levels
| qemu_off_vs_on | dom_sha_differ | board_off_vs_on(bitstream, control) | class | action | evidence`.
Result TSVs only, never captures. **Classification is reported before anything is removed, and
every `class=` row is reviewed by `claim-auditor` before it drives a retirement** — the prompt
names the weakest link ("what would make this silicon-debt verdict actually be a compiler bug
hiding behind a silicon excuse?"). A "silicon-debt" verdict that later turns out to be
compiler-debt is exactly the claim class the auditors have refuted four times on this project.
The same rule holds in Tiers 1 and 3: an XFAIL that flips green, a fuzz finding classified as
generator UB, or a crash signature added to `known-signatures.txt` is a claim, and gets the same
review before it is recorded.

### 2c. Random-program differential — days 3–4

- **Install** under `$HOME/opt` via `capstone/tests/fuzz/setup-fuzz-tools.sh` (idempotent; pins
  recorded in `TOOLS.lock`; verify csmith/yarpgen flag spellings at install, do not assume).
- **Reference** = native x86 `clang -O0`; native `-O2` also run, disagreement ⇒ discard the seed
  (generator UB). Capstone `-O0` is a **second subject, not a baseline** (finding 12: -O0 has its
  own tag-clearing bug). Matrix per program: {-O0, -O2} × {default ABI (`start.S`+`link.ld`),
  silicon ABI (`build-ladder-domain.sh` flags, `link-gpfree.ld`)} = 4 doms. Equalise char
  signedness on the *reference* (`-funsigned-char` on x86).
- **Runtime overlay** `capstone/tests/fuzz/csmith-rt/`: own `platform.h`, no stdio; the checksum
  lands in `volatile unsigned capstone_fuzz_checksum`; `fuzz_domain.c` returns it through `*res`
  (`FUZZ_XOR=1` build is the positive control). csmith flags: `--no-argc --no-float --no-unions
  --no-packed-struct --max-array-dim 2 --max-funcs 6`, every 4th seed `--builtins` (targets the
  C-20 class). yarpgen `--std=c`, driver printf folded into the same checksum.
- **Build** `build-fuzz-program.sh <src> <name> <opt> <abi>`: production flags +
  `-mllvm -verify-machineinstrs`, link `beebs_freestanding_string.c` at the same `-O` for the
  aggregate-copy libcalls -O2 emits; reject `.dom` > 1,376,256 B; native prefilter `timeout 2s`.
- **Batch runner** `capstone/tests/runtime-qemu/run-domain-batch.py` (generalises
  `run-domain-smoke.py`): manifest of doms, one boot, `BEGIN/END name rc=` markers, **resume after
  a fault** (a domain fault aborts QEMU — finding 22), `flock /tmp/capstone/nightly-qemu.lock`.
  Three calibration boots first: items-per-boot ceiling, the resume path (`stack_oob.dom`
  mid-manifest), the domain stack ceiling.
- **Verdicts** `MATCH | MISMATCH | FAULT | TIMEOUT | BUILD-CRASH | BUILD-ERROR | LINK-ERROR |
  SIZE-SKIP | GEN-UB | GEN-SLOW` → `capstone/tests/fuzz/results/<date>.tsv`.
- **Reduce** cheapest-first: per-function `optnone` bisection (all variants in one boot) →
  `-opt-bisect-limit` (16 per boot) → `llvm-reduce` with an execution interestingness script.
  Do not install cvise/creduce now; ask if findings stay too large. Findings →
  `capstone/tests/fuzz/findings/<ID>/` + `INDEX.tsv` (the nightly's KNOWN allowlist).
- **Throughput**: ~25 programs (100 doms) per 5-min cycle ⇒ 200–300 programs/hour; ~2000/night.

## Tier 3 — IR-level crash fuzzing (no QEMU) — day 5

- `capstone/tests/fuzz/run-llc-stress.sh --seeds N`: per seed `llvm-stress -seed S -size 200`,
  rewrite `\bptr\b` → `ptr addrspace(200)` and prepend the capstone datalayout + triple
  (`llvm-stress` emits AS0 pointers, illegal under `A200/P200`); first pass with FP/vector
  modules filtered (FP is libcall-only, no compiler-rt). Run `llc -mtriple=capstone64 -mattr=+m
  -verify-machineinstrs` at -O0 and -O2, `opt -O2 | llc -O2`, and ~30 random `opt -passes=`
  pipelines over the hand-written seeds. Classify `OK | CRASH(sig) | ERROR(msg) | HANG`;
  `triage-llc.py` groups by signature and honours `known-signatures.txt`. Reduce with
  `capstone/tests/reduce.sh`. Hand-off to Tier 1: `llvm/test/CodeGen/Capstone/fuzz-<id>.ll`
  (`not llc` before the fix, FileCheck after).
- **Shape seeds** `capstone/tests/fuzz/ir-seeds/*.ll` — the constructs the inventories flagged:
  byval/sret with pointer members; varargs with pointer args; memcpy/memmove/memset of
  16/17/24/32/48 B aligned and misaligned, zero and non-zero fill; i128 add/mul/shl/lshr/or/xor/
  select/icmp; ptrtoint/inttoptr round trips to i64 and i128; select/icmp on AS200 pointers with a
  >64-bit constant arm (C-17 `wide_arm`); cmpxchg/atomicrmw with `+a`; 128-bit `_Atomic`
  load/store/RMW; `musttail`; switch ≥ 8 cases; dynamic alloca; indirect calls through
  function-pointer globals; frames > 2 KiB and > 32 KiB; addrspacecast; `cttz`/`ctlz`/`ctpop`
  (finding 19); and one seed per `llvm.capstone.*` intrinsic.
- **Definition (b)'s execution half**: `capstone/tests/fuzz/intrinsics/<name>.c` — one per
  builtin, returns a sentinel via `*res`, run at -O0 and -O2 through the batch runner. The
  `cap_tighten(p, 999)` crash becomes a Sema test in Tier 4.
- Cost: 5000 seeds × 2 levels ≈ 2 min at 90-way parallelism; pre-commit: 300 seeds ≈ 10 s.

### Board confirmation — day 6 (silicon confirms *classified* items only)

Per the `board-run` skill: control first, expected-returners ascending, one unknown last, every
rung at a distinct `DOMAIN_BASE_VA`, `preflight-board-run.sh` green, results as result lines in
`capstone/tests/board-results/<date>.tsv` — never captures, never the console URL.
`known-good-controls.md` is stale: the first boot re-verifies `k800` on
`caplifive_s12fix_5097eb166.bit`.

First session, 8–10 boots (~1 h): (1) `k800` + `s06agg` (expect 15, no workaround); (2–3)
`k800` + BEEBS `bs`, `crc32` at -O2; (4) `k800` + `rv8_primes` -O2 (the known -O1 silicon hang);
(5) `k800` + `coremark_matrix` -O2; (6) `k800` + SLT `q_two.test` -O1 with the S-12 workaround
OFF (doubles as S-12 draws 5–6, p 0.071 → 0.0095); (7) `k800` + SLT `select1.test` -O1; (8)
`k800` + the `sm0`/`sm` pair (S-04); (9–10) two QEMU-clean csmith rungs in the silicon ABI.
After three boots show -O2 rungs returning, stack up to 5 per boot with resume-on-wedge
("read no further than the first failure" keeps the earlier results valid) — a throughput trade
to confirm with the project lead at that point.

## Tier 4 — semantics + ABI audit (≈ 80 h) — does the compiler's model match spec, QEMU, RTL?

Sources of truth: spec `capstone-spec` (`cap-man-insn.adoc`, `ctrl-flow-insn.adoc`,
`insn-list.adoc`, `prog-model.adoc`); QEMU `capstone-qemu/target/riscv/op_helper.c`, `cap.h`;
RTL `capstone-ariane/core/anvil_build/capstone_{flu,dyn}_unit.anvil`, `decoder.sv`,
`commit_stage.sv`. Directed RTL tests are `audit-<insn>-<case>.S` in
`capstone-ariane/verif/tests/custom/capstone/` (content from this session, landed by the RTL
session); compiler-side tests are `llvm/test/CodeGen/Capstone/sem-*.ll` and
`clang/test/CodeGen/capstone-sem-*.c`. **Every `.S` expectation names the exception
(`UNEXPECTED_CAP_TYPE`, `ILLEGAL_OPERAND_VALUE`…) never a numeric `mcause` — R-24 says the
execute-path encoder is +1.** rtl-sim rules apply (delete `out_*` first; `SUCCESS` at the timeout
is not a pass; no `MACRO(` in comments). The closure artifact is
**`llvm/lib/Target/Capstone/CapstoneISASemantics.md`**: one row per emitted instruction —
compiler assumption (file:line) / spec / QEMU / RTL / (i) consumes LINEAR rs1? (ii) untagged-
sealed-wrong-type behaviour (iii) result type / the test that settles it / "not audited" marked
explicitly.

### 4.1 Linearity — first, because it decides `PseudoCapGlobalBase`, `copyPhysReg`, and the CoreMark diagnosis
- **Verdict already in hand (finding 27):** spec, QEMU and RTL all *consume* a LINEAR `rs1` on
  `cincoffset`/`scc`; the compiler's pure-use model is wrong, and it cannot model per-value
  linearity at all. **Adopt the linearity contract**: compiled code holds only NONLIN, cnull or
  untagged values in GPCR; a LINEAR/UNINIT/REV/SEALED value exists only between a producing
  builtin (`cap_init/revoke/mrev/seal/call`, hand-encoded split) and exactly one consumer, with
  no copy, `cincoffset`, `scc`, `stc` or `cjalr` in between. Enforce in `llvm/` with an opt-in
  post-RA verifier `CapstoneLinearityVerifier.cpp` (`-capstone-verify-linearity`, ~150 lines:
  any vreg defined by INIT/REVOKE/MREV/SEAL/CAP_CALL with >1 use or a non-consuming use is an
  error), run over every corpus build; at the boundary the glue `delin`s the entry capabilities
  (runtime session). Record the contract next to `CIncOffset` in the `.td`.
- **DELIN decision (finding 28)** — the highest-consequence item in the audit. Under the default
  ABI every global access is `cincoffset; delin` on a NONLIN gp: a spec/RTL trap that QEMU hides.
  Recommend **(A) `-capstone-gp-captable` becomes the default for silicon builds** — it is the
  model "proven to run on captype-fixed CVA6" — and **(B) the QEMU-default path drops its DELIN**
  under the contract "gp is NONLIN at entry" (`cincoffset` copies metadata, so the result is
  already NONLIN on all three). Tests: `audit-delin-nonlin.S` (expect `UNEXPECTED_CAP_TYPE`;
  control on LINEAR → no exception), `sem-delin-not-on-nonlin-gp.ll`, and `CT-NOT: delin` added
  to `cap-gp-captable.ll`. This is the project lead's call on ABI direction — **decision point**.
- Per-instruction rows for `cincoffset/cincoffsetimm` (also: a *tagged* rs2 traps on spec/RTL,
  QEMU accepts it — and the compiler reads addresses as `EXTRACT_SUBREG` with no ALU write, so an
  X-half of a live capability can reach rs2 on silicon: `audit-cincoffset-tagged-rs2.S`,
  `sem-cincoffset-rs2-from-extract.ll`), `scc` (QEMU *asserts* — emulator abort, not a fault),
  `init` (RTL duplicates, spec/QEMU consume — R-21 stays OPEN for INIT/TIGHTEN only), `tighten`
  (RTL raises on imm > 7 — Sema range becomes 0..7), `movc` (RTL nulls an *untagged* source,
  QEMU keeps it — reachable via the `(void*)(uintptr_t)v` INSERT_SUBREG bridge when the source
  stays live: `audit-movc-untagged-source.S`, `sem-movc-untagged-live.c`), `stc`/`ldc`
  (linear-slot semantics; verifier rule "no two LDC of one spill slot on a path"), `seal`/
  `mrev`/`revoke`/`drop` (DROP: RTL keeps register bits, QEMU nulls — `audit-drop-then-lcc.S`).

### 4.2 Total vs trapping
- `cap_get_tag` → `lcc` selector 0, which traps on an untagged operand on both implementations
  (finding 29): lower as `lcc rd, rs, 1; sltiu rd, rd, 7` (selector 1 is total on both; answers
  "is a capability", not "is valid" — add `cap_get_type` if validity is needed). Frame
  realignment's `lcc sp, 2` is safe (sp is always tagged) but is the last `lcc 2` in the backend —
  replace with the plain move for uniformity. Tests: `audit-lcc-sel0-untagged.S`,
  `audit-lcc-sel1-untagged.S`, `sem-get-tag-total.c`, `sem-frame-realign-addi.ll`.
- `shrink` (default-on): spec/QEMU/RTL agree (29 on `rs1 >= rs2` or outside parent) — but
  **zero-size objects** (`alloca(0)`, zero-length arrays, empty structs) emit a trapping SHRINK
  today: `audit-shrink-zero.S`, `sem-shrink-zero-size.ll`, decide skip-vs-size-1. End-bound
  convention: RTL/QEMU/compiler are end-exclusive, spec text is inclusive, RTL SHRINKTO is
  inclusive against its own LDC — `audit-shrink-last-byte.S` pins the silicon convention; spec
  question to the hardware side.
- `cjalr`: spec consumes a non-NONLIN rs1, neither implementation does; compiler matches the
  implementations — `audit-cjalr-linear-target.S`, spec question.

### 4.3 Side-effect ordering
DROP/DELIN/MREV are `hasSideEffects` without memory flags, so a load through the same capability
may be scheduled across a DROP and read through a dropped cap. `sem-drop-orders-loads.ll`; if it
moves, add `mayLoad/mayStore` to DROP (cheap) — C-35 only if the test fails.

### 4.4 Domain-boundary ops (no in-tree C uses them; lit-only)
`CAP_CALL`: tie `$rd = $rs1`, `Defs = [C1]`, full clobber mask (the callee domain scrubs every
register — `getCallPreservedMask` falsely promises callee-saves survive). `CAPENTER`: fix funct7
to the decoder's; delete `CAPEXIT` (in neither spec nor RTL); operand model `(ins GPR, GPR)` with
implicit defs of C10/C11. `CAP_RETURN`: `(ins GPCR:$rd, GPR:$rs1, GPR:$rs2)` with `rd` encoded;
the `rd = 0` exception-return form is unsupported by RTL — separate def or drop. Confirm each
against `decoder.sv` and `insn-list.adoc` with `rtl-oracle` before editing (finding 29 marks
these as C's reading, not yet re-verified). One entry, C-36.
Undefined ops: define **SPLIT** only (`(outs GPCR:$rd, GPCR:$rs1_out)`, tied, MC + one lit
test, no ISel use — the glue hand-encodes it today); SHRINKTO not needed; BORROW/SDLIN/UNSEAL
exist nowhere.

### 4.5 c128 residue (finding 30)
Dead by construction — delete after a mechanical proof (`-Wunused-function`, then an
instrumented corpus build with `report_fatal_error("i128-residue: <site>")` behind a
`cl::opt` at the *possibly-live* sites only): `lowerSUB/lowerADD/lowerScalarI128*` (with the
false diagnostic at `:8470`), `ISelDAGToDAG.cpp:1285,1501,1532`, the i128 loadext actions.
Possibly live — instrument first: `BITCAST i128` + the bridge `:8806` (pinned by
`cap-constants*.ll`), `recoverCapabilityFromAddressArith :21699-21788` (accepts i128 source),
`canMergeStoresTo :2908`. Upstream-legitimate — keep: `:2583`, `:19514`, `:25467/:25534`,
`:26431/:26525` (RISCV's RV64 pair-register code). Then the stale comments and
`i128-capability-fixes.md` (replace with a 10-line pointer to `f12ae7d5`). Two probes: the
`SELECT_CC`-with-swapped-CC route to `:10832` (`-debug-only=isel` on `icmp ugt` of two
capabilities at -O2); 128-bit `_Atomic` RMW at three levels (expect libcalls; T14).

### 4.6 clang ABI decisions
- **Keep `DefaultABIInfo`; do not wire `CapstoneABIInfo` as it stands.** Indirect-everything is
  tag-correct (finding 6) and there is no other compiler to interoperate with; RISCV's classifier
  would flatten a 16-byte `{void*}` into `[2 x i64]` (tag loss) *and* derive XLen = 128. The
  CHERI-style "small cap-bearing struct in C registers" ABI is a separate project — file as a
  gap. Delete the dead class + `interrupt` handler or leave a one-line "unreferenced" comment.
  Tests `clang/test/CodeGen/capstone-abi-*.c`: byval/sret cap struct, `va_arg` of it, empty
  struct, > 2×XLen, union with a cap member, `_Complex double`, `long double` pass/return; one
  `-x c++` smoke that a class with a destructor is passed indirect (C++ is otherwise out of scope
  — no libcxx port, no C++ domain).
- **`uintptr_t` stays 64-bit** — `TargetInfo::IntType` has no 128-bit member, and an integer
  round trip cannot carry a tag anyway (CHERI needed `__intcap_t`, weeks of Sema/AST work).
  Soften the claim: `uintptr_t` is optional in C; "if provided, the round trip does not preserve
  provenance". Actions: (a) musl `__scc` passes pointer arguments as `void *` (runtime session —
  first run `run-hostcall-all.sh` on the *fixed* QEMU: by the typedefs it must fault, and if it
  passes the plan has misread the path); (b) new `-Wcapstone-pointer-roundtrip`, default on for
  capstone64, warning on `(T*)(integer derived from a pointer cast in the same expression)` and
  on `(T*)x` with `x` of `uintptr_t`/`intptr_t` type; test `clang/test/Sema/capstone-pointer-roundtrip.c`.
- **Sema**: new `clang/lib/Sema/SemaCapstone.{h,cpp}` mirroring `SemaRISCV::CheckBuiltinFunctionCall`
  + `Sema.h`/`Sema.cpp`/`CMakeLists.txt` plumbing + the dispatch case at
  `SemaChecking.cpp:2096-2098`. Checks: `cap_tighten` imm constant in **0..7**; `cap_ccsrrw` id
  in `{0,1,2,4,16..31}` (a set, not a range); first operand of every `cap_*` builtin must be a
  pointer; `cap_shrink` base < end when constant; inherited RISC-V crypto/bitmanip ranges ported
  by sharing the switch body. Tests `clang/test/Sema/capstone-builtins-range.c`,
  `capstone-inherited-riscv-ranges.c`. Also: define `__CAPSTONE__`/`__CAPSTONE_PURECAP__`
  (+ `Preprocessor/capstone-macros.c`); `setABI("lp64e")` returns **false** (a non-capability
  datalayout with `AddrSpaceMap = 200` is incoherent) + test; delete the capstone32 `errs()`
  prints + a no-stderr test; driver `-march`/`-mabi` recorded as out of scope.

### 4.7 Shared-LLVM patch audit
Drift base `b3a1c7778245` (parent of the RISCV copy; pure upstream). Checked-in manifest
`llvm/utils/capstone-shared-patches.txt` (path, expected ±line counts, one grep marker per
file — `isCheriCapability`, `getAddrSpace()` at `SelectionDAG.cpp:9204`, `NonIntegral` in
`ValueTracking.cpp`, …) and `llvm/utils/capstone-shared-drift.sh`, driven by
`llvm/test/CodeGen/Capstone/shared-patches-present.test` (`REQUIRES: shell`) so `llvm-lit`
reports drift after any upstream merge. Per site, a *target-specific* test so it is not guarded
only by RISCV tests: `sem-tblgen-iptr-cap.ll` (the build itself), `sem-mvt-c128.ll`,
`sem-memset-libcall-as200.ll` (C-16/18: argument stays c128, `CHECK-NOT: mv a0`),
`clang/…/capstone-licm-null-gep.c` (C-19, also booted at -O2), `.quad sym`/`.quad 0` on the
constant-initializer tests, `sem-gep-c128-index-width.ll`, a `__builtin___clear_cache` test.
The `CodeGenDAGPatterns.cpp:1406-1412` >64-bit-immediate guard may now be dead — instrument as
in 4.5.

### 4.8 Auditor protocol — every claim goes out with its weakest link named
| item | agent | weakest link to attack | evidence that counts |
|---|---|---|---|
| R-21 stale | `rtl-oracle` | is `check_fwd_rs1` (`ariane_pkg.sv:970-975`) still consulted after the 08-12 clear; does any existing `.S` read rs1 in the *very next* instruction (zero-gap)? If not, `audit-cincoffset-linear-gap0.S` is written first | quoted lines; an RVFI line one instruction after the clear |
| DELIN decision | `rtl-oracle` → `claim-auditor` | oracle: quote `dyn:470-490` + decoder remap; auditor: "the default ABI never ran on silicon" — find one passing board log of a non-captable domain, which would refute it | `.iss` `UNEXPECTED_CAP_TYPE` line; board log path or its absence |
| rs2 hazard | `rtl-oracle` | does `issue_read_operands.sv` really read the metadata shadow for rs2 of CINCOFFSET, and does an `EXTRACT_SUBREG`-only path leave it tagged? | `.iss` line; quoted write-enable |
| movc untagged | `claim-auditor` | "reachable from C" needs an actual -O2 object with a `movc` whose untagged source is live after — produce it or downgrade to LATENT | objdump lines + RVFI trace |
| C-28 tail call | `claim-auditor` | does removing `-fno-optimize-sibling-calls` flip `cjalr ra` → `cjalr zero` in `core_bench_matrix`, and does the -O2 boot then pass with no other change? | before/after disassembly of one function; boot log |
| C-29 LINEAR args | `claim-auditor` | the `lcc a1, 1` probe must read at entry, before any copy; QEMU's monitor stand-in may mint a different type than the board's monitor | probe output on QEMU and rtl-sim; the monitor's mint site |
| C-30 mixed lowering | `claim-auditor` | the workaround comments predate c128 — re-observe on the current compiler, one variable at a time | per-TU flag matrix with commit hash and boot result |
| dead i128 sites | `claim-auditor` | "no callers" by grep is not "unreachable" — `-Wunused-function` output and the instrumented corpus build, before deletion | build log; corpus command + exit status |
| hostcall / `__scc` | `claim-auditor` | by the typedefs the path must fault; if the suite passes, find the real `__syscallN` | suite log; file:line of the macro actually used |
| QEMU binary provenance | `claim-auditor` | which commit built the 08-27 binary; what else changed since | `git log` between candidates; `--version`/`strings` if it embeds a hash |

### 4.9 Order within the tier
Tier-0.0 QEMU fix (hand-off) → 4.1 rtl-sim tests (need no QEMU) → DELIN decision → C-28 fix
(smallest, highest value) → 4.5 dead-code proof and deletion → 4.2 fixes/tests → 4.6 Sema,
macros, `lp64e`, ABI tests → 4.7 manifest → 4.3/4.4 → C-29/C-30 reproductions on the fixed QEMU
→ -O2 corpus rebuild without workarounds → board (2 boots: -O2 CoreMark without workarounds under
gp-captable; the delin-free default-ABI confirmation). rtl-sim ≈ 60 runs × 14 s.

## Tier 5 — fix what is proven, retire what is classified

Every item below enters only after its Tier 2b classification row and its Tier 4 audit row are
filled, and every "fixed" passes `claim-auditor` before the commit message says so. Codegen
fixes are made in this session, one commit each, each with the regression test from Tier 1.

### 5.1 Compiler bugs to fix (proposed IDs; the docs session allocates)
| ID | defect | fix | test |
|---|---|---|---|
| **C-28** | tail calls selected as calls — falls off the end at -O1+ (finding 23) | `selectCall`: on `CapstoneISD::TAIL`/`SW_GUARDED_TAIL` build `PseudoTAILIndirect` (class `GPCRTC`); `isEligibleForTailCallOptimization` unchanged | T1; then rebuild `core_state.c`/`core_matrix.c` **without** `-fno-optimize-sibling-calls` and boot |
| **C-20** | `__builtin_ctz` crashes the legalizer at every -O (finding 19) | add the `CTTZ`/`CTLZ`/`CTPOP` legalization actions the RISCV copy lost when GPR became 64-bit-only, or Expand | T16 |
| **C-27** | `delin` on a NONLIN gp is a spec/RTL trap QEMU hides (finding 28) | per the Tier 4 DELIN decision: drop the DELIN under the NONLIN-gp contract; gp-captable default for silicon | T18, `CT-NOT: delin` |
| **C-33** | `cap_get_tag` traps on NULL (finding 29) | lower as `lcc rd, rs, 1; sltiu rd, rd, 7`; realign's `lcc sp, 2` → plain move | `sem-get-tag-total.c`, `sem-frame-realign-addi.ll` |
| **C-34** | zero-size objects emit a trapping SHRINK | skip SHRINK for size 0 (or clamp to 1) | `sem-shrink-zero-size.ll` |
| **C-32** | `movc` of an untagged c128 nulls the source on silicon while it is live | make the integer→capability INSERT_SUBREG pseudo rematerializable; PHI copies remain → file the residue | `sem-movc-untagged-live.c` |
| **C-31** | a tagged rs2 can reach `cincoffset` via `EXTRACT_SUBREG` with no ALU write | audit-dependent (4.1 rs2 row); if confirmed, force an `addi rd, rs, 0` between extract and use | `sem-cincoffset-rs2-from-extract.ll` |
| **C-36** | domain-op definitions (CAPENTER funct7, CAPEXIT phantom, CAP_RETURN operand roles, CAP_CALL clobbers) do not match the ISA | per 4.4, after `rtl-oracle` confirms each | MC M1/M5 + `sem-domain-ops-encoding.s` |
| **C-38** | `call a0, a1` unassemblable (finding 25; C-25 is the ptrdiff-untagged fix, so the mnemonic bug is C-38) | rename the `CAP_CALL` mnemonic or give `parseCallSymbol` lower precedence — main-session choice | M4 loses its XFAIL |
| — | missing Sema for every `__builtin_capstone_*` (finding 3) | `SemaCapstone.cpp` (4.6) | `capstone-builtins-range.c` |
| — | S12MovcLdcHazard inert on c128 (finding 24) | **retire the pass** (S-12 is fixed in RTL; the pass never fired post-c128) after two more clean board draws | T4 flips from XFAIL to deleted |
| — | `-capstone-memcpy-high-half-fixup` family | **delete** — S-06 fixed in silicon, deletion already authorised by `S06-WORKAROUNDS-TO-REVERT.md` §3, re-confirmed by one board rung | `flags-memcpy-fixup.ll` becomes a removal test |
| C-29 | "rd!=rs1 LINEAR sink" on domain arguments | **not a compiler bug** — the linearity contract: `delin` entry caps in the glue (runtime session); verifier over the CoreMark objects | `-capstone-verify-linearity` |
| C-30 | "mixed scalar/capability lowering" + the -O0/-O1 `-fno-inline` crash in `core_matrix.c` | probably **stale** (names the retired i128 carrier); re-observe one variable at a time; reduce whatever survives, retire the rest with the evidence line | whichever `.ll` falls out |

### 5.2 Build-script pins to retire, each with its evidence line
From the Tier 2b `CLASSIFICATION.tsv`, in this order: `-fno-optimize-sibling-calls` (after
C-28); SQLite `SQLITE_OPT_LEVEL=-O0` and RV8/BEEBS `DOMAIN_OPT_LEVEL=-O0` defaults (after the
-O2 twins agree and the board confirms); CoreMark's per-object pins (after each single-knob
variant passes); `-fno-jump-tables` everywhere (after the jump-table lowering is fixed or the
target default changed); `BEEBS_STRING_WRITERS_OPTNONE` (measured neutral; silicon-debt — keep
only if the board pair differs); `BEEBS_MEMCPY_OPTNONE` stays until S-04 is fixed in silicon.
**The `-O0` byte-spill tag clear (`build-coremark-capstone.sh:108-111`) is a -O0 miscompile
reproducible under QEMU — it gets an ID and a fix before -O0 is called safe.**

### 5.3 What is deliberately not done here
Wiring `CapstoneABIInfo` (a new ABI, not a fix); a 128-bit `uintptr_t`; driver `-march`/`-mabi`;
C++ support; SHRINKTO/BORROW definitions; changing the synthesis flow or the board bitstream.

## Hand-offs to the `capstone` session (owner of `docs/` and `ISSUES.md`)

This session edits `llvm/` and `clang/` only. Findings that belong in the registry are sent
by message, with evidence, as they are confirmed:

- **Four unfiled backend bugs** documented only in `build-coremark-capstone.sh` — proposed
  C-27 (tail-call lowering emits `cjalr ra` instead of restore-ra + `cjalr zero`), C-28
  (`rd!=rs1` LINEAR-cap sink above `-O0`), C-29 (mixed scalar/capability lowering,
  `core_matrix_capstone.c`), C-30 (shared gp-derived LINEAR table pointer consumed by the first
  `cincoffset`; "root fix is delin emission"). Plus the `-O0`-caused granule-share tag clear.
- **Missing Sema for `__builtin_capstone_*`** — `cap_tighten(p, 999)` crashes the compiler.
- **`uintptr_t` is 64-bit** while `void*` is 128 — a C-standard violation; musl `__scc` depends on
  the opposite.
- **`CapstoneABIInfo` is dead code** — the shipping ABI is `DefaultABIInfo`.
- **Registry hygiene** (not this session's to fix, but found): C-20..C-25 are absent from
  `ISSUES.md` (C-20/C-24 both name a `__builtin_ctz` crash; C-22's fix status unknown; C-23
  never used); the ISSUES.md C-26 (coverage gap) collides with commit `beab934804b9`'s C-26
  (vararg AS0 miscompile, fixed, no test, no entry); R-18's entry claims a flag
  (`-capstone-int-zero-for-zero-copy`) that does not exist in the tree; `ISSUES.md:628` says the
  QEMU smoke/authority suites are BROKEN while the nightly shows them passing.
- **Tier 0.0 — the QEMU oracle is unverified** (finding 26): `capstone-qemu` HEAD does not
  compile and the binary predates the c128 QEMU merge by 8 days. The `capstone` session holds
  that submodule; it needs to fix, rebuild, record the 08-27 binary's provenance, and re-run
  `sqlite-slt` + `smoke`. Every QEMU result line in this plan names the binary it ran on.
- **R-21 needs an update, not a report**: RTL now consumes a LINEAR `rs1` on `cincoffset`/`scc`
  (finding 27; `capstone_flu_unit.anvil:45-46,76,109-112`, in the flashed bitstream); the entry
  stays OPEN for INIT (duplicates) and TIGHTEN (passes through) only. Spec questions for the
  hardware side: `cjalr` consumption (spec yes, both implementations no); end-bound convention
  (spec inclusive, RTL/QEMU/compiler exclusive, RTL SHRINKTO inclusive against its own LDC).
- **QEMU fidelity items** found by the audit, for whoever owns the model: `scc`/`init`/`lcc`
  misuse are *asserts* (emulator abort) not guest faults; `delin` of a NONLIN cap is a silent
  no-op where RTL raises 26; `drop` nulls the register where RTL keeps the bits; `lcc` selector 0
  returns a constant 1; `cincoffset` accepts a tagged rs2 where RTL raises 24; `seal` size/
  alignment only printed where RTL raises.
- **Proposed IDs** C-27..C-36 as listed in Tier 5.1, with one-paragraph entries drafted by this
  session and sent with their evidence; the ISSUES.md C-26 collision resolved by renumbering the
  vararg fix.
- **New paths this session creates under `capstone/tests/`** — `fuzz/`, `twins/`,
  `workarounds/`, `board-results/`, `check-lit-coverage.sh` — are new and unheld; the `capstone`
  session is told before the first commit. `run-nightly.sh` rows and `build-coremark-capstone.sh`
  knobs are edits to held files and go by message with the exact diff.
- `audit-*.S` directed tests land in `capstone-ariane/verif/tests/custom/capstone/` via the RTL
  session, content supplied by this one; `known-good-controls.md` is stale and the first board
  boot re-verifies `k800` on the resident bitstream.

## Ownership and constraints that bind every tier

- `ninja -j90`, never `-j112`. QEMU suites serialized on the `rootfs.ext2` lock — one at a time,
  and check for a live `qemu-system-riscv64` before launching. Board serialized across lanes;
  reflash is ask-first.
- Every commit: `git commit --only <paths>`; `precommit-scan.sh` by absolute path, gated on
  `rc=0`; no names; no `Co-Authored-By`.
- Codegen fixes are made in this session. Subagents only review (`claim-auditor` before any
  "fixed" / "ruled out" enters a commit message), read RTL (`rtl-oracle`), run suites
  (`corpus-runner`), or classify logs (`board-log-forensics`). Every auditor prompt names the
  weakest link to attack.
- A clean result is not evidence until the check is shown to fire: every new CHECK-NOT gets a
  mutation that turns it red; every oracle gets an injected miscompile it must catch.

## Nightly integration (day 7)

The core tier is ~52 min today, so there is headroom. New `CORE_SUITES` rows in
`capstone/tests/run-nightly.sh`, each directly after its -O0 sibling:

| row | command | timeout | est. |
|---|---|---|---|
| `rv8-O2` | `DOMAIN_OPT_LEVEL=-O2 … run-all-rv8.sh` | 1800 | 2 min |
| `beebs-O2` | `DOMAIN_OPT_LEVEL=-O2 … run-all-beebs.sh` | 3600 | 20 min |
| `coremark-O2` | `DOMAIN_OPT_LEVEL=-O2 COREMARK_UNIFORM_OPT=1 … run-coremark.sh` | 1800 | 15 s |
| `sqlite-slt-O1`, `-O2` | `SQLITE_OPT_LEVEL=-O1/-O2 … run-sqlite-slt.sh` | 3600 | 5–15 min each |
| `authority-O2` | `DOMAIN_OPT_LEVEL=-O2 AUTHORITY_ONLY=<eligible> … run-authority-suite.sh` | 3600 | 15 min |
| `twins` | `compare-twins.sh` — the agreement gate, runs last, no QEMU | 300 | 5 s |
| `fuzz-llc-stress` | `run-llc-stress.sh --seeds 5000` (outside the flock) | 900 | 2 min |
| `fuzz-csmith`, `fuzz-yarpgen` | `run-fuzz-campaign.sh --gen … --count 60/40` | 2400 | ~25 min each |
| `intrinsics-exec` | `run-intrinsics-exec.sh` (-O0 + -O2) | 600 | 1 min |
| `coverage-gate` | Tier 1's accounting script — non-zero on any uncovered instruction/flag | 60 | 1 s |

Weekly (`--weekly`): `workaround-matrix` (~30 min), `authority-opt-matrix` (O1/O3),
`fuzz-csmith-long` (500). Total nightly ≈ 2.7 h. **Pre-commit tier** (`--quick`, ~4 min): today's
four rows + `fuzz-llc-stress --seeds 300` + `csmith-compile-only --count 20` (both ABIs,
`-verify-machineinstrs`, no boot) + the binary-currency check from Tier 0.4.

## Verification (end-to-end) — every instrument shown to fire before its first verdict

**Execution oracles — positive controls (Tier 2/3):**
1. Markers + twins gate: a hidden `-mllvm -capstone-chaos-inject=N` (main-session `llvm/` edit:
   emit the N-th capability `stc` as `sd`, or swap the operands of the N-th `sub`) applied at -O2
   must produce `O2-ONLY-FAIL` rows. Fallback with no compiler change: link a `memcpy` that drops
   its last byte into `rv8_miniz`.
2. SLT: the new `slt-compare.sh` harness must report every deliberately-wrong arm of
   `slt/negative-control.test` as a mismatch at every -O level, on the rebuilt QEMU, before any
   `sqlite-slt` row is read as a value result (`check-negative-control.sh` already proves the
   comparator can fail; the harness must prove the *per-record domain-vs-native* path can).
3. csmith: the `FUZZ_XOR=1` build → `MISMATCH`; `stack_oob.dom` mid-manifest → `FAULT` recorded,
   reboot, remaining items complete.
4. llvm-stress triage: `compiler-repros/C20-cttz-*/src/cttz.ll` and the C-17 `wide_arm` IR must
   classify CRASH/ERROR with the right signatures; a clean seed → OK.
5. Workaround matrix: the sha256-differs assertion; the `sm0`/`sm` pair is the known
   board-differing pair.
6. Native reference: 100 seeds native -O0 vs -O2 agree; `--no-safe-math` seeds must disagree
   (proves the UB filter fires).

**Static coverage (Tier 1):**
- Lit counts: before — `CodeGen/Capstone` 60, LICM 1, clang 14, MC **0** (76 in scope); after —
  `CodeGen/Capstone` ≈ 82 files / ≈ 200 RUN lines (every file with -O0 and -O1 arms), `MC/Capstone`
  6 + `Disassembler` 1, clang ≈ 16. Declared XFAILs: 5 at the end of Tier 1, **0** at the end of
  Tier 5 — each names its C-nn.
- `bash capstone/tests/check-lit-coverage.sh` exits **1 on the current tree** (11 instructions
  without negative controls, 23 without MC lines, 11/17 flags without RUN lines, 58/58 missing
  -O1, all fatal sites and C-nn unmapped) and **0** when Tier 1 completes; `--self-test` must
  exit 1 against a stub tree.
- Every negative check has its `MUTATION:` header and the mutation was performed once at write
  time (the six vacuous tests' mutations are listed in 1.2).
- Cross-check independent of the script: `for f in llvm/test/CodeGen/Capstone/*.ll; do llc
  -mtriple=capstone64 -mattr=+m -O1 -verify-machineinstrs < $f >/dev/null || echo $f; done` —
  prints nothing today and must keep printing nothing.
- Command: `llvm-lit -v llvm/test/CodeGen/Capstone llvm/test/MC/Capstone
  llvm/test/MC/Disassembler/Capstone llvm/test/Transforms/LICM/capstone-*.ll $(ls
  clang/test/{CodeGen,Sema,Preprocessor}/*capstone*)`.

**Audit (Tier 4):**
- `llvm/lib/Target/Capstone/CapstoneISASemantics.md` exists with one row per emitted instruction
  and **no cell reading "unknown"** — "not audited" is written explicitly where that is the truth.
- Every `audit-*.S` recorded with its `.iss` exception name, its cycle count against
  `+time_out`, and the artifact mtime against the wall clock.
- `shared-patches-present.test` green; `capstone-shared-drift.sh` fails when any manifest marker
  is removed (negative-tested by temporarily deleting one).
- Each audited claim in a commit message carries quoted file:line, a re-runnable command with
  captured output, or an RVFI/`.iss` line — and the `claim-auditor`/`rtl-oracle` report that
  attacked its named weakest link.
- The CoreMark build script's workaround list handed to the docs session with per-item status
  (retired with evidence / kept with C-nn).

**Silicon:** the first board session's result lines, control verdict quoted beside every result;
the `-O2` rows that QEMU passed confirmed on `caplifive_s12fix_5097eb166.bit`.

## Cost, critical path, and the decisions that are the project lead's

**Effort:** Tier 1 ≈ 66 h; Tiers 2–3 ≈ 56 h (7 days); Tier 4 ≈ 80 h; Tier 5 is the sum of the
fixes it inherits (≈ 40 h for C-28, C-20, C-27, C-33, C-34, Sema, the two retirements).
**≈ 240 engineer-hours, ≈ 6 weeks for one lane**, less with the static and execution tiers run
in parallel (they share no files).
**Machine time:** QEMU ≈ 800 runs at 5–15 s each, serialized on the rootfs lock; rtl-sim ≈ 60 runs
× 14 s; board ≈ 20 boots (first session 8–10; then csmith rungs, refreshed pairs, the two audit
confirmations). `ninja -j90` rebuilds after each codegen fix (Debug tree, minutes).

**Critical path** (what must precede what):
1. Tier 0.0 (QEMU fix, hand-off) → every QEMU verdict. Everything static and every rtl-sim test
   proceeds without it.
2. Tier 1 P0 (the five XFAILs) → the bugs are pinned before anything else moves, so a later
   "fixed" has a test that was red first.
3. Tier 2a (-O2 twins, day 1) → the retire-or-file list → Tier 2b classification → Tier 5.2
   retirements. Tier 2c (csmith/yarpgen) runs alongside. **All of 2a–2c is provisional until
   Tier 0.0 lands** (the QEMU rebuild is another session's work): results are dated with the
   binary they ran on and re-run once on the rebuilt binary before any classification or
   retirement rests on them. Day 1 can start on the 08-27 binary — it still finds crashes and
   -O0/-O2 disagreements — but nothing is *retired* on its say-so.
4. Tier 4.1 (linearity, rtl-sim) → the DELIN decision → C-27/C-29 → the -O2 corpus rebuild
   without workarounds → the board.
5. C-28 (tail call) can be fixed on day 1: smallest change, highest value, already pinned.

**Decisions that are the project lead's, surfaced here rather than taken:**
- **ABI direction** (Tier 4.1): make `-capstone-gp-captable` the silicon default and drop the
  DELIN from the QEMU-default path? This changes the global-access contract every domain is
  linked against; the plan recommends it because it is the only path with silicon evidence.
- **Board stacking** (Tier 3 board section): after three boots show -O2 rungs returning, stack
  up to five per boot with resume-on-wedge — a throughput trade against the one-unknown-last
  rule.
- **Retirement scope** already decided ("validate first, then retire proven compiler-debt");
  each retirement still reports its classification row before the removal commit.

## Execution log

### 2026-09-04 — Tier 1 P2 (MC suite) landed; observations for Tier 4

Measured while writing M6/M7, recorded here so the Tier 4 matrix starts with
them rather than rediscovering them:

- **PseudoTRUNC_CAP expands to a full `movc`**, not an integer `mv`, whenever
  the source and destination registers differ (`ptrtoint` of the second
  argument → `movc a0, a1`); with the same register it expands to nothing.
  The metadata half rides along into the "integer" destination. RTL nulls an
  UNTAGGED `movc` source (C-32 shape), so a `ptrtoint` of an integer-derived
  pointer whose source stays live is a candidate for the same defect class —
  add the row to 4.1 (`movc`) and a shape seed to Tier 3.
- **Function-pointer constants are minted from gp.** Under the default ABI
  `__capstone_cap_init` builds the capability stored over
  `@fp = constant ptr @f` with `auipc/addi; cincoffset a1, gp, a1; delin a1;
  stc` — a DATA-authority capability whose cursor is a code address. Under
  `-capstone-gp-captable` the same constant is initialised with the raw
  integer (`auipc/addi; mv; stc`), untagged. Neither is obviously a valid
  `cjalr` target on silicon; this is the "indirect call through a
  function-pointer global" row of 4.4, and `obj-relocs-cap-constant.ll` /
  `obj-relocs-gp-table.ll` pin today's behaviour so a fix has a red test.
- Under `-capstone-gp-captable` functions return with `ret` (plain `jalr`),
  under the default ABI with `cjalr zero, 0(ra)` — consistent with T1's
  gp-free arm; no action.
- **C-37 shape confirmed**: `Capstone.def` names the relocations
  `R_Capstone_*` (mixed case); `lib/Object/ELF.cpp` has no `EM_CAPSTONE` case
  at any of its four `EM_RISCV` switch sites (`:114`, `:220`, `:276`,
  `:570`). Every new type check matches number OR name, so only
  `reloc-names.s` flips when the fix lands.
- The gate's `mc` section scans `MC/Capstone/*.s` only; the `.ll` round-trip
  tests in that directory and the Disassembler `.txt` files carry their own
  `MUTATION:` lines but are not counted by the `mutations` section. Extend
  the scan when the section is next touched (P4).

### 2026-09-04 — Tier 1 P3–P4 landed; the coverage gate is green

- `check-lit-coverage.sh` reports **0 gaps** (271 when it was written): 81 CodeGen
  tests, each with -O0 and -O1 arms or a recorded exemption; 7 MC + 2 disassembler;
  16 clang. Every capability instruction has a positive CHECK, a negative control,
  an assembler line, an invalid-operand diagnostic and a decoder line; every
  intrinsic a CodeGen and a clang use; all 17 flags at both values; the last three
  fatal sites pinned (C-17's diagnostic; the scalable-stack error under `+v`; the
  dynamic-alloca size route recorded unreachable with its pinning test); every
  open C-nn mapped; every negative check in the tree carries a performed `MUTATION:`.
- **Six** declared XFAILs at the end of Tier 1 (the plan said five): `tail-call.ll`
  and `capstone-tail-call.c` (C-28), `s12-movc-ldc-rename.mir` (the S-12 pass is
  inert on c128), `cap-call-mnemonic.s` (C-38), `reloc-names.s` (C-37),
  `c20-cttz.ll` (C-20).
- Observations for Tier 4:
  - **A direct call target is not delin'd**: `cincoffset a3, gp, a0; cjalr ra, 0(a3)`
    (calling-conv.ll), whereas `__capstone_cap_init` delins the same function's
    address before storing it. Both derive a code address from gp's data
    authority; the `cjalr` linearity row gets this asymmetry.
  - **RVV is non-functional on capstone64**: a scalable load/store cannot be
    selected against a c128 pointer ("Cannot select: store (<vscale x 1 x s128>)"),
    and without `+v` a scalable alloca trips an upstream frame-lowering assertion.
    Out of scope for the project; only the `reportFatalUsageError` under `+v` is
    pinned (`fatal-scalable-stack.ll`).
  - The merge-strings pass header says it stays OFF because container references
    lowered through the integer `auipc` fallback; on the branch tools a pointer
    global initialised with a merged literal is minted through the cap table
    (`ldc`/`stc` in `__capstone_cap_init`). Either the gap closed with c128 or it
    needs a different shape; the default is untouched and the comment goes on the
    Tier 4.5 stale-comment list.
  - `-capstone-lower-memops-via-libcall` is declared as
    `DEBUG_TYPE "-memops-via-libcall"` in `CapstoneISelLowering.cpp`, so a grep for
    the flag string finds only the comment that names it; the gate lists it by
    its full name.
  - 128-bit `_Atomic` load/store lower to the size-generic `__atomic_load` /
    `__atomic_store`, the RMWs to the `_16` forms, at every level, with a
    `-Watomic-alignment` warning each.
  - Retired: `capstone-builtins.c` (a duplicate of `builtins-capstone.c`, which now
    covers all 19 builtins) and `capstone-i128-backend-diagnostic.c` (folded into
    `capstone-atomic-128.c`).
- Renamed per 1.5: `cap-i128-and-capability-mask` → `cap-addr-bitmask`,
  `cap-i128-or-undef` → `cap-ptrdiff-nonpow2-div`, `cap-i128-ptr-diff-const` →
  `cap-ptrdiff-const-offset`, `cap-i128-select-capability` → `select-cap-vs-scalar`.
- **Nightly rows — hand-off** (`run-nightly.sh` is held by the capstone session).
  Proposed diff to its lit stage: add `run_one "lit-mc" "$LIT llvm/test/MC/Capstone
  llvm/test/MC/Disassembler/Capstone"`, add `run_one "lit-coverage" "bash
  capstone/tests/check-lit-coverage.sh"`, and extend the `lit` row's file set with
  `clang/test/Sema/capstone*.c`, `clang/test/CodeGen/*capstone*.ll` and
  `llvm/test/Transforms/LICM/capstone-*.ll`.

### 2026-09-04 — Tier 2a first results, and two defects found by the new oracles

- **Tier 2a SLT twins** (harness `capstone/tests/twins/`, positive-controlled; QEMU
  5dc356547d7f built 22:34; compiler = the branch at db079043's codegen): at **-O0**
  select1 (1031 records), q_two and dd2_join **AGREE** with native, 0 failures — the first
  re-runnable SLT value verdict on this project. At **-O1 and -O2 every file faults at
  the domain's first loop** (ERROR: no summary; cause 24 at pc …9d4c).
- **C-40** (`capstone/tests/twins/findings/C40-lsr-null-gep-cincoffset/`): llc's Loop
  Strength Reduction rewrites a pointer loop's exit test into
  `(gep i8, null, %lsr.iv) == null` and the backend emits `cincoffset a0, zero, s4`,
  which raises UNEXPECTED_OPERAND on the untagged null base. Five sites at -O1, eight at
  -O2, none at -O0. Almost certainly the -O1 blocker behind C-3 and "SQLite ships at
  -O0". Pinned red-first (`c40-null-base-cincoffset.ll`); target-side fix written in
  `selectCIncOffset` (a null base lowers like inttoptr, an untagged value carrying the
  offset), rebuild after the twins driver exits.
- **C-39** (`capstone/tests/fuzz/findings/F01-vector-elt-pointer-zext/`): llvm-stress,
  every seed: a variable-index extract/insert on a wide vector is split through a stack
  temporary and `TargetLowering::getVectorSubVecPointer` zero-extends the index into the
  c128 pointer type. Pinned red-first with a value arm; shared-code fix written
  (compute the index in the AS0 pointer type, as this fork's `getMemBasePlusOffset` does);
  validation = Capstone suite for correctness, X86/RISCV/Generic lit as the regression
  check against an unintended effect (those targets cannot exercise the changed branch).
- **Tier 2c harness** (`capstone/tests/fuzz/`): csmith 2.4.0 and yarpgen 2.0 installed and
  pinned (`TOOLS.lock`, flag spellings verified); `build-fuzz-program.sh` builds a
  generated program into a domain with a freestanding csmith runtime overlay
  (`csmith-rt/`) that returns the checksum through the 32-bit result channel;
  `run-domain-batch.py` runs a manifest in one boot and survives a faulting item;
  `run-fuzz-campaign.py` does the native reference, the builds, the batch and the
  verdicts, with the XOR and fault positive controls. Dry run of 10 seeds: 16 domains
  built, 2 generator timeouts, 0 build failures. QEMU runs after the twins.
- Worktree note: the benchmark host programs include a header from the empty
  `caplifive-buildroot` submodule dir by relative path; the worktree now carries a
  RELATIVE symlink to the main checkout's (read-only use; never committed).

### 2026-09-04 (later) — Tier 2a interim, the datalayout cleanup, and the order of the rebuild

- **RV8 twins**: -O0 7/7 PASS; -O2 fails bench after bench (dhrystone: TIMEOUT after
  entry, i.e. a hang rather than a fault; qsort, sha512: FAIL) — the C-40 class is the
  expected cause, with dhrystone's hang to be re-checked once the fix is in. The driver is
  stopped after the CoreMark pair rather than spending an hour of QEMU on BEEBS against
  the unfixed compiler; BEEBS -O0/-O2, SLT -O1/-O2 and RV8 -O1/-O2 run on the fixed build.
- **22 CodeGen tests carried an unparseable datalayout** (`pf200:...`, rejected by the IR
  parser: "address space must be a 24-bit integer"); llc masked it because it replaces a
  module's datalayout with the target's before parsing. All 32 datalayout lines are now the
  target's own string; the opt-O2-then-llc-O2 sweep over the 80 hand-written tests reports
  142 OK and only filed or by-design failures (b82408cb).
- **yarpgen** (2.0, `--std=c`) generates multi-megabyte arrays that do not even link
  natively without a large code model, let alone fit the 1.3 MB domain ceiling; parked
  behind csmith until its array-size knobs are explored. csmith seeds build cleanly at
  -O0 and -O2 with `+m` (the first dry run's -O0 link failures were a missing `+m`).

### Execution log — Tier 4.7 landed, the semantics skeleton, and what the RV8 -O2 twin is saying (2026-09-04, evening)

- **Tier 4.7 is in** (`86d1cc4957fa`): `llvm/utils/capstone-shared-patches.txt` lists the 65
  shared files that differ from the drift base, `capstone-shared-drift.py` checks marker,
  size and unlisted files, and `shared-patches-present.test` runs it under lit. Both negative
  arms were made to fire on a copy of the manifest: an edited marker ("marker not found") and
  an edited count ("diff is +32 -1, manifest says +33 -1"). The first attempt at the count arm
  did not fire because the sed pattern did not match the manifest line -- the check reported
  clean against an unmutated file. Recorded here because it is the pattern this plan warns
  about: a negative test whose mutation silently did not apply reads exactly like a pass.
- **`llvm/lib/Target/Capstone/CapstoneISASemantics.md`** now exists as the Tier 4 closure
  artifact: one row per emitted instruction, the compiler column cited to `.td` lines, every
  other cell either cited or marked "not audited". The rtl-oracle rows (running) fill the RTL
  column next.
- **The RV8 -O2 twin is not the C-40 signature.** dhrystone, qsort and aes time out (the
  domain never returns) and sha512 halts with capability-fault cause 5, on both attempts.
  C-40 shows as cause 24 at the first loop. So RV8 at -O2 carries at least one further
  defect class; the C-40 fix rerun separates "C-40" from "other", and whatever remains gets
  its own ID and matched pair. Those images are not kept by the driver (each build overwrites
  the share dir), so the rerun must capture the doms.
- **Tier 4.5 by grep:** `lowerADD` (member) and the static `lowerSUB`,
  `lowerScalarI128Shift/Logical/LogicalOnCapability/And/Mul` each have exactly one mention
  outside comments -- their definition. Static functions with no caller are what
  `-Wunused-function` reports, so the deletion is one commit after the next build shows the
  warnings; it stays out of the fix commits.
- **Tier 4.6 facts, for the Sema work:** the clang target defines `__capstone`,
  `__capstone_xlen`, `__capstone_cmodel_*`, `__capstone_float_abi_*` and the rest of the
  RISCV-copy names (`clang/lib/Basic/Targets/Capstone.cpp:151-230`); `__capstone_v_intrinsic`
  is defined unconditionally at `:226`, vector or not. There is no `SemaCapstone.cpp`; the
  RISCV one is the model.

### Execution log — fix cycle 1 landed; cycle 2 prepared while QEMU validates (2026-09-05, small hours)

**Cycle 1** (`c11b8fb6` C-39+C-20, `ffcc7347` C-40+C-28, `3980e9bb` the semantics matrix and the
C-31 pin) went through the gates before the QEMU twins: Capstone lit 84 CodeGen + 7 MC + 2
Disassembler + 16 clang with only C-31, C-37 and C-38 still XFAIL; coverage gate 0 gaps;
fuzz-check ALL OK; llvm-stress 598/600 against 0/600 (F-02, F-03 filed with nine-line
reductions); the X86/RISCV/Generic suites 7727 tests, 32 failures, all missing-tool or
debug-info/emutls class, unattributed for want of a baseline, argued identity for AS0 targets.

**What the twins on the cycle-1 compiler say so far:** the -O0 SQLite image is byte-identical
to the pre-fix one (dom `2880bd1af983`, the control), and `select1` at -O1 AGREES with native
on all 1031 records (dom `04cc91bce3c7`), where the pre-fix image faulted at its first executed
zero-base site. `q_two` -O1 agrees; `dd2_join` -O1 running, then -O2, RV8, CoreMark, BEEBS.

**Two comparator lessons, recorded because both had a passing positive control:**
- compare-twins anchored the RV8 verdict regex on end-of-line while the runner's FAIL lines
  carry a log pointer, so an all-FAIL side parsed as "no summary". Its check arm had used a
  bare FAIL line -- a format the runner never emits. The control must use the producer's real
  output, not a hand-typed imitation (`f19eb1217e7b`).
- fuzz-check's crash controls were the C-20 reproducer, which the fix turned into an OK; a
  positive control has to be a bug that is still open, so they now use F-02.

**The rtl-oracle pass (Tier 4.1) changed the compiler's to-do list.** Beyond the DELIN
divergence already known: RTL raises on a TAGGED rs2 of cincoffset/scc/shrink and QEMU does
not, and a `ptrtoint` is a bare sub-register read, so `q + (long)p` at -O2 is
`cincoffset a0, a1, a0` with a0 the untouched capability -- a silicon-only fault in ordinary C
(**C-31**, pinned, fix in cycle 2: every c128->i64 truncate becomes PseudoTRUNC_CAP, an
integer write, kept a pseudo until MC lowering so copy propagation cannot delete it). MOVC
zeroes an UNTAGGED source on RTL (C-32). CAPENTER's funct7 in the `.td` (0100010) is decoded
by neither RTL nor QEMU (both 0001101) and CAPEXIT exists nowhere (**C-36**). `cap_get_tag`
traps on untagged on both (**C-33**). ldc of a LINEAR value clears the source granule and stc
nulls rs2 on RTL only. The hardware-side divergences (REVOKE's UNINIT cursor = start on RTL,
which RTL's own INIT rejects; INIT writing the LINEAR result to both rs1 and rd; TIGHTEN
raising on imm > 7 where spec and QEMU clamp, and not nulling rs1; the QEMU asserts; the
mcause+1 encoding) are in `CapstoneISASemantics.md` for hand-over.

**The C-40 audit** supported the root cause and the fix, refuted the fix's safety wording (a
ptrtoint-derived offset keeps its tag because the INSERT_SUBREG copy is elided when source and
destination share a register -- the pre-existing inttoptr hole), and corrected the README's
site counts (7 and 20, not five and eight) and "first loop" (first EXECUTED site). All applied.

**Cycle 2, written and object-compiled, waiting for the twins to release the toolchain:**
- Tier 4.5: `lowerSUB`, `lowerADD`, `lowerScalarI128{Shift,Logical,LogicalOnCapability,And,Mul}`
  and the four helpers only they used, 801 lines, deleted after `-Wunused-function` reported
  each static one on a solo recompile of the object (the member `lowerADD` had one mention: its
  definition). The object compiles with zero warnings after the deletion.
- C-31, C-33 (`lcc rd, rs, 1; sltiu rd, rd, 7`), C-36 (CAPENTER 0001101; CAPEXIT, its intrinsic,
  builtin, selector and clang lowering removed; the two retired encodings pinned INVALID in the
  disassembler suite; cap-control-flow.ll re-homed its mutation on `return`).
- Tier 4.6 Sema: `SemaCapstone.{h,cpp}` plumbed like SemaRISCV (Sema.h, Sema.cpp, CMakeLists,
  the SemaChecking dispatch for capstone32/64), two diagnostics; ranges: tighten 0..7, ccsrrw
  id in {0,1,2,4,16..31} (QEMU `helper_csccsrrw` asserts on anything else), shrink constant
  base < end, the inherited scalar-crypto ranges. Tests `Sema/capstone-builtins-range.c`,
  `Sema/capstone-inherited-riscv-ranges.c`; `capstone-tighten-range-diagnostic.c` is now a
  front-end -verify test, the backend route staying pinned by `fatal-tighten-range.ll`.
- After the build: the MC encodings for capenter re-derived from llvm-mc (predicted
  `[0x5b,0x95,0x05,0x1a]`), the MIR arm of cap-control-flow.ll updated for PseudoTRUNC_CAP,
  the c31 control's expectation set from the output, the manifest re-baselined for
  DiagnosticSemaKinds.td, X86/RISCV as before, llvm-stress, then the twins again.

### Execution log — cycle 2 grows by R-25, and two Tier 4 rows close without a fix (2026-09-05)

- **R-25 (INIT duplicates a LINEAR capability on RTL) is reachable from codegen.** The board
  lane verified the RTL reading against the other three `rd,rd` writebacks in the file (each
  guarded by `rs1 == rd`; INIT's is not) and asked whether codegen ever emits `rs1 != rd`.
  It does: INIT has no tied-operand constraint, so a source kept live after the builtin gets
  `init a1, a0, a1` at -O1 and -O2 (measured). No in-tree C calls `cap_init`, which is a fact
  about our programs, not about the silicon. Cycle 2 selects `cap_init` and `cap_seal` through
  tied pseudos (`PseudoINIT`/`PseudoSEAL`, rd = rs1), the ISA's own consume semantics; the MC
  instructions stay untied. Pinned by `sem-init-seal-tied.ll`, which fails on today's
  compiler exactly as the mutation header says.
- **The remaining `lcc rd, rs, 2` reads are gone**: the stack-shrink and global-shrink base
  reads and the realignment prologue used the cursor query; all three now use
  `PseudoTRUNC_CAP`. sp-derived operands are always tagged so none could trap, but Tier 4.2
  wanted no selector-2 query anywhere, and the pseudo also keeps copy propagation from
  handing SHRINK a tagged operand.
- **C-34 closes without a fix**: a zero-size alloca is laid out as one byte and its SHRINK
  covers one byte (`li a2, 1` in the sequence); a zero-size global gets no SHRINK at all. To
  be pinned by `sem-shrink-zero-size.ll` once the cycle-2 shapes are known.
- **Tier 4.3 closes without a fix**: a load through another capability is not moved across a
  `drop` at -O2 (the second load of the same address is re-issued after it); pinned by
  `sem-drop-orders-loads.ll` on today's compiler.
- **The gp-captable lead is dead**: the elided-copy mechanism needs -O1+ and produces a wrong
  tag, while that bug is present at -O0 and produces a wrong value (board lane, from the
  23-07 record). Not carried into cycle 2.
- `capstone/tests/lit-other-targets-baseline.txt` records the 32 X86/RISCV/Generic failures
  of the cycle-1 run so the next run is a diff, with what was verified marked as such.

### Execution log — RV8 -O2 7/7 on cycle 1; the domain-crossing model was wrong end to end (2026-09-05, ~01:00)

- **Twins on the cycle-1 compiler so far:** SLT AGREE at -O0/-O1/-O2 on all three files; RV8
  -O0 7/7 and **-O2 7/7 AGREE-PASS against 0/7 before** -- the five hangs and sha512's cause-5
  fault were C-28 (a sibling call falling off the end of its function) and norx's cause 24 was
  C-40; CoreMark AGREE. BEEBS -O0/-O2 running. "-O2 correct on QEMU" now holds for every suite
  that has reported.
- **The second rtl-oracle pass (CAP_CALL / CAP_RETURN / CAPENTER), all quoted in
  `CapstoneISASemantics.md`:** the compiler's `return` encoded rd = 0 and typed rs1/rs2 as
  capabilities, but spec, RTL and QEMU read the sealed-return capability from the rd FIELD and
  take rs1 as an integer (the re-entry PC) -- so every compiler-emitted return raised
  UNEXPECTED_OPERAND on the RTL (x0's capability slot is hard-zero) and took a different,
  trap-return-like branch under QEMU. The board-validated glue's hand-encoded
  `domreturn(t1, t2, x0)` has always had it right. Hardware saves, restores or scrubs NO general
  register across a synchronous call/return (PC plus seven CSRs are swapped, same list on both
  implementations); the compiler's call used an ordinary call's callee-saved mask, promising
  s0-s11 survive. QEMU asserts rd == rs1 on call and rs1 && rs2 on capenter; the RTL hardwires
  capenter's destination to a1. Cycle 2 now models all of it: `return rd, rs1, rs2` with the
  intrinsic `cap_return(cap, pc, code)`; codegen's call through a tied `PseudoDomCall`
  (`isCall`, `Defs = [C1]`, `getNoPreservedMask`); `capenter rs1, rs2` with fixed a0/a1 and the
  intrinsic `cap_enter(cap, i64)` returning a1. **OPEN C-36b:** no zeroing of non-argument
  registers and no gp save/restore around a domain call (the reference compiler does both);
  no in-tree C calls these builtins, so nothing shipping is affected yet.
- The tied pseudos have to be defined after the instructions they expand to (TableGen resolves
  `PseudoInstExpansion` names at that point): the first placement, after INIT but before SEAL,
  failed to build. Every changed object compiles cleanly; the full relink waits for BEEBS.

### Execution log — cycle 1 fully validated on QEMU; cycle 2 committed (2026-09-05, ~02:30)

- **Cycle-1 twins complete:** BEEBS -O0 81/81, -O2 78/81 AGREE-PASS. The three without an
  -O2 verdict were not the compiler's: `sqrt` failed to link (-O2 emits `__floatunsisf`,
  which the freestanding runtime's compiler-rt subset lacked; added to the shared list);
  `ctl-stack` and `ctl-vector` faulted with "Unaligned cap access" on a `stc` into the
  benchmark's own `static char heap[HEAP_SIZE]`, alignment 1, which the -O2 image's layout
  put 8-mod-16 and the -O0 image's 16-aligned -- a bump allocator handing out
  capability-holding memory without capability alignment; the array is now aligned to 16 in
  the source patch step. The runner printed no verdict line at all for either kind of
  failure; the comparator's MISSING (exit 2) is what surfaced them. So on cycle 1, every
  suite that has a verdict agrees at -O2 on QEMU.
- **Cycle 2 committed** as `357131bf` (C-31, C-33, C-36 including the domain-crossing
  operand models, the R-25 tie, the lcc-2 retirement, the gate keyed on messages),
  `ff5b20a5` (801 lines of dead i128 code), `4e94e3a2` (SemaCapstone,
  `-Wcapstone-pointer-roundtrip`, target macros, no lp64e, the ABI note; manifest at 71
  files). Gates on the tree: 103 LLVM-side and 21 clang tests as expected (XFAIL left:
  C-32, C-37, C-38), coverage 0 gaps, fuzz-check ALL OK, llvm-stress 897/900 with only
  F-02/F-03. The twins on cycle 2 are running.
- **What cycle 2 changed in emitted code, for anyone comparing images:** every `ptrtoint`
  and every pointer compare gains one `mv` (a self-move when the register is reused -- the
  write is the point; the combiner's re-formed `trunc(xor)` adds a second, harmless one in
  `hash_two`); the SHRINK sequences and stack realignment read the cursor with `mv`
  instead of `lcc ..., 2`; `cap_get_tag` is `lcc rd, rs, 1; sltiu rd, rd, 7`; `cap_init`
  and `cap_seal` overwrite their source register; a function making a domain call saves
  and reloads ra and s0-s11 around `call a0, a0` (13 spills -- the price of "the hardware
  preserves nothing" while honouring the C ABI for the caller); `return rd, rs1, rs2`;
  `capenter rs1, rs2` with fixed a0/a1; no `capexit`. The domain-call cost and the
  extra `mv`s are correctness-first choices; a use analysis that keeps the bare read for
  consumers that never check the shadow is a later optimisation.
- **Two build-order lessons:** a tied pseudo must be defined after the instruction its
  expansion names; and a virtual result on an instruction whose mask preserves nothing
  cannot be allocated ("ran out of registers") -- a call's result has to be a fixed
  physical register, copied out, like any return value.
- The results table's toolchain-identity lines used the executables' hashes; with shared
  libraries those never change. Corrected to the commit range and the four shared-library
  hashes from cycle 2 on.

### Execution log — the cycle-2 audit (2026-09-05, ~03:00)

The claim-auditor attacked the C-31 fix and the domain-crossing model. Outcomes:
- **C-31 mechanism SUPPORTED, with a matched pair:** the same post-PEI MIR with the truncate
  as `PseudoTRUNC_CAP` keeps `mv a0, a0` through to the object file; with the truncate as a
  plain `ADDI $x10, $x10, 0` the pre-emit MachineCopyPropagation (UseCopyInstr, via
  `isCopyInstrImpl`) deletes it. The auditor's own first control was vacuous --
  `-run-pass=machine-cp` builds the pass WITHOUT UseCopyInstr, so both variants survived
  it and read as "the addi is safe too". Only the real pipeline (`-start-after=prologepilog`)
  exercises the instance that deletes. Recorded here as another instance of the check that
  cannot fire.
- **RTL half UNSUPPORTED as cited:** neither `alu-write-clears-shadow.S` (different rs1,
  three nops) nor `cincoffset-stale-metadata.S` (rs1 already an integer) exercises
  `addi a5, a5, 0` on a live tagged a5 with the cincoffset next. The reading that it works
  (an ALU op never reads operand metadata; the shadow is written with the GPR write enable
  and '0 for a non-capability result -- `commit_stage.sv:279`, `issue_read_operands.sv:1900,1943`,
  not `cap_we`, which two documents had cited) is a reading. A one-arm addition was sent
  to the RTL lane; until it runs, C-31's silicon half is a prediction.
- **The C-31 test's CHECK-NOT was vacuous** (it scanned only the prologue; the fixed output
  legitimately contains `cincoffset a0, a1, a0` after the write). Removed; the ordering
  checks are the guard.
- **CAP_RETURN model SUPPORTED** (assembler and hand encoding agree, `.insn r` order settled
  by assembling both forms, decoder reads the rd field via the rs3 port).
- **CAP_CALL: mask, reserved registers and the ra reload SUPPORTED; "sp is preserved by
  convention" UNSUPPORTED** -- no such convention exists in-tree, and the only domain exit
  scrubs sp. Reworded as open beside C-36b, which now also names tp.
- **Residual recorded:** `copyPhysReg` emits nothing for a GPCR->GPR copy whose destination
  is the source's own address half (`CapstoneInstrInfo.cpp:695-697`) -- a bare read by a
  route other than the truncate; no IR found that reaches it from a tagged source.
- The other-targets lit run on cycle 2 reproduces the recorded 32-failure baseline exactly.

### Execution log — a serialization hole in the harness, closed before it fired (2026-09-05, ~03:20)

The twin drivers derived the rootfs lock path from `CAPSTONE_TMP_ROOT`, and the Tier 2b pair
runner gives each arm its own tmp root -- so a pair launched while the twins run would have
taken a DIFFERENT lock file and started a second QEMU against the shared rootfs, the exact
thing the "serialize QEMU suites" rule forbids. Found by reading the lock line before
launching, not by a collision. The lock is now one machine-wide path
(`CAPSTONE_QEMU_LOCK`, default `/tmp/capstone/nightly-qemu.lock`) in run-twin-suite.sh
and the fuzz campaign; run-slt-twin.sh gets the same line once its running instance
exits. Lesson, in one sentence: a lock that guards a shared resource must not be named
after a per-run directory. The Tier 2b sequence (W-16, W-17 on CoreMark; W-06, W-07, W-04,
W-05 on RV8; W-08 on SLT, all at -O2) is queued behind the cycle-2 twins on that lock.
Two CLASSIFICATION.tsv rows were corrected first: W-16 and W-17's OFF arms were written as
"(flag removed)", which the wrapper would have passed to clang as two words; they are now
`-foptimize-sibling-calls` and `-fjump-tables`, appended after the build script's own
flags so the later one wins.

### Execution log — C-31's silicon half measured (2026-09-05, ~03:40)

The RTL lane ran the proposed arm within the hour: `MOVC(a5, a6); addi a5, a5, 0;
CINCOFFSET(a7, a1, a5); CAPPRINT(a7)` inserted between ARM C and ARM B of
`alu-write-clears-shadow.S`. ARM D's CAPPRINT reached the log at cycle 362 and the run's only
exception is ARM B's UNEXPECTED_OPERAND at cycle 365 -- the positive control fired, so the
negative means something; the run ends at the +time_out cycle count because ARM B wedges the
core rather than trapping to mtvec, which the file places last for that reason. Both
untested mechanisms (an ALU op reading a register whose shadow says capability, and a
consumer issuing before the write settles) are covered by the one arm, since the cincoffset
is the very next instruction. C-31's fix is therefore measured on the RTL, not read. The
arm lands on the RTL lane's branch with the result in its commit message; the matrix and the
PseudoTRUNC_CAP comment now cite it. Cycle-2 SLT: 9/9 AGREE (dd2_join -O2 in).

### Execution log — the first csmith campaign, and what it found in its own harness (2026-09-05, ~04:00)

Ten seeds with both controls, on the cycle-2 compiler. Seeds 1-5 MATCH the native reference at
-O0 and -O2 (the first random-program agreement this project has had); seeds 6 and 8 GEN-SLOW
(native timeout / native link failure); the XOR control fired (CONTROL-OK). Then two harness
defects, both of the "silent" class:
- the fault control never ran: `fault_domain.c` defines `domain_main` itself and the build
  script linked the csmith entry wrapper as well -- a duplicate-symbol link error that the
  campaign skipped with `if rc == 0`. A positive control that does not build is now an
  ERROR that ends the run (exit 2); the build script has a `bare` mode for a source that is
  the whole domain.
- seed 7 at -O0 hung the guest (QEMU printed a `csdebugprint` of 0x1234: the domain executed
  bytes that decode as the debug-print instruction, so control left the code; the image has no
  such opcode, no switch, no indirect jump, and its whole stack demand is under 3 KB). The
  batch runner recorded ERROR without rebooting because QEMU was alive, and every later item
  errored in seconds against the dead guest. That is now a WEDGE verdict that reboots like
  FAULT. Seed 7 is a candidate F-04 and needs a solo run with a trace once QEMU is free; the
  rerun (queued) will also say whether -O2 wedges too.
Cycle-2 twins so far: SLT 9/9, RV8 -O2 7/7, CoreMark AGREE; BEEBS running.

### Execution log — the first two Tier 2b pairs (2026-09-05, ~04:20)

Run interleaved with the cycle-2 twins on the shared lock (one QEMU at a time, verified
with fuser), after the pair runner's first real invocation exposed an `unbound variable`
in its own `local` line -- the skeleton had never run end to end.
- **W-16 `-fno-optimize-sibling-calls`, CoreMark -O2: COMPILER-DEBT, retired by C-28.** OFF
  (sibling calls enabled) and ON both validate the CRC; the images differ. The flag masked
  the tail-call miscompile since June. Removal from the CoreMark and SQLite build scripts
  waits for one board rung on the current bitstream, per the plan's retirement rule.
- **W-17 `-fno-jump-tables`, CoreMark -O2: COMPILER-DEBT, still needed.** With jump tables
  enabled the domain faults at the dispatch (cause 24, "Cap mem access requires capability",
  rs1 = x10): the jump-table lowering loads the target through an integer register. The pin
  stays; the fix (a capability base for the table load, or no-jump-tables as the target
  default) is a cycle-3 item.
- The CoreMark comparator read that fault as "no summary": run-coremark.sh writes its failure
  lines to stderr, so stdout held only the build's lines. Any non-empty CoreMark summary
  without the marker is now FAIL (an empty one stays no-verdict), with a check arm.

**Cycle-3 design note, jump tables (W-17).** The fault is at the table LOAD, not the jump:
`lowerJumpTable` returns the table's capability (`getAddr`), but the generic BR_JT expansion
computes `Table + Index*4` in the AS0 pointer type (i64) and loads the EK_Custom32 entry
through that integer -- "Cap mem access requires capability, rs1 = x10". The branch itself
(`PseudoBRIND` -> `jalr x0`) is an integer jump within PCC and is fine. The fix is a custom
BR_JT lowering: `getMemBasePlusOffset(TableCap, Index*4)` for the entry load (a `cincoffset`
on the capability, then `lw`), the 32-bit absolute target to `brind`. About thirty lines plus
a test with a 10-way switch at -O0 and -O2 and the W-17 OFF pair as the QEMU verdict. Worth
doing rather than defaulting to no-jump-tables: SQLite's VDBE dispatch is one big switch and
the pin turns it into a compare chain.

### Execution log — the csmith campaign rerun: instrument clean, one real -O0 finding (2026-09-05, ~04:45)

With the bare-mode fault control and the WEDGE-reboot path: 18 items, both positive controls
fired (the XOR build MISMATCHed, the fault domain FAULTed and the batch rebooted and ran the
item after it), 15 of 16 program runs MATCH the native reference at -O0 and -O2 (seeds 1-5,
9, 10 at both levels, seed 7 at -O2), seeds 6 and 8 skipped as GEN-SLOW. The one non-match is
now attributable: seed 7 WEDGES at -O0 and matches native at -O2 -- an -O0-only defect, the
class the plan said -O0 has (the CoreMark crcu8 byte-spill tag clear is another). Filed as
F-04 with the source, image hashes and the trace facts; the next step is a solo run with an
instruction trace and a per-function optnone bisection run as one batch.

### Execution log — cycle-2 twins: every suite agrees at -O2 on QEMU (2026-09-05, ~05:00)

On the cycle-2 compiler (commits 357131bf..8a155cc7): SQLite SLT AGREE with native at -O0,
-O1 and -O2 on select1 (1031 records), q_two and dd2_join; RV8 -O0 7/7 and -O2 7/7; CoreMark
-O0/-O2 AGREE; BEEBS -O0 81/81 and **-O2 81/81 AGREE-PASS**, including ctl-stack,
ctl-vector and sqrt, which had no verdict in cycle 1 (the heap alignment and the soft-float
routine were the benchmark's, not the compiler's). That is the plan's bar -- "-O2 correct on
QEMU" -- met on every execution suite the project has, against a starting point of SQLite
faulting at -O1 and -O2, RV8 0/7 at -O2, and 60 lit tests none of which ran at -O1.
Silicon remains the other half of the bar and is the board lane's: the images to compare are
identified by the shared-library hashes in results/2026-09-05.tsv.

### Execution log — the last four pairs, and F-04 is not what it looked like (2026-09-05, ~05:30)

- **W-07 shrink-stack, W-04 memcpy-high-half-fixup, W-05 memops-via-libcall (RV8 -O2): 7/7
  AGREE each**, images differing. W-07 joins W-06 as a silicon-debt candidate for a board pair;
  W-04 is silicon-debt whose deletion is already authorised (S-06 fixed in the RTL) pending the
  one board rung; W-05 is not a workaround and the two lowerings agree on this corpus.
- **W-08 merge-string-constants (SLT -O2): IDENTICAL IMAGES** -- the knob does not change the
  QEMU SLT build at all, so the pair runner aborted without a verdict, as designed. The pair
  belongs in the silicon SQLite config, where the knob is turned on.
- **F-04 retraction in progress.** The bisection batch's -O0 reference of cs7 RETURNED the native
  checksum as the second domain of a fresh boot, while the campaign's cs7-O0 wedged twice as the
  TWELFTH domain of its boot; the two -O0 images also differ in bytes (being checked: code or
  strings). So "an -O0-only miscompile" was one step past the evidence -- the wedge may be a
  per-boot state effect (the twelfth domain) rather than the image. A position test is running:
  the campaign's own cs7-O0.dom first and last in one boot, and cs7-O0 / cs7-O2 each after
  eleven fillers.

### RETRACTION — F-04 is not a compiler finding (2026-09-05, ~05:45)

The position test: cs7-O2 (which had passed as the first item after a reboot) WEDGED as the
twelfth domain; the campaign's own cs7-O0 image passed as the FIRST domain of a boot; cs2-O2,
which had passed in every earlier batch, wedged as the fifth; and a further boot never
reached its login prompt. The wedge follows the guest's state within a boot, not the image or
the -O level; every instance ends with QEMU's `Print = Scalar(0x1234)`, which also appears in
a normal boot. "An -O0-only miscompile" was one step past the evidence -- it rested on one
-O0 wedge and one -O2 pass, and the pass had been the first item after a reboot. F-04's folder
now records the retraction with the table of runs; the symptom goes to the board lane as a
runtime item. What would have caught it earlier: rerunning the wedged image as the first
item of a fresh boot before writing "-O0-only" -- the batch runner makes that a one-line
manifest. The campaign now states that a WEDGE is not a compiler verdict until reproduced
first-in-boot.

### Execution log — the board regression session: preparation (2026-09-05, ~04:45)

The project lead's decisions before the session: gp-captable becomes the silicon default and the
QEMU-default path drops its DELIN (a cycle-3 code change, not made before the board measures the
cycle-2 images); board stacking is control + at most three unknowns per boot after three clean
boots, ascending, read no further than the first failure; domain calls scrub and save gp/tp in
the compiler (cycle 3); jump tables are the first cycle-3 fix; the board lane's R-20 pair is
included. The board lane then corrected its own request to a SINGLE arm (rc_p1, rc_const0 as the
matched control): the compiler-side R-20 workaround was reverted on 2026-08-10 (cdbb92360e2b).

**Lineage, answered by git rather than assumed.** The resident bitstream
`caplifive_s12fix_5097eb166.bit` is RTL commit 5097eb166 ("S-12 fix: gate the FPR arm of the WAW
escape on commit_ack too", 2026-09-04). The r20-fix branch tip 2efb3604f ("Fix R-20: keep the
CAPENTER x10 clobber additive", 2026-08-10) is NOT its ancestor; merge-base e1b3db6ba
(2026-08-08). So the resident silicon carries the S-12 fix and not the R-20 fix, and with the
compiler workaround retired **R-20 is unmitigated on the board** — the board lane's consequence,
which neither of us had followed through: an address-shaped wrong answer on any rung now has two
candidate causes, so the R-20 repro runs as a second control in the first boot. *[Superseded
— see the RETRACTION further down: the fix IS in 5097eb166 as the cherry-pick f623c48a1;
"provably lacks" was an ancestry-by-hash claim.]*

**The rung set**, all silicon-config builds (`build-ladder-domain.sh`: gp-captable, shrink off,
no jump tables) with the cycle-2 compiler through `verify-and-stage-rung.sh`, i.e. every image
was QEMU-verified against its native oracle and carries a `.qemu-pass` marker before it costs a
boot; every rung at its own `DOMAIN_BASE_VA` (k800 keeps 0x10000, the hash its marker vouches
for):

| rung | VA | level | oracle | note |
|---|---|---|---|---|
| k800 | 0x10000 | -O0 | 4 | the known-good control, image 40d765da… unchanged |
| r20sbx | 0xb0000 | -O0 | 0xD0000000 | R-20 repro, rebuilt (below) |
| s06agg | 0x20000 | -O0 | 15 | S-06 confirmation, no memcpy fixup |
| beebs_bs, beebs_crc32 | 0x30000, 0x40000 | -O2 | 887447230, 1703161001 | published rungs are -O1 |
| rv8_primes | 0x50000 | -O2 | 99991 | the old -O1 silicon hang |
| coremark_matrix | 0x60000 | -O2 | 14343 | needs the 32 KiB window (.text 0x2443 B > 4 KiB); the W-16 rung, sibling calls on |
| rc_const0, rc_p1 | 0x70000, 0x80000 | -O0 | 2016, 2080 | the board lane's reconstruction, rebuilt at distinct VAs |
| csm4, csm7 | 0x90000, 0xa0000 | -O2 | 0xA988DFF7, 0x1E21A964 | csmith seeds 4 and 7 in the silicon ABI via `csm_ladder.h` (csm7 with the 32 KiB window); both match their campaign checksums under QEMU |

Plus the SLT arm: `sqslt1.dom` = the SQLite silicon build at **-O1** (sha c01e6b89cad0f17a) with
its host (a1895d35f768b5d0), QEMU-verified on `q_two.test` (records=2, completed=1) and, while
the board ran, on `select1.test`; it doubles as S-12 draws on the fixed bitstream (the S-12
compiler pass is inert on c128 code, so "workaround off" needs no flag).

**Two things the preparation found.** (1) `beebs_bs -O2` at 0x30000 produced NO retval on its
first QEMU verification (the 120 s timeout, only kernel boot lines in the log). Bisection:
-O1 at the same VA, -O2 at the default VA, -O1 default, -O0 at 0x30000 all pass; the exact
failing arm then passed 2/2 reruns plus a 0x40000 arm, and the re-verification passed —
4/4 on the identical image after one no-result. Consistent with the per-boot guest wedge F-04
documents, not with the image; it stays in the sweep, last in its boot. (2) The R-20 repro's
frozen `sbx8.dom` is linked at 0x10000, the control's VA (R-3), so it was rebuilt from the
frozen sources at 0xb0000. The main-checkout and cycle-2 compilers build the IDENTICAL image
(e3b38a4429d2), which differs from the frozen one in 19 of 142 `sbx_compute` instructions —
all outside the inline-asm arms: frame immediates, and the arm-result spill `stc/ldc` →
`sd/ld` (base s0, not a0). The board lane ruled it admissible on the package's own criteria
(triple opcodes/registers/separation unchanged) with an asymmetry that the result line
carries: 0xD0000001 (R-20 live) is conclusive; 0xD0000000 on a rebuilt draw is not.

**Images.** The overlay gate (C9) blocks an image carrying a 1.25 MB domain the boot does not
use, so two images: the ladder image (11 doms, sha 4a5677c73b0eedd8, 15.4 MB, every staged file
byte-verified inside `rootfs.cpio`) and an SLT image baked after the ladder boots. Firmware
embeds the 65536-node DTS (marker check by hand; the gate prints "unknown bitstream" for the
s12fix name). Boot plan: B1 k800 + r20sbx + s06agg; B2 k800 + beebs_crc32; B3 k800 + rv8_primes;
B4 k800 + rc_const0 + rc_p1 + coremark_matrix (the pair ahead of the riskier 32k-window rung,
at the board lane's suggestion); B5 k800 + csm4 + csm7 + beebs_bs; then one SLT arm per boot.
Results: `capstone/tests/board-results/2026-09-05.tsv`.

### Execution log — the board regression session: the ladder boots (2026-09-05, ~05:10)

Five boots on `caplifive_s12fix_5097eb166.bit`, ladder image 4a5677c73b0eedd8, control first
in every boot and returning 4 every time. **Every rung returned its native oracle.** Result
lines are in `capstone/tests/board-results/2026-09-05.tsv`; the summary:

| boot | rungs (position order) | readings |
|---|---|---|
| B1 | k800, r20sbx, s06agg | 4; **0xD0000000** (R-20 NOT observed on the rebuilt draw — not conclusive, see below); 15 |
| B2 | k800, beebs_crc32 -O2 | 4; 1703161001 |
| B3 | k800, rv8_primes -O2 | 4; 99991 (6.32 M cycles) |
| B4 | k800, rc_const0, rc_p1, coremark_matrix -O2 | 4; 2016; **2080**; 14343 |
| B5 | k800, csm4 -O2, csm7 -O2, beebs_bs -O2 | 4; 0xA988DFF7; 0x1E21A964; 887447230 |

What that settles for the compiler: the cycle-2 `-O2` images that agreed with `-O0` on QEMU
also compute correctly on silicon in the silicon configuration — BEEBS bs and crc32, RV8 primes
(the rung that hung at -O1 on older silicon), CoreMark matrix **with sibling calls enabled**
(C-28's tail-call fix runs on the board, so W-16's `-fno-optimize-sibling-calls` pin can be
retired), and two csmith programs whose checksums match the native reference. `s06agg` returns
15 with no memcpy high-half fixup, re-confirming W-04's deletion. The project lead's bar —
"-O2 correct on QEMU and silicon" — is met for every rung that was run.

What it says about R-20: the R-20 pair's test arm `rc_p1` (the July "array store + live
accumulator" shape, the board lane's reconstruction, rebuilt at 0x80000 with the cycle-2
compiler) computed 2080 on a bitstream that provably lacks the R-20 RTL fix, and the rebuilt
R-20 repro read 0xD0000000 in B1. Neither is conclusive on its own (the rebuilt repro is a
weaker probe; the pair is a reconstruction), so the frozen `sbx8.dom` runs next at its own
0x10000 base with the control relinked to 0x20000 — the board lane's resolution of the R-3
collision ("relink the control, not the artifact", now in the board-run skill together with
the entry-point check; that check was run over all three image sets of this session and its
negative control fired). Prediction written before the boot: 0xD0000001; a 0xD0000000 with a
valid control would be evidence that the s12fix lineage cures the x10 forwarding path by a
route other than 2efb3604f. *[Superseded — see the RETRACTION below: the fix is present as
the cherry-pick f623c48a1, so the prediction's premise was wrong and the "another route"
hypothesis is struck.]*

### RETRACTION — the resident bitstream DOES carry the R-20 fix (2026-09-05, ~05:15)

The two entries above say the resident `caplifive_s12fix_5097eb166.bit` "provably lacks" the
R-20 RTL fix, on the strength of `git merge-base --is-ancestor 2efb3604f 5097eb166` answering
no. That answered a question about a HASH. `git log e1b3db6ba..5097eb166 --
core/issue_read_operands.sv` lists **f623c48a1 "Fix R-20: keep the CAPENTER x10 clobber
additive instead of overwriting"** (authored 2026-08-10, committed onto the s12 lineage
2026-08-11 13:44 +0800 — the committer date is when it entered this lineage) — the same fix
as 2efb3604f, change lines identical (the board lane diffed them) —
and `git show 5097eb166:core/issue_read_operands.sv` carries the additive form at lines
573-580 with the old overwrite described as the bug. The board lane had "verified
independently" by the same ancestry test, which is why the wrong claim looked confirmed twice.

Consequences. R-20 is FIXED on the resident silicon, so the session's three R-20 readings —
r20sbx 0xD0000000 (B1), rc_p1 = 2080 (B4), the FROZEN sbx8 0xD0000000 with a valid control
(B6) — are the package's own "fixed" reading and need no new hypothesis; "the s12fix lineage
cures the x10 path by another route" is struck. The prediction "0xD0000001" was wrong because
its premise was. Nothing about the compiler changes: the pair and the repro were controls.
The TSV header and the B1/B4/B6 notes carry the correction beside the original wording.

What would have caught it, and the rule worth keeping: **ancestry by hash is not presence by
content.** A rebased or cherry-picked fix is present without being an ancestor. Before writing
"lacks commit X", search the range for the fix by subject (`git log --grep`) or by the cited
lines (`git show <rev>:<file>`), and quote the line. Recorded as a memory note; not a
CLAUDE.md change (the "clean result is not evidence" section already covers the shape —
the instrument answered a different question from the one asked).

### Execution log — the board regression session: the frozen repro and the SLT arm (2026-09-05, ~05:35)

Three more boots, control (the 0x20000 build, 589ceee3853c6092) first and returning 4 in each:

| boot | arm | reading |
|---|---|---|
| B6 | frozen `sbx8.dom` (91499d57…, untouched at 0x10000) | **0xD0000000** — the package's "fixed" reading; see the retraction above |
| B7 | `sqslt1` -O1, `q_two.test` | records=2, stmt_fail=0, query_fail=0, completed=1 |
| B8 | `sqslt1` -O1, `select1.test` | records=1031 (stmt_pass=31, query_pass=1000), 0 failures, completed=1 |

B8 is the first SQLite validation on silicon above -O0: 1031 SQLLogicTest records at -O1 in
the silicon configuration, matching the QEMU run of the same image byte for byte (the
select1 verification rebuilt the image and its hash was unchanged, c01e6b89cad0f17a). It is
also two S-12 draws on the S-12-fixed bitstream, both clean. The SLT boots needed the
relinked control because the SQLite domain, like every domain built without
`DOMAIN_BASE_VA`, enters at 0x10000 — the board lane measured 24 of 40 committed images at
that address and wrote the rule and a one-line entry-point check into the board-run skill;
run over this session's three image sets it reported no collision, and its negative control
(the 0x10000 control beside the frozen image) fired.

**Session totals: 8 boots, 8 valid controls, 19 rung readings, every rung at its oracle**, all
result lines in `capstone/tests/board-results/2026-09-05.tsv`. The -O2 bar is met on silicon
for every rung run; the -O1 SQLite domain is silicon-clean; C-28 is confirmed on the board;
W-04 and W-16 can be retired; R-20 is fixed in the resident hardware and the July gp-captable
miscompute pair computes correctly. One retraction (the lineage claim), recorded with its
cause and its rule.

Not run this session, deliberately: the S-04 `sm0`/`sm` memcpy pair (no sources survive; a
reconstruction is a separate task), and the W-08 SQLite -O2 pair in the silicon
configuration (the -O2 SQLite domain was not built for the board — the SLT twin at -O2 is
QEMU-only so far). Next on silicon: the cycle-3 images (jump tables, C-36b, the
gp-captable-default/DELIN change) through the same rung set, which now has a recorded
silicon baseline to compare against.

### Execution log — merged to dev (2026-09-05, ~06:05)

dev took the branch by fast-forward at 628adf8d0e80 (the lane that owns the main checkout ran
the merge and pushed dev), after the merged tree was validated: Capstone lit 106/106, BEEBS
janne-complex at -O0 and -O2 under QEMU, and the SLT twin select1 at -O2 (AGREE, 1031 records,
0 failures, the same image hash as the cycle-2 run since the merge moved no compiler file).
The shared-patch drift gate fired on the merged tree for the two Sema files cycle 2 added
without re-baselining the manifest — the gate working — and the manifest was re-baselined in
its own commit. dev now carries the cycle-2 compiler; the default ABI is unchanged (the
gp-captable default and the DELIN drop are cycle 3), and the one visible change for other
lanes is the pointer-round-trip warning, which fires in SQLite's amalgamation as a warning.

### Execution log — cycle 3, item 1: jump tables (W-17), and the .rodata wall (2026-09-05, ~07:30)

**The design note above was half right.** The table LOAD did go through an integer (the
generic BR_JT expansion computes `table + index*4` in the AS0 pointer type), and a custom
`lowerBR_JT` with a capability base (LGA -> PseudoCapGlobalBase, `cincoffset`, `lw`) fixes
that half. Two things it did not foresee:

1. **Entries must be label differences.** With the capability base but the old absolute
   `.word .LBB` entries (EK_Custom32), CoreMark's `core_state_transition` dispatch still
   failed — an instruction access fault AT the target, cause 1, pc = 0x13c08, the link-time
   address. A domain does not execute at its link address; every other code reference the
   backend makes is PC-relative and works. `getJumpTableEncoding` now returns
   EK_LabelDifference32 unconditionally and the dispatch adds the table's runtime address
   (the cursor of the table capability, `TRUNCATE c128 -> i64`) before `jr`. RV8's seven
   benchmarks had passed with absolute entries in the same intermediate build, which is the
   counterexample the auditor was asked to attack (verdict below).
2. **Under gp-captable no capability reaches .rodata at all**, so no table lowering can work
   there: gp is bounded to the cap table and only globals with a slot are addressable. The
   build-script comment said so; the premise was then measured rather than trusted, with a
   hidden knob (`-capstone-gp-captable-jump-tables`) that lets a table through: the dispatch
   load faults OOB at the table's own address (cause 5, rs1 = x10, cursor 0x10156105c), while
   the same source with the table refused returns the native 7419. So `areJTsAllowed` refuses
   jump tables under that ABI, which makes the twelve `-fno-jump-tables` pins redundant on
   the silicon path and is honest about the state: SQLite's VDBE dispatch stays a compare
   tree on silicon until tables live in cap-table-managed data (a descriptor slot like a
   global, initialised by the glue's copy path). **That relocation is a scope item for the
   lead**, not something this session decided.

**The same wall takes cttz.** C-20's fix uses the generic de Bruijn lookup table, a constant
pool in .rodata. Under gp-captable it faults OOB at -O0 and -O2 (cursor 0x1015a1047), never
exercised by a board rung because no rung has a `cttz`. Under that ABI cttz now lowers to
`popcount(~x & (x-1))` (arithmetic, no memory; only ISD::CTTZ is Custom, never ZERO_UNDEF,
because the generic code rewrites each into the other when the other is LegalOrCustom and
marking both would loop); the default ABI keeps the table. Controls after the fix: 41 = native
at -O0 and -O2 under gp-captable and in the default ABI. Any other constant pool under
gp-captable is the same latent fault; none exists in the corpus today (the SQLite -O1 domain
ran 1031 records on silicon), which is luck, not a guarantee — same scope item.

**Red first.** `jump-table.ll` failed on the cycle-2 compiler (the `%hi(.LJTI)` integer
path) and pins, at -O0/-O1/-O2, `auipc %pcrel_hi(.LJTI)`, `cincoffset gp`/`delin`,
`cincoffset`, `lw`, `add`, `jr`, `.word .LBB-.LJTI`, plus the gp-captable arm (no table) and
the knob arm (table back). `c20-cttz.ll` gained a TABLE arm (constant pool present in the
default ABI) and CAPTABLE arms (absent). The coverage gate lists the new flag; 0 gaps.

**QEMU verdicts on the final compiler (lib 57b5c5846ec3):** Capstone lit 107/107; the W-17
pairs at -O2 with jump tables ON: CoreMark AGREE-PASS, RV8 7/7 AGREE-PASS; a default-ABI
switch domain 7419. **Pins retired** with a dated note each: `-fno-jump-tables` in
build-coremark-capstone.sh (5), the five rv8 scripts, build-beebs-newlib-log-capstone.sh,
multi-tu-slot-collision.sh (2), and — redundant under the backend rule — build-sqlite-silicon.sh
and build-ladder-domain.sh (a silicon rung, k800, is byte-identical with and without the
flag: b2d60e525f807ea4). Left alone: build-ladder-base-fpga.sh (the plain-riscv64 baseline
half) and the two frozen copies under fpga-repros/. **W-16 retired too**: its board condition
(an -O2 CoreMark image with sibling calls on, on the current bitstream) was met by the
coremark_matrix rung in B4 (14343, control 4), so `-fno-optimize-sibling-calls` left
build-coremark-capstone.sh (2), build-sqlite-capstone.sh and build-sqlite-silicon.sh.
CLASSIFICATION.tsv: W-17 FIXED (cycle 3), W-16 FIXED (C-28), pin retired.

**After the retirements:** CoreMark -O0 and -O2 through the final script PASS (one boot-login
infra flake on the way, retried by the harness; the image was byte-identical to the
W-17 OFF-arm image that had passed, d6a791796abc0aa5); BEEBS newlib-log -O0 PASS; the
multi-tu probe prints its designed "separate objects collide; LTO does not" summary
(exit 1 is that probe's documented result, unchanged). Two of my own invocations were wrong
and are rerun through the twin runners (RV8 -O0 had the host binary outside the guest share;
the SLT runner wants an absolute test path) — results below, together with BEEBS -O2 as the
wide check and the auditor's verdict on the absolute-entry mechanism.

**Audit of the jump-table claims (claim-auditor, 2026-09-05, ~07:45)** — three refutations
and two blocks, all acted on before the commit:

- *Absolute entries.* The CoreMark cause-1 fault was **recalled**, not recorded: the pair
  runner wipes its tree, so the artifact was gone. And my RV8 "control" (7/7 with absolute
  entries) was **void**: the compiler was rebuilt mid-run, and RV8's only tabled benchmark
  (miniz, last in the list) was built after the swap. So the mechanism was re-established
  with a kept artifact: the compiler temporarily emitting absolute entries (EK_BlockAddress),
  a default-ABI switch domain whose table reads 0x00010268, 0x000103a4… (link addresses),
  and the run halting with **cause 1 at pc = 0x10338, one of the table's own entries** —
  log under `/tmp/capstone/board-cycle2/absentry/`, outside any runner's wiped path. QEMU's
  own trace fixes the link-vs-runtime relation: the dispatch load at link 0x102b4 executes
  at "pc = 1015602b4, pcc_base = 101560000". Recorded, not recalled.
- *The guard could not fire.* `CHECK-NOT: .word .LBB0_{{[0-9]+}}$` — a bare `$` is a literal
  dollar to FileCheck. Fixed to `{{\.word \.LBB0_[0-9]+$}}` and checked both ways: an
  absolute entry in the input → exit 1, label differences only → exit 0.
- *A comment contradicted the code* (lowerBR_JT still described absolute entries) and
  `LowerCustomJumpTableEntry` was dead; both gone.
- *"No change outside gp-captable" was false*: marking CTTZ Custom unconditionally made the
  generic expansion route CTTZ_ZERO_UNDEF through CTTZ and add a zero check to the default
  ABI's zero-poison form. CTTZ is now Custom only when gp-captable is on; `cttz64_zu` in
  c20-cttz.ll pins the absence of that check in the default ABI (its mutation is the
  unconditional action).
- *Wording*: "no capability reaches .rodata" is stronger than the evidence (whether `sp`'s
  half of the split spans .rodata is unresolved). Everything now says "no capability the
  compiler can derive a global address from reaches .rodata".

**Two process slips of my own, same day, same class**: the compiler was rebuilt while a
QEMU suite was mid-run twice (the RV8 pair above; then a BEEBS twin that started building
under the absolute-entry experiment compiler, killed by PID tree and rerun). A shared-libs
build swaps the compiler under a running suite silently. Saved as a memory note; the rule
was already in my constraints and was not followed.

**Final validation, all on the committed-state compiler after its last rebuild (~08:20):**
Capstone lit 107/107; coverage gate 0 gaps; the six QEMU controls at their expected values
(cttz 41 under gp-captable at -O0/-O2 and in the default ABI; forced table OOB; refused
switch 7419; default-ABI switch with tables 7419); CoreMark -O2 PASS through the final
script; RV8 twins 7/7 at -O0 and -O2; SQLite SLT twins with sibling calls on, q_two -O1
AGREE (2 records) and select1 -O2 AGREE (1031 records, 0 failures); BEEBS -O2 twin 81/81;
BEEBS newlib-log -O0 PASS. The board lane filed the .rodata finding as **C-43** with the
sharper line — named globals get cap-table slots, anonymous compiler-generated data (.LJTI,
.LCPI) does not; the -O1 SQLite silicon domain's 48 KB of named .rodata is reached and it
has 0 LCPI labels — and the code comments now say it that way. Committed as one change:
lowering, refusal, cttz, tests, coverage list, classification rows, retired pins.

**Record hygiene after the commit (~08:45).** The commit's "all after the last rebuild"
was true of everything except three runs that had happened on the intermediate build
(before the 07:41 rebuild that restored the default ABI's cttz expansion): the RV8 -O0 twin,
BEEBS newlib-log -O0 and the multi-tu probe. Re-run on the final compiler: RV8 -O0 7/7,
newlib-log -O0 PASS, the probe its designed summary. The workaround results file now marks
its 06:21–06:43 W-17 rows as intermediate-build rows superseded by the 06:58 ones, so they
cannot be read as evidence, and the last stale jump-table sentence in the CoreMark build
script is gone. Compiler identity note: the final library hash is ae821a017089; the
57b5c5846ec3 quoted in the W-17 classification row's evidence is the build immediately
before the cttz action was restricted to gp-captable (the QEMU controls and twins were run
on both; the lit and control results are identical, and the row now carries both).
