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
