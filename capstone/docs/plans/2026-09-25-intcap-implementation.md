# intcap implementation plan: the copy rule, the type, the back end, PostgreSQL

*Status: APPROVED 2026-09-25, execution started. Companion to `2026-09-25-intcap-for-postgres.md`
(the analysis and M2b measurement) and `intcap-uintptr-model.md` (the ABI-wide option). Branches:
`compiler/movc-live-source-copy` (Phase A), `compiler/intcap` (B, C), `postgres/10-intcap-datum` (M0, D).*

## Context

PostgreSQL's backend boots in a Capstone domain up to the first `Datum`, and stops there
(`postgres/9-boot-attempt`, `f9b7c9ffa746`). `create pg_proc` faults at `GetTableAmRoutine+0x38`,
because `Datum` is `uintptr_t` (8 bytes), and every pointer the fmgr passes through it loses its
tag. The counts are 781 `PointerGetDatum`, 729 bare `(Datum)` casts and 4750 conversions. This
cannot be patched in PostgreSQL. It needs CHERI C's `__intcap`/`__uintcap_t`.

The decisions come from `plans/intcap-datum` (`capstone/docs/plans/2026-09-25-intcap-for-postgres.md`):
- **Stage 1 is the type only.** `uintptr_t`, musl and the other ports stay unchanged. PostgreSQL
  opts in with `typedef __uintcap_t Datum`.
- **No ISA change for QEMU.** Arithmetic on an intcap becomes a type dispatch.
- **No RTL change for silicon.** `MOVC` nulls an untagged source on silicon (C-32). A compiler rule
  copies a register whose source is still live through a stack slot (`STC`+`LDC`). This is measured
  at 30,625 of 62,757 capability copies on CPython's core, 3.4 % of static instructions. It sits
  behind a SubtargetFeature, so a later RTL fix (Q-04 b) is a flag flip.

The outcome: PostgreSQL `--single` runs the survey workload in a domain on QEMU with output
byte-identical to native. It also does so under the RTL's `MOVC` rule
(`CAPSTONE_MOVC_NULL_SCALAR=1`). The copy rule also closes the C-32 class today (SQLite lookaside,
musl `iconv_open`).

## Step 0 — land what this stacks on (DONE 2026-09-25: C-66 at `82c42289c2e7`, cpython/9 at `5806ee42c0dd`)

- **C-66** (`compiler/c66-machinecse-pre-trapping-cap-arith`, local) is a prerequisite: the tag
  dispatch is a trap under a guard.
  - Its gates are done: lit 107/108, X86+Generic 0 new failures, HostCall 28/28.
  - The four red nightly suites are shown pre-existing: `shared-patches` is dev drift, `dwarf` is a
    missing tool, `beebs` is host headers (identical on dev), and `linear-uninit` has an identical
    fault signature on dev.
- Commit the keyed-cache change, the tests, ISSUES and the manifest. Run precommit-scan, then push.
- Then push `cpython/9-sublet-interpreter`: 0014 v2, the guard and the results README, after
  finishing its README.

## Order (decided)

1. Step 0.
2. **Phase A first.**
3. M0 runs alongside A. It needs no compiler of ours, only the CHERI SDK, while A's builds and
   tests run.
4. Then B, C and D, each tested under the silicon `MOVC` rule from the start.

## Branches and worktrees (one per change, stacked)

Before creating each worktree, check that the path does not exist.
- **W1:** `/home/biecho/llvm-capstone-movc-copy`, branch `compiler/movc-live-source-copy`, from the
  C-66 tip. It holds Phase A.
- **W2:** `/home/biecho/llvm-capstone-intcap`, branch `compiler/intcap`, from W1. It holds Phases B
  and C.
- **W3:** a sparse checkout of `capstone/` at `/home/biecho/llvm-capstone-pg-intcap`, branch
  `postgres/10-intcap-datum`, from `origin/postgres/9-boot-attempt` with W2 merged in. It holds M0
  and Phase D, and builds with W2's compiler.
- Build dirs mirror `llvm-capstone-c66/llvm/cmake-build-release`: Release+Asserts, `Capstone;X86`,
  clang-18, lld, `ninja -j24`, and the tools lit needs (`count`, `not`, `llvm-config`, `llvm-readobj`,
  `llvm-dwarfdump`).

## M0 — measure PostgreSQL under intcap first (W3, about 2 days)

1. Cross-compile PostgreSQL 17.5 (`/tmp/capstone/pg-mmgr-host/pg.tar.bz2`) for CheriBSD
   riscv64-purecap with the local CHERI SDK: `/home/biecho/cheri/output/sdk/bin/clang-17`, sysroot
   `sdk/sysroot-riscv64-purecap`.
   - Use the flags of `capstone/ports/common/cmake/toolchains/cheribsd.cmake`:
     `-march=rv64imafdcxcheri -mabi=l64pc128d -mno-relax`, and `--host=riscv64-unknown-freebsd13`.
   - Use cheribuild's postgres configure args (`--without-libxml --without-readline --without-gssapi`).
   - Add `-Wcheri-provenance -Wcheri-capability-misuse -Wcheri-bitwise-operations -Wshorten-cap-to-int`,
     with no `-Werror`.
2. Commit result lines only, to `capstone/ports/postgres/single-user/results/<date>/intcap-m0.txt`:
   - counts per diagnostic;
   - the ambiguous-provenance sites;
   - the other capability-misuse sites;
   - the 13 `SIZEOF_DATUM == 8` branches, each classified.

   This is Phase D's work list, and it replaces the port estimate.

## Phase A — the live-source copy rule (W1, about 1.5-2 weeks)

After review, the rule lives in one late pass that decides on final liveness, so every earlier pass
may keep treating `COPY`/`MOVC` as a pure copy.

1. **Feature** (`CapstoneFeatures.td`): `FeatureMovcKeepsIntegerSource`, spelled
   `"movc-keeps-integer-source"`. It is off by default, so the rule is on. A fixed bitstream passes
   `+movc-keeps-integer-source`. The generated getter is `STI.movcKeepsIntegerSource()`.
2. **`CapstoneMachineFunctionInfo.{h,cpp}`:** add `int LiveSourceCopyFrameIndex = -1` with getter
   and setter. Copy it in `clone`, and add a YAML field for MIR tests (pattern: `varArgsFrameIndex`).
3. **Shared helpers in `CapstoneInstrInfo`:**
   - `isLiveSourceCopyCandidate(MI, MF)`: a COPY or MOVC with both registers in GPCR, source ≠
     destination, and a source that is not exempt. Exempt means `c0`, `sp`, `gp`, `tp`, `fp` if
     `hasFP`, and `bp`/x9 if `hasBP`. Do not use `MRI.isReserved()`, because `-ffixed-xN` registers
     can hold integers. The tp exemption rests on tp always being tagged; comment it, the way musl's
     `pthread_arch.h:38` already relies on it.
   - `needsLiveSourceCopySlot(MF)`: the rule is on and some candidate exists. This is a conservative
     test, because the MCP after PEI can turn a dead-source copy into a live-source one.
4. **`CapstoneFrameLowering.cpp`:**
   - `enableShrinkWrapping` returns false when `needsLiveSourceCopySlot(MF)`.
   - `processFunctionBeforeFrameFinalized`: when the slot is needed, set `ScavSlotsNum` to at least 1
     and record the first scavenging FI as the copy slot. The slot is 16 bytes, 16-aligned, and
     PEI places it next to fp/sp.
     - Hard error if `MFI.getSavePoints()` is non-empty: `-enable-shrink-wrap=true` bypasses the
       hook (`ShrinkWrap.cpp:1022`).
     - Hard error if the function is `naked`.
   - `estimateFunctionSizeInBytes` (:1785) counts each candidate as 8 bytes, which protects the
     branch-relaxation scratch decision.
5. **New `CapstoneLiveSourceCopy.cpp`**, `createCapstoneLiveSourceCopyPass(bool CheckOnly)`, wired
   through `Capstone.h`, `CMakeLists.txt` and `initialize…` in `CapstoneTargetMachine.cpp`:
   - Return immediately if the feature is on. No `skipFunction`: it must run at -O0 and on `optnone`.
   - For each block, walk backwards with `LiveRegUnits` and `addLiveOuts(MBB)`. For each MOVC with a
     non-exempt, live source (liveness just after the MI), replace it with
     `STC src, off(FrameReg)` + `LDC dst, off(FrameReg)`. Take `FrameReg` and `off` from
     `TFI->getFrameIndexReference`, and attach the FixedStack MMOs (16 bytes, align 16).
     - Do not use `eliminateFrameIndex`.
     - Require a zero scalable part and a simm12 offset; otherwise `reportFatalInternalError` naming
       the function.
     - Assert `FrameReg != SP || hasReservedCallFrame(MF)`.
     - A missing slot is fatal.
   - Without `TracksLiveness`, every candidate counts as live.
   - Optional hardening: a kept MOVC gets `implicit-def dead $src`.
   - STATISTICs: `converted`, `keptDeadSource`, `keptExemptSource`.
   - CheckOnly mode: any remaining live-source candidate is fatal.
6. **`addPreEmitPass`** (`CapstoneTargetMachine.cpp:600-617`, whose order is MCP, LateBranchOpt,
   IndirectBranchTracking, BranchRelaxation, MakeCompressible):
   - the pass goes after IndirectBranchTracking and **before BranchRelaxation**, because each
     conversion adds 4 bytes;
   - a CheckOnly instance goes after MakeCompressible.

   After that point nothing builds a MOVC or re-reads a MOVC source. The only builders are
   `CapstoneInstrInfo.cpp:550` and `CapstoneRegisterInfo.cpp:313`. MakeCompressible only scavenges
   GPR, FPR and GPRPair classes (`CapstoneMakeCompressible.cpp:343-357`); add an assert there.
7. **Comments:** rewrite the "ponytail" paragraph in `copyPhysReg` (:535-548) and the MOVC/C-46 note
   in the `.td` (:2484-2522). Add one line to `CapstoneISASemantics.md`: under QEMU,
   `STC`+`LDC` duplicates a LINEAR value, which is only observable if code already breaks the
   no-copy contract.
8. **Docs:**
   - ISSUES C-32: the class is fixed under the rule; keep the evidence.
   - Amend Q-04's claim that `iconv_open` is "out of reach of any compiler fix".
   - Register every shared-file edit in `llvm/utils/capstone-shared-patches.txt`.
9. **Before any board use: rtl-sim** (the `rtl-sim` skill). Cases:
   - an adjacent `stc`/`ldc` on one granule, with an integer, a NONLIN and a LINEAR source;
   - back-to-back pairs through the same granule;
   - an `ldc` that uses the copied register as its base (the S-07 shape).

   This is the R-29/S-07 store-to-load family, so the check is required.

## Phase B — `__intcap`/`__uintcap_t` in clang (W2, about 2-3 weeks)

A purecap-only port from CHERI LLVM 17 (`/home/biecho/cheri/llvm-project`, `7e122876ee01`). The file
and line map is from the inventory. The port is about 1,100-1,400 lines without tests; hybrid mode
and offset mode are dropped. **`IntPtrType` stays `SignedLong`**: skip CHERI's `RISCV.h:61-63`.

1. **Type plumbing:**
   - `BuiltinTypes.def` (`UIntCap` before `UInt128`, `IntCap` before `Int128`), the `__intcap`
     keyword, `TST_intcap`, `DeclSpec`, the parser sites and `SemaType.cpp:1517`;
   - `ASTContext`: size and align from new `TargetInfo::getIntCapWidth/Align()` (128),
     `getIntWidth` equal to the address width (64), and rank `UINT_MAX`;
   - `Type.cpp` predicates (`isIntCapType`, `isCHERICapabilityType`, `canCarryProvenance`),
     serialization IDs, Itanium mangling, debug info and printf `P`.
2. **Target:**
   - `clang/lib/Basic/Targets/Capstone.h` (`Capstone64TargetInfo`, :229-294): a
     `SupportsCapabilities()` hook, `getIntCapWidth/Align() = 128` and address range 64;
   - macros `__SIZEOF_INTCAP__`, `__INTCAP_MAX__`, `__UINTCAP_MAX__`, `__PTRADDR_TYPE__` and
     `__has_feature(capabilities)`;
   - `stddef.h` `ptraddr_t` and `stdint.h` `intcap_t`/`uintcap_t`;
   - `__null` stays `long`.
3. **Sema:**
   - the `cheri_no_provenance` attribute;
   - `DiagnoseAmbiguousProvenance` for add, mul, bitwise and the overflow builtins;
   - cast rules: pointer ↔ intcap keeps the capability, integer → intcap is null-derived (marked
     no-provenance), intcap → integer is the address;
   - diagnostic groups under CHERI's names (`cheri-provenance`, `cheri-capability-misuse`,
     `cheri-bitwise-operations`, `shorten-cap-to-int`), so M0's findings and CheriBSD practice
     carry over;
   - the provenance check at Sema's cast-creation points instead of threading `ASTContext` through
     every `CastExpr` constructor. That saves about 150-200 lines of shared-file churn.
4. **CodeGen:**
   - intcap is `ptr addrspace(200)` (`CodeGenTypes`);
   - `GetBinOpVal`/`GetBinOpResult` (CHERI `CGExprScalar.cpp:801-889`), add/sub, inc/dec, unary
     operators, compound assignment, conversions (`EmitScalarConversion`), `CK_IntegralToPointer`
     and `CK_PointerToIntegral`, `switch` on the address, constants as null-derived GEPs, and
     `__builtin_align_*` on intcap;
   - two target hooks:
     - `getPointerAddress` becomes the existing `llvm.capstone.cap.get.cursor` (already used at
       `clang/lib/CodeGen/CGExpr.cpp:5332`);
     - `setPointerAddress` becomes a **new** intrinsic `llvm.capstone.cap.set.address` (Phase C),
       not raw `cap.scc`, because the result must be defined for an untagged value.
5. **Shared-file manifest:** every clang file touched goes in `capstone-shared-patches.txt`. That
   will be dozens of entries; regenerate with `capstone-shared-drift.py --write` and review the
   diff by hand.

## Phase C — back end for intcap (W2, about 1.5-2 weeks)

1. **`llvm.capstone.cap.set.address(ptr addrspace(200), i64)`** (`IntrinsicsCapstone.td`,
   `IntrNoMem`, no `IntrSpeculatable`). It selects to `PseudoSetAddr`. A pre-RA pass modelled on
   `CapstoneLdcRetry.cpp:136-215` expands it in `addPreRegAlloc`, into a branch diamond:
   - `LCC sel 1` into a type;
   - if the type is 1 (NONLIN): `SCC`;
   - otherwise (untagged, sealed and so on): the integer bridge, giving an untagged value with the
     new address. CHERI likewise clears the tag on sealed.

   This never executes a trapping `SCC`. After expansion, `SCC` is in `trapsOnUntaggedOperand`, so
   C-66 keeps it guarded. Expanding after early MachineLICM/MachineCSE means those passes see only
   the pseudo.
2. **Verify the existing lowering for intcap IR**, with lit tests for each:
   - `gep i8, ptr addrspace(200) null, %x`, the C-40 path at `CapstoneISelDAGToDAG.cpp:1485-1525`;
   - `icmp` signed and unsigned on `addrspace(200)`: it must compare cursors;
   - `LDC`/`STC` of intcap values;
   - calls and returns;
   - `PseudoTRUNC_CAP` for intcap → integer.

   A missing piece becomes a small DAG fix in `CapstoneISelLowering.cpp`.
3. No 16-byte atomics: PostgreSQL's `pg_atomic_*` never take a `Datum`. A diagnostic points at
   C-54 if one appears.

## Phase D — PostgreSQL on intcap (W3, about 1-3 weeks; M0 refines this)

1. **New patches on top of 0001-0009** (`capstone/ports/postgres/single-user/patches/`):
   - `0010`: `postgres.h:64` becomes `typedef __uintcap_t Datum`. `SIZEOF_DATUM` stays
     `SIZEOF_VOID_P` (16), and `USE_FLOAT8_BYVAL` stays on.
   - `0011`: the 13 `SIZEOF_DATUM == 8` branches become `>= 8`. That includes `gistproc`, `mac`,
     `uuid` and `network`; `numeric` is already 0005.
   - `0012+`: M0's site list, provenance fixes (`cheri_no_provenance` or an explicit address) at
     the sites where `Datum` carries hash, bit or sort-abbreviation arithmetic.
   - Guard: the PostgreSQL build must be warning-clean on `-Wcheri-provenance`.
2. **Build** with `build-domain.sh`, pointing `toolchain/capstone-cc` at W2's compiler, with
   `PGSU_FROM=make PGSU_CLEAN=1`.
3. **Boot:**
   - `run-domain.sh` with `--boot` first; it must get past `create pg_proc`;
   - then all five initdb calls from `record-initdb.sh`; a native `postgres` must accept the data
     directory;
   - then `--single` with `work.sql`, whose output must be byte-identical to the native run in
     `/tmp/capstone/pg-single-user/native`.
4. **Board route:** repeat step 3 under `CAPSTONE_MOVC_NULL_SCALAR=1` with Phase A on. That is the
   silicon-equivalent correctness check without a board.

## Verification (every gate can fail; controls are named)

- **Phase A:**
  - **Lit:**
    - new `live-source-copy.ll`, with a control RUN line at `-mattr=+movc-keeps-integer-source`
      that keeps `movc`. Cases:
      - an argument saved across a call becomes `stc`/`ldc`;
      - dead-source and `c0`/`sp` copies stay `movc`;
      - a leaf function gets a 16-byte frame;
      - no shrink-wrap;
      - `-frame-pointer=all` uses fp; dynamic alloca uses fp; `align 64` uses bp;
      - `not llc -enable-shrink-wrap=true` errors;
    - an MIR test with `-start-before=shrink-wrap -stop-after=capstone-live-source-copy`, covering
      live through a successor's live-in, a call's implicit use, `PseudoRET`, and an x9-half read;
      dead through a regmask or a redefinition; and a post-RA-scheduler case;
    - update `c32-movc-untagged-live.ll`, keeping the old checks under the control, and regenerate
      the other 17 `movc` tests;
    - full Capstone lit directory, X86 and Generic: no new failures.
  - **QEMU:** `capstone/tests/runtime-qemu/movc-null-scalar/`.
    - `MOVC-C32` becomes a judged check: with the rule and `CAPSTONE_MOVC_NULL_SCALAR=1` it must read
      `got=0x5000`; a `+movc-keeps-integer-source` build must read `0x1`, which is the positive
      control;
    - add an iconv-shaped case, one integer passed to three calls;
    - `exposure.sh`: both arms identical with the rule on, and `iconv_open` passes. A rebuild with
      `+movc-keeps-integer-source` must fault at `iconv_open` at the 2026-09-24 pc; that is the
      positive control.
  - **Cost:** the CPython-core stats (`converted` against the measured 30,625), code growth,
    functions that lost shrink-wrapping or gained a frame, and CoreMark, SQLite and CPython case
    timing with the rule on and off.
  - **Nightly core tier:** identical to dev, except for the known four.
  - **rtl-sim:** the cases in A.9, before any board run.
- **Phases B and C:**
  - retargeted CHERI tests (about 25 selected files: `intcap.c`, `intcapswitch.c`,
    `cap-provenance*.c`, `convert-*-cap*.c`, `intcap-rank.c`, `uintcap-add.c`, `cheri-intcap-range.c`,
    and so on) on `capstone64`;
  - `set.address` lit, covering the NONLIN, untagged and sealed arms plus the C-66 no-hoist check;
  - a QEMU probe modelled on `runtime-qemu/untagged-cap-arith/` (`run.sh`/`entry.c`/`-DCASE=`), each
    case with a `uintptr_t` control that must fault where intcap does not:
    - a pointer through intcap survives a copy, a spill and a call;
    - an integer through intcap survives the same;
    - arithmetic, comparison and `switch` are right;
    - `CAPSTONE_CINC_UNTAGGED_SURVIVE=1` logs zero untagged `cincoffset`s;
  - the same probe under `CAPSTONE_MOVC_NULL_SCALAR=1`.
- **Phase D:** the one-variable pair is the same tree with `0010` reverted, which must still fault at
  `GetTableAmRoutine+0x38`. Gates:
  - `create pg_proc` passes;
  - all five initdb calls complete, and native `pg_controldata` and `postgres` accept the directory;
  - the `work.sql` output is byte-identical to native;
  - all of that is repeated under `CAPSTONE_MOVC_NULL_SCALAR=1`;
  - the time and memory of a 16-byte `Datum` are measured against native.
- **Always:** precommit-scan by absolute path before every commit and push; `-o` commits; result
  lines, not captures; an auditor before any root-cause or FIXED claim in ISSUES.

## Estimate

| Step | Weeks |
|---|---|
| Step 0 | about 0.5 days |
| M0 | 0.4 |
| Phase A | 1.5-2 |
| Phase B | 2-3 |
| Phase C | 1.5-2 |
| Phase D | 1-3 |
| **Total to PostgreSQL on QEMU, under the silicon `MOVC` rule** | **about 7-10** |

These are judgement anchored on the measured inventory. M0 turns Phase D into a count.
