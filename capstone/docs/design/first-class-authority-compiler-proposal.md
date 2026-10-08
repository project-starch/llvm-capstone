# First-class authority in the compiler: a research proposal (2026-10-08)

**Status: proposal for discussion. Nothing is implemented.** It records the idea, the evidence that
motivates it, a prior-art check against primary sources, and the smallest experiment that would
test it. It is not a plan of record.

## 1. The problem, as it exists in this repository today

- **The LLVM backend has no notion of a capability's kind.**
  - Every capability is a pointer in `addrspace(200)` (`CapstoneISelLowering.cpp:1853`).
  - Linear, revocation and sealed capabilities are reached only through intrinsics:
    `int_capstone_cap_mrev`, `_revoke`, `_delin`, `_init`, `_seal` and the rest
    (`llvm/include/llvm/IR/IntrinsicsCapstone.td`).
  - Misuse is visible only at run time: after a move the register holds `cnull`, and the next
    access faults.
- **LLVM's value model assumes every value is copyable, and on this ISA that is unsound for linear
  values.**
  - MOVC, STC and LDC clear their linear source
    (`capstone-academic-spec/parts/cap-man-insn.adoc:37-38`, `mem-access-insn.adoc:54-55, 105`).
  - Two failures are already recorded in `docs/ref/ISSUES.md`:
    - **C-32:** MachineCSE extended a linear value's live range, and the register allocator's own
      copy of it was a clearing MOVC. This was observed on silicon on 2026-09-15 in the SQLite
      Sublet port's `setupLookaside`.
    - **Q-12:** `create_domain` spills the linear `dom_gp` and loads the slot twice. On silicon
      the first load moves the value out, so the second delivers `cnull`. It is latent only
      because no glue reads that slot.
  - Each instance has been fixed pass by pass.
- **A first-class `__linear` qualifier already exists, half-finished.** `capstone/capstone-c` is a
  separate C-subset compiler. It carries the qualifier, and buildroot domain sources use it 31
  times, but its use-after-move checks are commented out (`src/lang_defs.rs:461-467`).

So linear capabilities are usable safely only in hand-written runtime code, and the compiler is
the component most likely to break them.

## 2. The idea

An MLIR dialect in which hardware authority values are first-class SSA values with substructural
types:

```mlir
!cap.linear<perms, len>     !cap.nonlinear<...>     !cap.rev     !cap.sealed     !cap.borrowed<perms>

%r = capstone.lend %pkt : !cap.linear<rw, 1500> {      // the borrow scope is an IR region
  ^bb(%b: !cap.borrowed<r>):
    %x = capstone.domain.call @plugin::@main(%b, %n)  // the authority crossing is explicit
}

capstone.domain @plugin attributes {entry = [@main]} { ... }   // IsolatedFromAbove
```

Three things MLIR gives that LLVM IR does not:

1. **Lifetimes as regions.** The end of a borrow is a structural fact of the IR, not something
   reconstructed from the CFG.
2. **Domains as isolated nested symbol tables.** MLIR's `IsolatedFromAbove` trait forbids an op
   from implicitly capturing outer values. That is domain isolation stated in the IR's own
   structure: no authority enters except as an explicit argument.
3. **A verifier per op.** Use-once of a linear value is a dialect invariant, not a lint.

### C surface

- **Parameter attributes:** `__borrowed`, `__linear`. Here the region is the call.
- **An explicit scope:** `__lend(pkt, CAP_READ) as (v) { ... }`. This lowers to `capstone.lend`.
  It is in the spirit of GNU `__attribute__((cleanup))`.
- **Later, inferred regions,** in the spirit of non-lexical lifetimes. Annotations are then needed
  only where authority crosses a boundary.

## 3. The decision the compiler makes: who enforces the guarantee

```text
for each authority value v:
    facts = analyze(v)          // escapes? crosses a domain/FFI? needs copies? derived nodes?
    if not facts.crosses and not facts.escapes:      mode = STATIC        // proved by the compiler
    elif facts.crosses and verified_binary(callee):  mode = STATIC_CERT   // proved on the callee's binary
    else:                                            mode = HARDWARE      // left to the hardware
    cost_hw = c_mrev + c_revoke_per_node * facts.derived + c_move * moves(v)  // silicon-calibrated

lower capstone.lend %c {body} with view %b:
    STATIC, STATIC_CERT:  %b = shrink_perms(%c); lower(body); emit_certificate(%b, facts)
    HARDWARE:             %rev = mrev(%c); %b = derive(%c); lower(body); revoke(%rev)
```

**Cost-model inputs that are already measured on silicon** (`docs/ref/fpga-silicon-measurements-for-paper.md`):
- the R1 release cost: about 22.9 cycles per affected revocation node warm, 28.15 cold;
- the dependent-load ratio;
- the per-global `ldc` cost (the ladder's `cnt` and `bs` rows).

**Consequence for the LLVM erasure problem.** What reaches LLVM IR is a *decision*, not a type.
- Statically enforced values become plain non-linear capabilities, and LLVM may treat them as
  ordinary pointers.
- Only the HARDWARE residue needs care:
  - its own address space;
  - consuming operations modelled as intrinsics with memory effects and `noduplicate`;
  - a MachineVerifier rule against copying a linear vreg whose source is still live.

  That residue is the C-32 and Q-12 class, and fixing it is worth doing regardless of this
  proposal.

### Code generation

The main path is: the dialect, then `llvm`, then LLVM IR, then the existing backend.

**Direct emission from MLIR** would keep types through register allocation. It is plausible only
for small trusted code such as the monitor, allocator kernels and the Sublet runtime, where
correctness matters more than code quality. A possible existence proof is xDSL's `riscv` dialect (unverified).

### Trust

| mode | effect of a compiler bug | what must be carried |
|---|---|---|
| HARDWARE | precision loss only; authority cannot be forged | nothing |
| STATIC (own trusted code) | a hole | a certificate. Trusting the toolchain for one's own code is the Sublet paper's stated model (`02-problem.tex:205`) |
| STATIC_CERT (untrusted callee) | a hole, and the attacker controls the callee's compiler | a certificate checked on the **binary**, by a small validator |

Full type-preserving compilation is therefore not required. Only the evidence behind the static
decisions has to survive to the binary.

## 4. Prior art

These were checked against primary sources on 2026-10-08:
- the CapsLock and Capstone quotations, read in full text;
- the FRESCO and SafeFFI abstracts, read on arXiv.

Rows marked (summary) were not re-read.

| work | overlap | what it does not do |
|---|---|---|
| **Capstone** (USENIX Security 2023, https://arxiv.org/abs/2302.13863) | the hardware this targets | It enforces Rust-style ownership "through the correct use of capabilities during runtime, rather than through static type checking, offering a dynamic alternative". The static/hardware hybrid is left open |
| **CapsLock: Securing Mixed Rust with Hardware Capabilities** (CCS 2025, https://arxiv.org/abs/2507.03344) | the same hardware lineage and goal; a rustc MIR pass injects borrow instructions | It replaces linear and revocation capabilities with a new revoke-on-use mechanism, QEMU only. **Its argument against this route:** "looking at use_p() alone is not sufficient to statically know if p corresponds to a revocation capability that needs revocation. Such information may only be determined during run-time." |
| **FRESCO** (arXiv 2608.26353) | Color Saver, a static capability-aware escape analysis that decides where hardware stack temporal enforcement is applied on CHERI | The static gate is function-granular ("confines coloring to functions needing it") and applies to stack coloring. Heap and stack temporal safety coexist through color segmentation. It has no substructural authority types. It runs on QEMU and the CHERI-Toooba FPGA softcore; our board is an FPGA too, so "silicon" is not a differentiator against it |
| **Linear capabilities for fully abstract compilation of separation-logic-verified code** (ICFP 2019 / JFP 2021) | a static proof directs lowering to hardware linear capabilities | Its compilation is linear everywhere and never elides hardware linearity. It is formal only |
| **QSSA** (CC 2022), **Mojo** (MLIR-based; https://mojolang.org/docs/manual/values/lifetimes) (summary) | linear or affine SSA values checked in MLIR | No hardware authority, no fallback |
| **MLIR ownership-based buffer deallocation** (https://mlir.llvm.org/docs/OwnershipBasedBufferDeallocation/) | ownership as SSA dataflow | Inserts frees; no security semantics |
| **CCured** (POPL 2002), **Gradual Ownership Types** (ESOP 2012), **SafeFFI** (arXiv 2510.20688) (summary except SafeFFI's abstract) | a per-pointer static/dynamic split, or checks only at safe/unsafe boundaries | Software checks; no capability hardware |
| **CHERIoT compartments**, **Wasm component `own`/`borrow`** (summary) | first-class compartments; borrows that end at the call | Runtime or ABI enforcement; no per-value choice |

**Answer to CapsLock's argument.** It holds for unannotated code: inside an opaque callee nothing
is known. The proposal moves exactly that information onto the boundary (`domain.call`'s authority
signature and `lend` scopes), and falls back to hardware where it is absent.

Two things it must also do:
- **compare against revoke-on-use, or rule it out explicitly.** Revoke-on-use needs hardware that
  does not exist on the board;
- **make clear that CapsLock checks a stronger property** (aliasing-XOR-mutability across FFI) than
  non-escape of a lent capability. The claim must be scoped to the weaker one.

**Novelty claim, as narrow as the prior art allows.** No work found combines all three of:
- first-class typed authority (linear, non-linear, revocation, sealed, domains) as IR values;
- a **per-value** choice between static proof and hardware linear capabilities plus `mrev`;
- a cost model calibrated on the deployed hardware.

Linear SSA in MLIR is **not** new, and neither is a static gate on hardware enforcement (FRESCO).
The contribution has to be the authority semantics and the decision procedure.

## 5. Signals that it may be a dead end

- **CapsLock's designers dropped linear and revocation capabilities** as a compiler target. That is
  the strongest negative signal.
- **If revoke and linear moves are cheap,** the hybrid buys little. R1's ~23 cycles per node
  suggests a single lend is tens of cycles. The payoff may lie in revocation-node pressure (a
  finite pool) and in moving errors to compile time, more than in cycles.
- **Authority that flows into untyped code** (musl, SQLite) may dominate real programs and collapse
  everything to HARDWARE.

## 6. Smallest experiment that tests it

1. **First, independently: the backend invariant.** A linear address space plus the MachineVerifier
   rule. This closes the C-32/Q-12 class for C today and is a prerequisite for any frontend.
2. **A dialect with five types, `lend`, `domain.call` and consuming ops,** with input written by
   hand or from attribute-annotated C.
3. **One hybrid decision,** revoke elision for provably non-escaping lends, lowered to the existing
   backend.
4. **One real case that needs both modes:**
   - the Sublet runtime, or FFmpeg's pools under leases (406 leases per 1 s decode,
     `ports/ffmpeg/app/results/2026-09-24-qemu-pool-safety/`);
   - measured on the board: revokes and moves removed, cycles, and revocation nodes saved.
5. **Static detection** of the recorded latent bugs (Q-12's double load) as an expressiveness
   check.

## 7. Open questions

- **Which source language comes first?** Attribute-annotated C via ClangIR is the path to MLIR.
  Rust needs a `capstone64` target in rustc and the `usize`/pointer-width split (Rust strict
  provenance helps). Its natural mapping is to keep `&`/`&mut` as non-linear capabilities and to
  use hardware linearity only at trust boundaries, because a reborrow per `&mut` would otherwise
  cost a derive and a revoke.
- **Where does the certificate live, and how small can its checker be?**
- **Is per-value granularity measurably better than FRESCO-style per-function gating** on the same
  workloads?
