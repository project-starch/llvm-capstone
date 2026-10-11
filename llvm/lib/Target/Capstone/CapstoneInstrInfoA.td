//===-- CapstoneInstrInfoA.td - Capstone 'A' instructions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the Capstone instructions from the standard 'A', Atomic
// Instructions extension.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class LR_r<bit aq, bit rl, bits<3> funct3, string opcodestr,
           RegisterOperand AddrRC = GPRMemZeroOffset>
    : RVInstRAtomic<0b00010, aq, rl, funct3, OPC_AMO,
                    (outs GPR:$rd), (ins AddrRC:$rs1),
                    opcodestr, "$rd, $rs1"> {
  let rs2 = 0;
}

multiclass LR_r_aq_rl<bits<3> funct3, string opcodestr> {
  def ""     : LR_r<0, 0, funct3, opcodestr>;
  def _AQ    : LR_r<1, 0, funct3, opcodestr # ".aq">;
  def _RL    : LR_r<0, 1, funct3, opcodestr # ".rl">;
  def _AQ_RL : LR_r<1, 1, funct3, opcodestr # ".aqrl">;
  // The encoding is shared; AS200 bases must remain capability registers.
  let isCodeGenOnly = 1 in {
    def _CAP       : LR_r<0, 0, funct3, opcodestr, GPCRMemZeroOffset>;
    def _AQ_CAP    : LR_r<1, 0, funct3, opcodestr # ".aq", GPCRMemZeroOffset>;
    def _RL_CAP    : LR_r<0, 1, funct3, opcodestr # ".rl", GPCRMemZeroOffset>;
    def _AQ_RL_CAP : LR_r<1, 1, funct3, opcodestr # ".aqrl", GPCRMemZeroOffset>;
  }
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
class SC_r<bit aq, bit rl, bits<3> funct3, string opcodestr,
           RegisterOperand AddrRC = GPRMemZeroOffset>
    : RVInstRAtomic<0b00011, aq, rl, funct3, OPC_AMO,
                    (outs GPR:$rd), (ins AddrRC:$rs1, GPR:$rs2),
                    opcodestr, "$rd, $rs2, $rs1">;

multiclass SC_r_aq_rl<bits<3> funct3, string opcodestr> {
  def ""     : SC_r<0, 0, funct3, opcodestr>;
  def _AQ    : SC_r<1, 0, funct3, opcodestr # ".aq">;
  def _RL    : SC_r<0, 1, funct3, opcodestr # ".rl">;
  def _AQ_RL : SC_r<1, 1, funct3, opcodestr # ".aqrl">;
  // The encoding is shared; AS200 bases must remain capability registers.
  let isCodeGenOnly = 1 in {
    def _CAP       : SC_r<0, 0, funct3, opcodestr, GPCRMemZeroOffset>;
    def _AQ_CAP    : SC_r<1, 0, funct3, opcodestr # ".aq", GPCRMemZeroOffset>;
    def _RL_CAP    : SC_r<0, 1, funct3, opcodestr # ".rl", GPCRMemZeroOffset>;
    def _AQ_RL_CAP : SC_r<1, 1, funct3, opcodestr # ".aqrl", GPCRMemZeroOffset>;
  }
}

let hasSideEffects = 0, mayLoad = 1, mayStore = 1 in
class AMO_rr<bits<5> funct5, bit aq, bit rl, bits<3> funct3, string opcodestr,
             RegisterOperand AddrRC = GPRMemZeroOffset>
    : RVInstRAtomic<funct5, aq, rl, funct3, OPC_AMO,
                    (outs GPR:$rd), (ins AddrRC:$rs1, GPR:$rs2),
                    opcodestr, "$rd, $rs2, $rs1">;

multiclass AMO_rr_aq_rl<bits<5> funct5, bits<3> funct3, string opcodestr> {
  def ""     : AMO_rr<funct5, 0, 0, funct3, opcodestr>;
  def _AQ    : AMO_rr<funct5, 1, 0, funct3, opcodestr # ".aq">;
  def _RL    : AMO_rr<funct5, 0, 1, funct3, opcodestr # ".rl">;
  def _AQ_RL : AMO_rr<funct5, 1, 1, funct3, opcodestr # ".aqrl">;
  // The encoding is shared; AS200 bases must remain capability registers.
  let isCodeGenOnly = 1 in {
    def _CAP       : AMO_rr<funct5, 0, 0, funct3, opcodestr, GPCRMemZeroOffset>;
    def _AQ_CAP    : AMO_rr<funct5, 1, 0, funct3, opcodestr # ".aq", GPCRMemZeroOffset>;
    def _RL_CAP    : AMO_rr<funct5, 0, 1, funct3, opcodestr # ".rl", GPCRMemZeroOffset>;
    def _AQ_RL_CAP : AMO_rr<funct5, 1, 1, funct3, opcodestr # ".aqrl", GPCRMemZeroOffset>;
  }
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZalrsc], IsSignExtendingOpW = 1 in {
defm LR_W       : LR_r_aq_rl<0b010, "lr.w">, Sched<[WriteAtomicLDW, ReadAtomicLDW]>;
defm SC_W       : SC_r_aq_rl<0b010, "sc.w">,
                  Sched<[WriteAtomicSTW, ReadAtomicSTW, ReadAtomicSTW]>;
} // Predicates = [HasStdExtZalrsc], IsSignExtendingOpW = 1

let Predicates = [HasStdExtZaamo], IsSignExtendingOpW = 1 in {
defm AMOSWAP_W  : AMO_rr_aq_rl<0b00001, 0b010, "amoswap.w">,
                  Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>;
defm AMOADD_W   : AMO_rr_aq_rl<0b00000, 0b010, "amoadd.w">,
                  Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>;
defm AMOXOR_W   : AMO_rr_aq_rl<0b00100, 0b010, "amoxor.w">,
                  Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>;
defm AMOAND_W   : AMO_rr_aq_rl<0b01100, 0b010, "amoand.w">,
                  Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>;
defm AMOOR_W    : AMO_rr_aq_rl<0b01000, 0b010, "amoor.w">,
                  Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>;
defm AMOMIN_W   : AMO_rr_aq_rl<0b10000, 0b010, "amomin.w">,
                  Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>;
defm AMOMAX_W   : AMO_rr_aq_rl<0b10100, 0b010, "amomax.w">,
                  Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>;
defm AMOMINU_W  : AMO_rr_aq_rl<0b11000, 0b010, "amominu.w">,
                  Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>;
defm AMOMAXU_W  : AMO_rr_aq_rl<0b11100, 0b010, "amomaxu.w">,
                  Sched<[WriteAtomicW, ReadAtomicWA, ReadAtomicWD]>;
} // Predicates = [HasStdExtZaamo], IsSignExtendingOpW = 1

let Predicates = [HasStdExtZalrsc, IsRV64] in {
defm LR_D       : LR_r_aq_rl<0b011, "lr.d">, Sched<[WriteAtomicLDD, ReadAtomicLDD]>;
defm SC_D       : SC_r_aq_rl<0b011, "sc.d">,
                  Sched<[WriteAtomicSTD, ReadAtomicSTD, ReadAtomicSTD]>;
} // Predicates = [HasStdExtZalrsc, IsRV64]

let Predicates = [HasStdExtZaamo, IsRV64] in {
defm AMOSWAP_D  : AMO_rr_aq_rl<0b00001, 0b011, "amoswap.d">,
                  Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>;
defm AMOADD_D   : AMO_rr_aq_rl<0b00000, 0b011, "amoadd.d">,
                  Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>;
defm AMOXOR_D   : AMO_rr_aq_rl<0b00100, 0b011, "amoxor.d">,
                  Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>;
defm AMOAND_D   : AMO_rr_aq_rl<0b01100, 0b011, "amoand.d">,
                  Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>;
defm AMOOR_D    : AMO_rr_aq_rl<0b01000, 0b011, "amoor.d">,
                  Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>;
defm AMOMIN_D   : AMO_rr_aq_rl<0b10000, 0b011, "amomin.d">,
                  Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>;
defm AMOMAX_D   : AMO_rr_aq_rl<0b10100, 0b011, "amomax.d">,
                  Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>;
defm AMOMINU_D  : AMO_rr_aq_rl<0b11000, 0b011, "amominu.d">,
                  Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>;
defm AMOMAXU_D  : AMO_rr_aq_rl<0b11100, 0b011, "amomaxu.d">,
                  Sched<[WriteAtomicD, ReadAtomicDA, ReadAtomicDD]>;
} // Predicates = [HasStdExtZaamo, IsRV64]

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//

let IsAtomic = 1 in {
// An atomic load operation that does not need either acquire or release
// semantics.
class relaxed_load<PatFrags base>
    : PatFrag<(ops node:$ptr), (base node:$ptr)> {
  let IsAtomicOrderingAcquireOrStronger = 0;
}

// A atomic load operation that actually needs acquire semantics.
class acquiring_load<PatFrags base>
    : PatFrag<(ops node:$ptr), (base node:$ptr)> {
  let IsAtomicOrderingAcquire = 1;
}

// An atomic load operation that needs sequential consistency.
class seq_cst_load<PatFrags base>
    : PatFrag<(ops node:$ptr), (base node:$ptr)> {
  let IsAtomicOrderingSequentiallyConsistent = 1;
}

// An atomic store operation that does not need either acquire or release
// semantics.
class relaxed_store<PatFrag base>
    : PatFrag<(ops node:$val, node:$ptr), (base node:$val, node:$ptr)> {
  let IsAtomicOrderingReleaseOrStronger = 0;
}

// A store operation that actually needs release semantics.
class releasing_store<PatFrag base>
    : PatFrag<(ops node:$val, node:$ptr), (base node:$val, node:$ptr)> {
  let IsAtomicOrderingRelease = 1;
}

// A store operation that actually needs sequential consistency.
class seq_cst_store<PatFrag base>
    : PatFrag<(ops node:$val, node:$ptr), (base node:$val, node:$ptr)> {
  let IsAtomicOrderingSequentiallyConsistent = 1;
}
} // IsAtomic = 1

// Atomic load/store are available under both +a and +force-atomics.
// Fences will be inserted for atomic load/stores according to the logic in
// CapstoneTargetLowering::{emitLeadingFence,emitTrailingFence}.
// The normal loads/stores are relaxed (unordered) loads/stores that don't have
// any ordering. This is necessary because AtomicExpandPass has added fences to
// atomic load/stores and changed them to unordered ones.
let Predicates = [HasAtomicLdSt] in {
  def : LdPat<relaxed_load<atomic_load_asext_8>,  LB>;
  def : LdPat<relaxed_load<atomic_load_asext_16>, LH>;
  def : LdPat<relaxed_load<atomic_load_zext_8>,  LBU>;
  def : LdPat<relaxed_load<atomic_load_zext_16>, LHU>;

  def : StPat<relaxed_store<atomic_store_8>,  SB, GPR, XLenVT>;
  def : StPat<relaxed_store<atomic_store_16>, SH, GPR, XLenVT>;
  def : StPat<relaxed_store<atomic_store_32>, SW, GPR, XLenVT>;
}

let Predicates = [HasAtomicLdSt, IsRV32] in {
  def : LdPat<relaxed_load<atomic_load_nonext_32>, LW>;
}

let Predicates = [HasAtomicLdSt, IsRV64] in {
  def : LdPat<relaxed_load<atomic_load_asext_32>, LW>;
  def : LdPat<relaxed_load<atomic_load_zext_32>, LWU>;
  def : LdPat<relaxed_load<atomic_load_nonext_64>, LD, i64>;
  def : StPat<relaxed_store<atomic_store_64>, SD, GPR, i64>;
}

/// AMOs

class AtomicPat<PatFrag Op, RVInst Inst, ValueType VT,
                RegisterClass AddrRC, ValueType AddrVT>
    : Pat<(VT (Op (AddrVT AddrRC:$rs1), (VT GPR:$rs2))),
          (Inst AddrRC:$rs1, GPR:$rs2)>;

multiclass AMOPat<string AtomicOp, string BaseInst, ValueType vt = XLenVT,
                  list<Predicate> ExtraPreds = []> {
  // Match address and value types independently: CLenVT is not XLenVT.
  foreach IsCap = [0, 1] in {
    defvar AddrRC = !if(IsCap, GPCR, GPR);
    defvar AddrVT = !if(IsCap, CLenVT, vt);
    defvar Suffix = !if(IsCap, "_CAP", "");
    let Predicates = !listconcat([HasStdExtA, NoStdExtZtso], ExtraPreds) in {
      def : AtomicPat<!cast<PatFrag>(AtomicOp#"_monotonic"),
                      !cast<RVInst>(BaseInst#Suffix), vt, AddrRC, AddrVT>;
      def : AtomicPat<!cast<PatFrag>(AtomicOp#"_acquire"),
                      !cast<RVInst>(BaseInst#"_AQ"#Suffix), vt, AddrRC, AddrVT>;
      def : AtomicPat<!cast<PatFrag>(AtomicOp#"_release"),
                      !cast<RVInst>(BaseInst#"_RL"#Suffix), vt, AddrRC, AddrVT>;
      def : AtomicPat<!cast<PatFrag>(AtomicOp#"_acq_rel"),
                      !cast<RVInst>(BaseInst#"_AQ_RL"#Suffix), vt, AddrRC, AddrVT>;
      def : AtomicPat<!cast<PatFrag>(AtomicOp#"_seq_cst"),
                      !cast<RVInst>(BaseInst#"_AQ_RL"#Suffix), vt, AddrRC, AddrVT>;
    }
    let Predicates = !listconcat([HasStdExtA, HasStdExtZtso], ExtraPreds) in {
      def : AtomicPat<!cast<PatFrag>(AtomicOp#"_monotonic"),
                      !cast<RVInst>(BaseInst#Suffix), vt, AddrRC, AddrVT>;
      def : AtomicPat<!cast<PatFrag>(AtomicOp#"_acquire"),
                      !cast<RVInst>(BaseInst#Suffix), vt, AddrRC, AddrVT>;
      def : AtomicPat<!cast<PatFrag>(AtomicOp#"_release"),
                      !cast<RVInst>(BaseInst#Suffix), vt, AddrRC, AddrVT>;
      def : AtomicPat<!cast<PatFrag>(AtomicOp#"_acq_rel"),
                      !cast<RVInst>(BaseInst#Suffix), vt, AddrRC, AddrVT>;
      def : AtomicPat<!cast<PatFrag>(AtomicOp#"_seq_cst"),
                      !cast<RVInst>(BaseInst#Suffix), vt, AddrRC, AddrVT>;
    }
  }
}

defm : AMOPat<"atomic_swap_i32", "AMOSWAP_W">;
defm : AMOPat<"atomic_load_add_i32", "AMOADD_W">;
defm : AMOPat<"atomic_load_and_i32", "AMOAND_W">;
defm : AMOPat<"atomic_load_or_i32", "AMOOR_W">;
defm : AMOPat<"atomic_load_xor_i32", "AMOXOR_W">;
defm : AMOPat<"atomic_load_max_i32", "AMOMAX_W">;
defm : AMOPat<"atomic_load_min_i32", "AMOMIN_W">;
defm : AMOPat<"atomic_load_umax_i32", "AMOMAXU_W">;
defm : AMOPat<"atomic_load_umin_i32", "AMOMINU_W">;

defm : AMOPat<"atomic_swap_i64", "AMOSWAP_D", i64, [IsRV64]>;
defm : AMOPat<"atomic_load_add_i64", "AMOADD_D", i64, [IsRV64]>;
defm : AMOPat<"atomic_load_and_i64", "AMOAND_D", i64, [IsRV64]>;
defm : AMOPat<"atomic_load_or_i64", "AMOOR_D", i64, [IsRV64]>;
defm : AMOPat<"atomic_load_xor_i64", "AMOXOR_D", i64, [IsRV64]>;
defm : AMOPat<"atomic_load_max_i64", "AMOMAX_D", i64, [IsRV64]>;
defm : AMOPat<"atomic_load_min_i64", "AMOMIN_D", i64, [IsRV64]>;
defm : AMOPat<"atomic_load_umax_i64", "AMOMAXU_D", i64, [IsRV64]>;
defm : AMOPat<"atomic_load_umin_i64", "AMOMINU_D", i64, [IsRV64]>;


/// Pseudo AMOs

class PseudoAMO<RegisterClass AddrRC = GPR>
    : Pseudo<(outs GPR:$res, GPR:$scratch),
             (ins AddrRC:$addr, GPR:$incr, ixlenimm:$ordering), []> {
  let Constraints = "@earlyclobber $res,@earlyclobber $scratch";
  let mayLoad = 1;
  let mayStore = 1;
  let hasSideEffects = 0;
}

// AddrRC is GPCR for the capability-address forms (the _CAP pseudos below):
// the aligned word a masked (8/16-bit) atomic works on is reached through the
// capability, so LR/SC take the _CAP encodings at expansion.
class PseudoMaskedAMO<RegisterClass AddrRC = GPR>
    : Pseudo<(outs GPR:$res, GPR:$scratch),
             (ins AddrRC:$addr, GPR:$incr, GPR:$mask, ixlenimm:$ordering), []> {
  let Constraints = "@earlyclobber $res,@earlyclobber $scratch";
  let mayLoad = 1;
  let mayStore = 1;
  let hasSideEffects = 0;
}

class PseudoMaskedAMOMinMax<RegisterClass AddrRC = GPR>
    : Pseudo<(outs GPR:$res, GPR:$scratch1, GPR:$scratch2),
             (ins AddrRC:$addr, GPR:$incr, GPR:$mask, ixlenimm:$sextshamt,
              ixlenimm:$ordering), []> {
  let Constraints = "@earlyclobber $res,@earlyclobber $scratch1,"
                    "@earlyclobber $scratch2";
  let mayLoad = 1;
  let mayStore = 1;
  let hasSideEffects = 0;
}

class PseudoMaskedAMOUMinUMax<RegisterClass AddrRC = GPR>
    : Pseudo<(outs GPR:$res, GPR:$scratch1, GPR:$scratch2),
             (ins AddrRC:$addr, GPR:$incr, GPR:$mask, ixlenimm:$ordering), []> {
  let Constraints = "@earlyclobber $res,@earlyclobber $scratch1,"
                    "@earlyclobber $scratch2";
  let mayLoad = 1;
  let mayStore = 1;
  let hasSideEffects = 0;
}

// Ordering constants must be kept in sync with the AtomicOrdering enum in
// AtomicOrdering.h.
multiclass PseudoAMOPat<string AtomicOp, Pseudo AMOInst, ValueType vt = XLenVT,
                       RegisterClass AddrRC = GPR, ValueType AddrVT = vt> {
  def : Pat<(vt (!cast<PatFrag>(AtomicOp#"_monotonic")
                (AddrVT AddrRC:$addr), GPR:$incr)),
            (AMOInst AddrRC:$addr, GPR:$incr, 2)>;
  def : Pat<(vt (!cast<PatFrag>(AtomicOp#"_acquire")
                (AddrVT AddrRC:$addr), GPR:$incr)),
            (AMOInst AddrRC:$addr, GPR:$incr, 4)>;
  def : Pat<(vt (!cast<PatFrag>(AtomicOp#"_release")
                (AddrVT AddrRC:$addr), GPR:$incr)),
            (AMOInst AddrRC:$addr, GPR:$incr, 5)>;
  def : Pat<(vt (!cast<PatFrag>(AtomicOp#"_acq_rel")
                (AddrVT AddrRC:$addr), GPR:$incr)),
            (AMOInst AddrRC:$addr, GPR:$incr, 6)>;
  def : Pat<(vt (!cast<PatFrag>(AtomicOp#"_seq_cst")
                (AddrVT AddrRC:$addr), GPR:$incr)),
            (AMOInst AddrRC:$addr, GPR:$incr, 7)>;
}

// Address and data types are matched independently, as for AMOPat: the
// masked intrinsics are overloaded on the pointer, and an addrspace(200)
// address is CLenVT in GPCR, not XLenVT.
class PseudoMaskedAMOPat<Intrinsic intrin, Pseudo AMOInst,
                         RegisterClass AddrRC = GPR, ValueType AddrVT = XLenVT>
    : Pat<(XLenVT (intrin (AddrVT AddrRC:$addr), (XLenVT GPR:$incr),
                          (XLenVT GPR:$mask), (XLenVT timm:$ordering))),
          (AMOInst AddrRC:$addr, GPR:$incr, GPR:$mask, timm:$ordering)>;

class PseudoMaskedAMOMinMaxPat<Intrinsic intrin, Pseudo AMOInst,
                               RegisterClass AddrRC = GPR,
                               ValueType AddrVT = XLenVT>
    : Pat<(XLenVT (intrin (AddrVT AddrRC:$addr), (XLenVT GPR:$incr),
                          (XLenVT GPR:$mask), (XLenVT GPR:$shiftamt),
           (XLenVT timm:$ordering))),
          (AMOInst AddrRC:$addr, GPR:$incr, GPR:$mask, GPR:$shiftamt,
           timm:$ordering)>;

let Predicates = [HasStdExtA] in {

let Size = 20 in
def PseudoAtomicLoadNand32 : PseudoAMO;
defm : PseudoAMOPat<"atomic_load_nand_i32", PseudoAtomicLoadNand32>;
let Size = 20 in
def PseudoAtomicLoadNand32_CAP : PseudoAMO<GPCR>;
defm : PseudoAMOPat<"atomic_load_nand_i32", PseudoAtomicLoadNand32_CAP,
                    XLenVT, GPCR, CLenVT>;

let Size = 28 in {
  def PseudoMaskedAtomicSwap32 : PseudoMaskedAMO;
  def PseudoMaskedAtomicLoadAdd32 : PseudoMaskedAMO;
  def PseudoMaskedAtomicLoadSub32 : PseudoMaskedAMO;
  def PseudoMaskedAtomicSwap32_CAP : PseudoMaskedAMO<GPCR>;
  def PseudoMaskedAtomicLoadAdd32_CAP : PseudoMaskedAMO<GPCR>;
  def PseudoMaskedAtomicLoadSub32_CAP : PseudoMaskedAMO<GPCR>;
}
let Size = 32 in {
  def PseudoMaskedAtomicLoadNand32 : PseudoMaskedAMO;
  def PseudoMaskedAtomicLoadNand32_CAP : PseudoMaskedAMO<GPCR>;
}
let Size = 44 in {
  def PseudoMaskedAtomicLoadMax32 : PseudoMaskedAMOMinMax;
  def PseudoMaskedAtomicLoadMin32 : PseudoMaskedAMOMinMax;
  def PseudoMaskedAtomicLoadMax32_CAP : PseudoMaskedAMOMinMax<GPCR>;
  def PseudoMaskedAtomicLoadMin32_CAP : PseudoMaskedAMOMinMax<GPCR>;
}
let Size = 36 in {
 def PseudoMaskedAtomicLoadUMax32 : PseudoMaskedAMOUMinUMax;
 def PseudoMaskedAtomicLoadUMin32 : PseudoMaskedAMOUMinUMax;
 def PseudoMaskedAtomicLoadUMax32_CAP : PseudoMaskedAMOUMinUMax<GPCR>;
 def PseudoMaskedAtomicLoadUMin32_CAP : PseudoMaskedAMOUMinUMax<GPCR>;
}

def : PseudoMaskedAMOPat<int_capstone_masked_atomicrmw_xchg,
                         PseudoMaskedAtomicSwap32>;
def : PseudoMaskedAMOPat<int_capstone_masked_atomicrmw_add,
                         PseudoMaskedAtomicLoadAdd32>;
def : PseudoMaskedAMOPat<int_capstone_masked_atomicrmw_sub,
                         PseudoMaskedAtomicLoadSub32>;
def : PseudoMaskedAMOPat<int_capstone_masked_atomicrmw_nand,
                         PseudoMaskedAtomicLoadNand32>;
def : PseudoMaskedAMOMinMaxPat<int_capstone_masked_atomicrmw_max,
                               PseudoMaskedAtomicLoadMax32>;
def : PseudoMaskedAMOMinMaxPat<int_capstone_masked_atomicrmw_min,
                               PseudoMaskedAtomicLoadMin32>;
def : PseudoMaskedAMOPat<int_capstone_masked_atomicrmw_umax,
                         PseudoMaskedAtomicLoadUMax32>;
def : PseudoMaskedAMOPat<int_capstone_masked_atomicrmw_umin,
                         PseudoMaskedAtomicLoadUMin32>;

def : PseudoMaskedAMOPat<int_capstone_masked_atomicrmw_xchg,
                         PseudoMaskedAtomicSwap32_CAP, GPCR, CLenVT>;
def : PseudoMaskedAMOPat<int_capstone_masked_atomicrmw_add,
                         PseudoMaskedAtomicLoadAdd32_CAP, GPCR, CLenVT>;
def : PseudoMaskedAMOPat<int_capstone_masked_atomicrmw_sub,
                         PseudoMaskedAtomicLoadSub32_CAP, GPCR, CLenVT>;
def : PseudoMaskedAMOPat<int_capstone_masked_atomicrmw_nand,
                         PseudoMaskedAtomicLoadNand32_CAP, GPCR, CLenVT>;
def : PseudoMaskedAMOMinMaxPat<int_capstone_masked_atomicrmw_max,
                               PseudoMaskedAtomicLoadMax32_CAP, GPCR, CLenVT>;
def : PseudoMaskedAMOMinMaxPat<int_capstone_masked_atomicrmw_min,
                               PseudoMaskedAtomicLoadMin32_CAP, GPCR, CLenVT>;
def : PseudoMaskedAMOPat<int_capstone_masked_atomicrmw_umax,
                         PseudoMaskedAtomicLoadUMax32_CAP, GPCR, CLenVT>;
def : PseudoMaskedAMOPat<int_capstone_masked_atomicrmw_umin,
                         PseudoMaskedAtomicLoadUMin32_CAP, GPCR, CLenVT>;
} // Predicates = [HasStdExtA]

let Predicates = [HasStdExtA, IsRV64] in {

let Size = 20 in
def PseudoAtomicLoadNand64 : PseudoAMO;
defm : PseudoAMOPat<"atomic_load_nand_i64", PseudoAtomicLoadNand64, i64>;
let Size = 20 in
def PseudoAtomicLoadNand64_CAP : PseudoAMO<GPCR>;
defm : PseudoAMOPat<"atomic_load_nand_i64", PseudoAtomicLoadNand64_CAP,
                    i64, GPCR, CLenVT>;
} // Predicates = [HasStdExtA, IsRV64]


/// Compare and exchange

class PseudoCmpXchg<RegisterClass AddrRC = GPR>
    : Pseudo<(outs GPR:$res, GPR:$scratch),
             (ins AddrRC:$addr, GPR:$cmpval, GPR:$newval, ixlenimm:$ordering), []> {
  let Constraints = "@earlyclobber $res,@earlyclobber $scratch";
  let mayLoad = 1;
  let mayStore = 1;
  let hasSideEffects = 0;
  let Size = 16;
}

// Ordering constants must be kept in sync with the AtomicOrdering enum in
// AtomicOrdering.h.
multiclass PseudoCmpXchgPat<string Op, Pseudo CmpXchgInst, ValueType vt = XLenVT,
                          RegisterClass AddrRC = GPR, ValueType AddrVT = vt> {
  def : Pat<(vt (!cast<PatFrag>(Op#"_monotonic")
                (AddrVT AddrRC:$addr), GPR:$cmp, GPR:$new)),
            (CmpXchgInst AddrRC:$addr, GPR:$cmp, GPR:$new, 2)>;
  def : Pat<(vt (!cast<PatFrag>(Op#"_acquire")
                (AddrVT AddrRC:$addr), GPR:$cmp, GPR:$new)),
            (CmpXchgInst AddrRC:$addr, GPR:$cmp, GPR:$new, 4)>;
  def : Pat<(vt (!cast<PatFrag>(Op#"_release")
                (AddrVT AddrRC:$addr), GPR:$cmp, GPR:$new)),
            (CmpXchgInst AddrRC:$addr, GPR:$cmp, GPR:$new, 5)>;
  def : Pat<(vt (!cast<PatFrag>(Op#"_acq_rel")
                (AddrVT AddrRC:$addr), GPR:$cmp, GPR:$new)),
            (CmpXchgInst AddrRC:$addr, GPR:$cmp, GPR:$new, 6)>;
  def : Pat<(vt (!cast<PatFrag>(Op#"_seq_cst")
                (AddrVT AddrRC:$addr), GPR:$cmp, GPR:$new)),
            (CmpXchgInst AddrRC:$addr, GPR:$cmp, GPR:$new, 7)>;
}

let Predicates = [HasStdExtA, NoStdExtZacas] in {
def PseudoCmpXchg32 : PseudoCmpXchg;
defm : PseudoCmpXchgPat<"atomic_cmp_swap_i32", PseudoCmpXchg32>;
def PseudoCmpXchg32_CAP : PseudoCmpXchg<GPCR>;
defm : PseudoCmpXchgPat<"atomic_cmp_swap_i32", PseudoCmpXchg32_CAP,
                       XLenVT, GPCR, CLenVT>;
}

let Predicates = [HasStdExtA, NoStdExtZacas, IsRV64] in {
def PseudoCmpXchg64 : PseudoCmpXchg;
defm : PseudoCmpXchgPat<"atomic_cmp_swap_i64", PseudoCmpXchg64, i64>;
def PseudoCmpXchg64_CAP : PseudoCmpXchg<GPCR>;
defm : PseudoCmpXchgPat<"atomic_cmp_swap_i64", PseudoCmpXchg64_CAP,
                       i64, GPCR, CLenVT>;
}

let Predicates = [HasStdExtA] in {
class PseudoMaskedCmpXchg<RegisterClass AddrRC>
    : Pseudo<(outs GPR:$res, GPR:$scratch),
             (ins AddrRC:$addr, GPR:$cmpval, GPR:$newval, GPR:$mask,
              ixlenimm:$ordering), []> {
  let Constraints = "@earlyclobber $res,@earlyclobber $scratch";
  let mayLoad = 1;
  let mayStore = 1;
  let hasSideEffects = 0;
  let Size = 32;
}
def PseudoMaskedCmpXchg32 : PseudoMaskedCmpXchg<GPR>;
def PseudoMaskedCmpXchg32_CAP : PseudoMaskedCmpXchg<GPCR>;

foreach IsCap = [0, 1] in {
  defvar AddrRC = !if(IsCap, GPCR, GPR);
  defvar AddrVT = !if(IsCap, CLenVT, XLenVT);
  defvar Inst = !cast<Pseudo>("PseudoMaskedCmpXchg32" # !if(IsCap, "_CAP", ""));
  def : Pat<(XLenVT (int_capstone_masked_cmpxchg
              (AddrVT AddrRC:$addr), (XLenVT GPR:$cmpval), (XLenVT GPR:$newval),
              (XLenVT GPR:$mask), (XLenVT timm:$ordering))),
            (Inst AddrRC:$addr, GPR:$cmpval, GPR:$newval, GPR:$mask,
                  timm:$ordering)>;
}
} // Predicates = [HasStdExtA]
