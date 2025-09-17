//===-- CapstoneISelDAGToDAG.cpp - A dag to dag inst selector for Capstone -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the Capstone target.
//
//===----------------------------------------------------------------------===//

#include "CapstoneISelDAGToDAG.h"
#include "MCTargetDesc/CapstoneBaseInfo.h"
#include "MCTargetDesc/CapstoneMCTargetDesc.h"
#include "MCTargetDesc/CapstoneMatInt.h"
#include "CapstoneISelLowering.h"
#include "CapstoneInstrInfo.h"
#include "CapstoneSelectionDAGInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/SDPatternMatch.h"
#include "llvm/IR/IntrinsicsCapstone.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "capstone-isel"
#define PASS_NAME "Capstone DAG->DAG Pattern Instruction Selection"

static cl::opt<bool> UsePseudoMovImm(
    "capstone-use-rematerializable-movimm", cl::Hidden,
    cl::desc("Use a rematerializable pseudoinstruction for 2 instruction "
             "constant materialization"),
    cl::init(false));

#define GET_DAGISEL_BODY CapstoneDAGToDAGISel
#include "CapstoneGenDAGISel.inc"

void CapstoneDAGToDAGISel::PreprocessISelDAG() {
  SelectionDAG::allnodes_iterator Position = CurDAG->allnodes_end();

  bool MadeChange = false;
  while (Position != CurDAG->allnodes_begin()) {
    SDNode *N = &*--Position;
    if (N->use_empty())
      continue;

    SDValue Result;
    switch (N->getOpcode()) {
    case ISD::SPLAT_VECTOR: {
      // Convert integer SPLAT_VECTOR to VMV_V_X_VL and floating-point
      // SPLAT_VECTOR to VFMV_V_F_VL to reduce isel burden.
      MVT VT = N->getSimpleValueType(0);
      unsigned Opc =
          VT.isInteger() ? CapstoneISD::VMV_V_X_VL : CapstoneISD::VFMV_V_F_VL;
      SDLoc DL(N);
      SDValue VL = CurDAG->getRegister(Capstone::X0, Subtarget->getXLenVT());
      SDValue Src = N->getOperand(0);
      if (VT.isInteger())
        Src = CurDAG->getNode(ISD::ANY_EXTEND, DL, Subtarget->getXLenVT(),
                              N->getOperand(0));
      Result = CurDAG->getNode(Opc, DL, VT, CurDAG->getUNDEF(VT), Src, VL);
      break;
    }
    case CapstoneISD::SPLAT_VECTOR_SPLIT_I64_VL: {
      // Lower SPLAT_VECTOR_SPLIT_I64 to two scalar stores and a stride 0 vector
      // load. Done after lowering and combining so that we have a chance to
      // optimize this to VMV_V_X_VL when the upper bits aren't needed.
      assert(N->getNumOperands() == 4 && "Unexpected number of operands");
      MVT VT = N->getSimpleValueType(0);
      SDValue Passthru = N->getOperand(0);
      SDValue Lo = N->getOperand(1);
      SDValue Hi = N->getOperand(2);
      SDValue VL = N->getOperand(3);
      assert(VT.getVectorElementType() == MVT::i64 && VT.isScalableVector() &&
             Lo.getValueType() == MVT::i32 && Hi.getValueType() == MVT::i32 &&
             "Unexpected VTs!");
      MachineFunction &MF = CurDAG->getMachineFunction();
      SDLoc DL(N);

      // Create temporary stack for each expanding node.
      SDValue StackSlot =
          CurDAG->CreateStackTemporary(TypeSize::getFixed(8), Align(8));
      int FI = cast<FrameIndexSDNode>(StackSlot.getNode())->getIndex();
      MachinePointerInfo MPI = MachinePointerInfo::getFixedStack(MF, FI);

      SDValue Chain = CurDAG->getEntryNode();
      Lo = CurDAG->getStore(Chain, DL, Lo, StackSlot, MPI, Align(8));

      SDValue OffsetSlot =
          CurDAG->getMemBasePlusOffset(StackSlot, TypeSize::getFixed(4), DL);
      Hi = CurDAG->getStore(Chain, DL, Hi, OffsetSlot, MPI.getWithOffset(4),
                            Align(8));

      Chain = CurDAG->getNode(ISD::TokenFactor, DL, MVT::Other, Lo, Hi);

      SDVTList VTs = CurDAG->getVTList({VT, MVT::Other});
      SDValue IntID =
          CurDAG->getTargetConstant(Intrinsic::capstone_vlse, DL, MVT::i64);
      SDValue Ops[] = {Chain,
                       IntID,
                       Passthru,
                       StackSlot,
                       CurDAG->getRegister(Capstone::X0, MVT::i64),
                       VL};

      Result = CurDAG->getMemIntrinsicNode(ISD::INTRINSIC_W_CHAIN, DL, VTs, Ops,
                                           MVT::i64, MPI, Align(8),
                                           MachineMemOperand::MOLoad);
      break;
    }
    case ISD::FP_EXTEND: {
      // We only have vector patterns for capstone_fpextend_vl in isel.
      SDLoc DL(N);
      MVT VT = N->getSimpleValueType(0);
      if (!VT.isVector())
        break;
      SDValue VLMAX = CurDAG->getRegister(Capstone::X0, Subtarget->getXLenVT());
      SDValue TrueMask = CurDAG->getNode(
          CapstoneISD::VMSET_VL, DL, VT.changeVectorElementType(MVT::i1), VLMAX);
      Result = CurDAG->getNode(CapstoneISD::FP_EXTEND_VL, DL, VT, N->getOperand(0),
                               TrueMask, VLMAX);
      break;
    }
    }

    if (Result) {
      LLVM_DEBUG(dbgs() << "Capstone DAG preprocessing replacing:\nOld:    ");
      LLVM_DEBUG(N->dump(CurDAG));
      LLVM_DEBUG(dbgs() << "\nNew: ");
      LLVM_DEBUG(Result->dump(CurDAG));
      LLVM_DEBUG(dbgs() << "\n");

      CurDAG->ReplaceAllUsesOfValueWith(SDValue(N, 0), Result);
      MadeChange = true;
    }
  }

  if (MadeChange)
    CurDAG->RemoveDeadNodes();
}

void CapstoneDAGToDAGISel::PostprocessISelDAG() {
  HandleSDNode Dummy(CurDAG->getRoot());
  SelectionDAG::allnodes_iterator Position = CurDAG->allnodes_end();

  bool MadeChange = false;
  while (Position != CurDAG->allnodes_begin()) {
    SDNode *N = &*--Position;
    // Skip dead nodes and any non-machine opcodes.
    if (N->use_empty() || !N->isMachineOpcode())
      continue;

    MadeChange |= doPeepholeSExtW(N);

    // FIXME: This is here only because the VMerge transform doesn't
    // know how to handle masked true inputs.  Once that has been moved
    // to post-ISEL, this can be deleted as well.
    MadeChange |= doPeepholeMaskedRVV(cast<MachineSDNode>(N));
  }

  CurDAG->setRoot(Dummy.getValue());

  // After we're done with everything else, convert IMPLICIT_DEF
  // passthru operands to NoRegister.  This is required to workaround
  // an optimization deficiency in MachineCSE.  This really should
  // be merged back into each of the patterns (i.e. there's no good
  // reason not to go directly to NoReg), but is being done this way
  // to allow easy backporting.
  MadeChange |= doPeepholeNoRegPassThru();

  if (MadeChange)
    CurDAG->RemoveDeadNodes();
}

static SDValue selectImmSeq(SelectionDAG *CurDAG, const SDLoc &DL, const MVT VT,
                            CapstoneMatInt::InstSeq &Seq) {
  SDValue SrcReg = CurDAG->getRegister(Capstone::X0, VT);
  for (const CapstoneMatInt::Inst &Inst : Seq) {
    SDValue SDImm = CurDAG->getSignedTargetConstant(Inst.getImm(), DL, VT);
    SDNode *Result = nullptr;
    switch (Inst.getOpndKind()) {
    case CapstoneMatInt::Imm:
      Result = CurDAG->getMachineNode(Inst.getOpcode(), DL, VT, SDImm);
      break;
    case CapstoneMatInt::RegX0:
      Result = CurDAG->getMachineNode(Inst.getOpcode(), DL, VT, SrcReg,
                                      CurDAG->getRegister(Capstone::X0, VT));
      break;
    case CapstoneMatInt::RegReg:
      Result = CurDAG->getMachineNode(Inst.getOpcode(), DL, VT, SrcReg, SrcReg);
      break;
    case CapstoneMatInt::RegImm:
      Result = CurDAG->getMachineNode(Inst.getOpcode(), DL, VT, SrcReg, SDImm);
      break;
    }

    // Only the first instruction has X0 as its source.
    SrcReg = SDValue(Result, 0);
  }

  return SrcReg;
}

static SDValue selectImm(SelectionDAG *CurDAG, const SDLoc &DL, const MVT VT,
                         int64_t Imm, const CapstoneSubtarget &Subtarget) {
  CapstoneMatInt::InstSeq Seq = CapstoneMatInt::generateInstSeq(Imm, Subtarget);

  // Use a rematerializable pseudo instruction for short sequences if enabled.
  if (Seq.size() == 2 && UsePseudoMovImm)
    return SDValue(
        CurDAG->getMachineNode(Capstone::PseudoMovImm, DL, VT,
                               CurDAG->getSignedTargetConstant(Imm, DL, VT)),
        0);

  // See if we can create this constant as (ADD (SLLI X, C), X) where X is at
  // worst an LUI+ADDIW. This will require an extra register, but avoids a
  // constant pool.
  // If we have Zba we can use (ADD_UW X, (SLLI X, 32)) to handle cases where
  // low and high 32 bits are the same and bit 31 and 63 are set.
  if (Seq.size() > 3) {
    unsigned ShiftAmt, AddOpc;
    CapstoneMatInt::InstSeq SeqLo =
        CapstoneMatInt::generateTwoRegInstSeq(Imm, Subtarget, ShiftAmt, AddOpc);
    if (!SeqLo.empty() && (SeqLo.size() + 2) < Seq.size()) {
      SDValue Lo = selectImmSeq(CurDAG, DL, VT, SeqLo);

      SDValue SLLI = SDValue(
          CurDAG->getMachineNode(Capstone::SLLI, DL, VT, Lo,
                                 CurDAG->getTargetConstant(ShiftAmt, DL, VT)),
          0);
      return SDValue(CurDAG->getMachineNode(AddOpc, DL, VT, Lo, SLLI), 0);
    }
  }

  // Otherwise, use the original sequence.
  return selectImmSeq(CurDAG, DL, VT, Seq);
}

void CapstoneDAGToDAGISel::addVectorLoadStoreOperands(
    SDNode *Node, unsigned Log2SEW, const SDLoc &DL, unsigned CurOp,
    bool IsMasked, bool IsStridedOrIndexed, SmallVectorImpl<SDValue> &Operands,
    bool IsLoad, MVT *IndexVT) {
  SDValue Chain = Node->getOperand(0);

  Operands.push_back(Node->getOperand(CurOp++)); // Base pointer.

  if (IsStridedOrIndexed) {
    Operands.push_back(Node->getOperand(CurOp++)); // Index.
    if (IndexVT)
      *IndexVT = Operands.back()->getSimpleValueType(0);
  }

  if (IsMasked) {
    SDValue Mask = Node->getOperand(CurOp++);
    Operands.push_back(Mask);
  }
  SDValue VL;
  selectVLOp(Node->getOperand(CurOp++), VL);
  Operands.push_back(VL);

  MVT XLenVT = Subtarget->getXLenVT();
  SDValue SEWOp = CurDAG->getTargetConstant(Log2SEW, DL, XLenVT);
  Operands.push_back(SEWOp);

  // At the IR layer, all the masked load intrinsics have policy operands,
  // none of the others do.  All have passthru operands.  For our pseudos,
  // all loads have policy operands.
  if (IsLoad) {
    uint64_t Policy = CapstoneVType::MASK_AGNOSTIC;
    if (IsMasked)
      Policy = Node->getConstantOperandVal(CurOp++);
    SDValue PolicyOp = CurDAG->getTargetConstant(Policy, DL, XLenVT);
    Operands.push_back(PolicyOp);
  }

  Operands.push_back(Chain); // Chain.
}

void CapstoneDAGToDAGISel::selectVLSEG(SDNode *Node, unsigned NF, bool IsMasked,
                                    bool IsStrided) {
  SDLoc DL(Node);
  MVT VT = Node->getSimpleValueType(0);
  unsigned Log2SEW = Node->getConstantOperandVal(Node->getNumOperands() - 1);
  CapstoneVType::VLMUL LMUL = CapstoneTargetLowering::getLMUL(VT);

  unsigned CurOp = 2;
  SmallVector<SDValue, 8> Operands;

  Operands.push_back(Node->getOperand(CurOp++));

  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked, IsStrided,
                             Operands, /*IsLoad=*/true);

  const Capstone::VLSEGPseudo *P =
      Capstone::getVLSEGPseudo(NF, IsMasked, IsStrided, /*FF*/ false, Log2SEW,
                            static_cast<unsigned>(LMUL));
  MachineSDNode *Load =
      CurDAG->getMachineNode(P->Pseudo, DL, MVT::Untyped, MVT::Other, Operands);

  CurDAG->setNodeMemRefs(Load, {cast<MemSDNode>(Node)->getMemOperand()});

  ReplaceUses(SDValue(Node, 0), SDValue(Load, 0));
  ReplaceUses(SDValue(Node, 1), SDValue(Load, 1));
  CurDAG->RemoveDeadNode(Node);
}

void CapstoneDAGToDAGISel::selectVLSEGFF(SDNode *Node, unsigned NF,
                                      bool IsMasked) {
  SDLoc DL(Node);
  MVT VT = Node->getSimpleValueType(0);
  MVT XLenVT = Subtarget->getXLenVT();
  unsigned Log2SEW = Node->getConstantOperandVal(Node->getNumOperands() - 1);
  CapstoneVType::VLMUL LMUL = CapstoneTargetLowering::getLMUL(VT);

  unsigned CurOp = 2;
  SmallVector<SDValue, 7> Operands;

  Operands.push_back(Node->getOperand(CurOp++));

  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                             /*IsStridedOrIndexed*/ false, Operands,
                             /*IsLoad=*/true);

  const Capstone::VLSEGPseudo *P =
      Capstone::getVLSEGPseudo(NF, IsMasked, /*Strided*/ false, /*FF*/ true,
                            Log2SEW, static_cast<unsigned>(LMUL));
  MachineSDNode *Load = CurDAG->getMachineNode(P->Pseudo, DL, MVT::Untyped,
                                               XLenVT, MVT::Other, Operands);

  CurDAG->setNodeMemRefs(Load, {cast<MemSDNode>(Node)->getMemOperand()});

  ReplaceUses(SDValue(Node, 0), SDValue(Load, 0)); // Result
  ReplaceUses(SDValue(Node, 1), SDValue(Load, 1)); // VL
  ReplaceUses(SDValue(Node, 2), SDValue(Load, 2)); // Chain
  CurDAG->RemoveDeadNode(Node);
}

void CapstoneDAGToDAGISel::selectVLXSEG(SDNode *Node, unsigned NF, bool IsMasked,
                                     bool IsOrdered) {
  SDLoc DL(Node);
  MVT VT = Node->getSimpleValueType(0);
  unsigned Log2SEW = Node->getConstantOperandVal(Node->getNumOperands() - 1);
  CapstoneVType::VLMUL LMUL = CapstoneTargetLowering::getLMUL(VT);

  unsigned CurOp = 2;
  SmallVector<SDValue, 8> Operands;

  Operands.push_back(Node->getOperand(CurOp++));

  MVT IndexVT;
  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                             /*IsStridedOrIndexed*/ true, Operands,
                             /*IsLoad=*/true, &IndexVT);

#ifndef NDEBUG
  // Number of element = RVVBitsPerBlock * LMUL / SEW
  unsigned ContainedTyNumElts = Capstone::RVVBitsPerBlock >> Log2SEW;
  auto DecodedLMUL = CapstoneVType::decodeVLMUL(LMUL);
  if (DecodedLMUL.second)
    ContainedTyNumElts /= DecodedLMUL.first;
  else
    ContainedTyNumElts *= DecodedLMUL.first;
  assert(ContainedTyNumElts == IndexVT.getVectorMinNumElements() &&
         "Element count mismatch");
#endif

  CapstoneVType::VLMUL IndexLMUL = CapstoneTargetLowering::getLMUL(IndexVT);
  unsigned IndexLog2EEW = Log2_32(IndexVT.getScalarSizeInBits());
  if (IndexLog2EEW == 6 && !Subtarget->is64Bit()) {
    report_fatal_error("The V extension does not support EEW=64 for index "
                       "values when XLEN=32");
  }
  const Capstone::VLXSEGPseudo *P = Capstone::getVLXSEGPseudo(
      NF, IsMasked, IsOrdered, IndexLog2EEW, static_cast<unsigned>(LMUL),
      static_cast<unsigned>(IndexLMUL));
  MachineSDNode *Load =
      CurDAG->getMachineNode(P->Pseudo, DL, MVT::Untyped, MVT::Other, Operands);

  CurDAG->setNodeMemRefs(Load, {cast<MemSDNode>(Node)->getMemOperand()});

  ReplaceUses(SDValue(Node, 0), SDValue(Load, 0));
  ReplaceUses(SDValue(Node, 1), SDValue(Load, 1));
  CurDAG->RemoveDeadNode(Node);
}

void CapstoneDAGToDAGISel::selectVSSEG(SDNode *Node, unsigned NF, bool IsMasked,
                                    bool IsStrided) {
  SDLoc DL(Node);
  MVT VT = Node->getOperand(2)->getSimpleValueType(0);
  unsigned Log2SEW = Node->getConstantOperandVal(Node->getNumOperands() - 1);
  CapstoneVType::VLMUL LMUL = CapstoneTargetLowering::getLMUL(VT);

  unsigned CurOp = 2;
  SmallVector<SDValue, 8> Operands;

  Operands.push_back(Node->getOperand(CurOp++));

  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked, IsStrided,
                             Operands);

  const Capstone::VSSEGPseudo *P = Capstone::getVSSEGPseudo(
      NF, IsMasked, IsStrided, Log2SEW, static_cast<unsigned>(LMUL));
  MachineSDNode *Store =
      CurDAG->getMachineNode(P->Pseudo, DL, Node->getValueType(0), Operands);

  CurDAG->setNodeMemRefs(Store, {cast<MemSDNode>(Node)->getMemOperand()});

  ReplaceNode(Node, Store);
}

void CapstoneDAGToDAGISel::selectVSXSEG(SDNode *Node, unsigned NF, bool IsMasked,
                                     bool IsOrdered) {
  SDLoc DL(Node);
  MVT VT = Node->getOperand(2)->getSimpleValueType(0);
  unsigned Log2SEW = Node->getConstantOperandVal(Node->getNumOperands() - 1);
  CapstoneVType::VLMUL LMUL = CapstoneTargetLowering::getLMUL(VT);

  unsigned CurOp = 2;
  SmallVector<SDValue, 8> Operands;

  Operands.push_back(Node->getOperand(CurOp++));

  MVT IndexVT;
  addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                             /*IsStridedOrIndexed*/ true, Operands,
                             /*IsLoad=*/false, &IndexVT);

#ifndef NDEBUG
  // Number of element = RVVBitsPerBlock * LMUL / SEW
  unsigned ContainedTyNumElts = Capstone::RVVBitsPerBlock >> Log2SEW;
  auto DecodedLMUL = CapstoneVType::decodeVLMUL(LMUL);
  if (DecodedLMUL.second)
    ContainedTyNumElts /= DecodedLMUL.first;
  else
    ContainedTyNumElts *= DecodedLMUL.first;
  assert(ContainedTyNumElts == IndexVT.getVectorMinNumElements() &&
         "Element count mismatch");
#endif

  CapstoneVType::VLMUL IndexLMUL = CapstoneTargetLowering::getLMUL(IndexVT);
  unsigned IndexLog2EEW = Log2_32(IndexVT.getScalarSizeInBits());
  if (IndexLog2EEW == 6 && !Subtarget->is64Bit()) {
    report_fatal_error("The V extension does not support EEW=64 for index "
                       "values when XLEN=32");
  }
  const Capstone::VSXSEGPseudo *P = Capstone::getVSXSEGPseudo(
      NF, IsMasked, IsOrdered, IndexLog2EEW, static_cast<unsigned>(LMUL),
      static_cast<unsigned>(IndexLMUL));
  MachineSDNode *Store =
      CurDAG->getMachineNode(P->Pseudo, DL, Node->getValueType(0), Operands);

  CurDAG->setNodeMemRefs(Store, {cast<MemSDNode>(Node)->getMemOperand()});

  ReplaceNode(Node, Store);
}

void CapstoneDAGToDAGISel::selectVSETVLI(SDNode *Node) {
  if (!Subtarget->hasVInstructions())
    return;

  assert(Node->getOpcode() == ISD::INTRINSIC_WO_CHAIN && "Unexpected opcode");

  SDLoc DL(Node);
  MVT XLenVT = Subtarget->getXLenVT();

  unsigned IntNo = Node->getConstantOperandVal(0);

  assert((IntNo == Intrinsic::capstone_vsetvli ||
          IntNo == Intrinsic::capstone_vsetvlimax) &&
         "Unexpected vsetvli intrinsic");

  bool VLMax = IntNo == Intrinsic::capstone_vsetvlimax;
  unsigned Offset = (VLMax ? 1 : 2);

  assert(Node->getNumOperands() == Offset + 2 &&
         "Unexpected number of operands");

  unsigned SEW =
      CapstoneVType::decodeVSEW(Node->getConstantOperandVal(Offset) & 0x7);
  CapstoneVType::VLMUL VLMul = static_cast<CapstoneVType::VLMUL>(
      Node->getConstantOperandVal(Offset + 1) & 0x7);

  unsigned VTypeI = CapstoneVType::encodeVTYPE(VLMul, SEW, /*TailAgnostic*/ true,
                                            /*MaskAgnostic*/ true);
  SDValue VTypeIOp = CurDAG->getTargetConstant(VTypeI, DL, XLenVT);

  SDValue VLOperand;
  unsigned Opcode = Capstone::PseudoVSETVLI;
  if (auto *C = dyn_cast<ConstantSDNode>(Node->getOperand(1))) {
    if (auto VLEN = Subtarget->getRealVLen())
      if (*VLEN / CapstoneVType::getSEWLMULRatio(SEW, VLMul) == C->getZExtValue())
        VLMax = true;
  }
  if (VLMax || isAllOnesConstant(Node->getOperand(1))) {
    VLOperand = CurDAG->getRegister(Capstone::X0, XLenVT);
    Opcode = Capstone::PseudoVSETVLIX0;
  } else {
    VLOperand = Node->getOperand(1);

    if (auto *C = dyn_cast<ConstantSDNode>(VLOperand)) {
      uint64_t AVL = C->getZExtValue();
      if (isUInt<5>(AVL)) {
        SDValue VLImm = CurDAG->getTargetConstant(AVL, DL, XLenVT);
        ReplaceNode(Node, CurDAG->getMachineNode(Capstone::PseudoVSETIVLI, DL,
                                                 XLenVT, VLImm, VTypeIOp));
        return;
      }
    }
  }

  ReplaceNode(Node,
              CurDAG->getMachineNode(Opcode, DL, XLenVT, VLOperand, VTypeIOp));
}

bool CapstoneDAGToDAGISel::tryShrinkShlLogicImm(SDNode *Node) {
  MVT VT = Node->getSimpleValueType(0);
  unsigned Opcode = Node->getOpcode();
  assert((Opcode == ISD::AND || Opcode == ISD::OR || Opcode == ISD::XOR) &&
         "Unexpected opcode");
  SDLoc DL(Node);

  // For operations of the form (x << C1) op C2, check if we can use
  // ANDI/ORI/XORI by transforming it into (x op (C2>>C1)) << C1.
  SDValue N0 = Node->getOperand(0);
  SDValue N1 = Node->getOperand(1);

  ConstantSDNode *Cst = dyn_cast<ConstantSDNode>(N1);
  if (!Cst)
    return false;

  int64_t Val = Cst->getSExtValue();

  // Check if immediate can already use ANDI/ORI/XORI.
  if (isInt<12>(Val))
    return false;

  SDValue Shift = N0;

  // If Val is simm32 and we have a sext_inreg from i32, then the binop
  // produces at least 33 sign bits. We can peek through the sext_inreg and use
  // a SLLIW at the end.
  bool SignExt = false;
  if (isInt<32>(Val) && N0.getOpcode() == ISD::SIGN_EXTEND_INREG &&
      N0.hasOneUse() && cast<VTSDNode>(N0.getOperand(1))->getVT() == MVT::i32) {
    SignExt = true;
    Shift = N0.getOperand(0);
  }

  if (Shift.getOpcode() != ISD::SHL || !Shift.hasOneUse())
    return false;

  ConstantSDNode *ShlCst = dyn_cast<ConstantSDNode>(Shift.getOperand(1));
  if (!ShlCst)
    return false;

  uint64_t ShAmt = ShlCst->getZExtValue();

  // Make sure that we don't change the operation by removing bits.
  // This only matters for OR and XOR, AND is unaffected.
  uint64_t RemovedBitsMask = maskTrailingOnes<uint64_t>(ShAmt);
  if (Opcode != ISD::AND && (Val & RemovedBitsMask) != 0)
    return false;

  int64_t ShiftedVal = Val >> ShAmt;
  if (!isInt<12>(ShiftedVal))
    return false;

  // If we peeked through a sext_inreg, make sure the shift is valid for SLLIW.
  if (SignExt && ShAmt >= 32)
    return false;

  // Ok, we can reorder to get a smaller immediate.
  unsigned BinOpc;
  switch (Opcode) {
  default: llvm_unreachable("Unexpected opcode");
  case ISD::AND: BinOpc = Capstone::ANDI; break;
  case ISD::OR:  BinOpc = Capstone::ORI;  break;
  case ISD::XOR: BinOpc = Capstone::XORI; break;
  }

  unsigned ShOpc = SignExt ? Capstone::SLLIW : Capstone::SLLI;

  SDNode *BinOp = CurDAG->getMachineNode(
      BinOpc, DL, VT, Shift.getOperand(0),
      CurDAG->getSignedTargetConstant(ShiftedVal, DL, VT));
  SDNode *SLLI =
      CurDAG->getMachineNode(ShOpc, DL, VT, SDValue(BinOp, 0),
                             CurDAG->getTargetConstant(ShAmt, DL, VT));
  ReplaceNode(Node, SLLI);
  return true;
}

bool CapstoneDAGToDAGISel::trySignedBitfieldExtract(SDNode *Node) {
  unsigned Opc;

  if (Subtarget->hasVendorXTHeadBb())
    Opc = Capstone::TH_EXT;
  else if (Subtarget->hasVendorXAndesPerf())
    Opc = Capstone::NDS_BFOS;
  else if (Subtarget->hasVendorXqcibm())
    Opc = Capstone::QC_EXT;
  else
    // Only supported with XTHeadBb/XAndesPerf/Xqcibm at the moment.
    return false;

  auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
  if (!N1C)
    return false;

  SDValue N0 = Node->getOperand(0);
  if (!N0.hasOneUse())
    return false;

  auto BitfieldExtract = [&](SDValue N0, unsigned Msb, unsigned Lsb,
                             const SDLoc &DL, MVT VT) {
    if (Opc == Capstone::QC_EXT) {
      // QC.EXT X, width, shamt
      // shamt is the same as Lsb
      // width is the number of bits to extract from the Lsb
      Msb = Msb - Lsb + 1;
    }
    return CurDAG->getMachineNode(Opc, DL, VT, N0.getOperand(0),
                                  CurDAG->getTargetConstant(Msb, DL, VT),
                                  CurDAG->getTargetConstant(Lsb, DL, VT));
  };

  SDLoc DL(Node);
  MVT VT = Node->getSimpleValueType(0);
  const unsigned RightShAmt = N1C->getZExtValue();

  // Transform (sra (shl X, C1) C2) with C1 < C2
  //        -> (SignedBitfieldExtract X, msb, lsb)
  if (N0.getOpcode() == ISD::SHL) {
    auto *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (!N01C)
      return false;

    const unsigned LeftShAmt = N01C->getZExtValue();
    // Make sure that this is a bitfield extraction (i.e., the shift-right
    // amount can not be less than the left-shift).
    if (LeftShAmt > RightShAmt)
      return false;

    const unsigned MsbPlusOne = VT.getSizeInBits() - LeftShAmt;
    const unsigned Msb = MsbPlusOne - 1;
    const unsigned Lsb = RightShAmt - LeftShAmt;

    SDNode *Sbe = BitfieldExtract(N0, Msb, Lsb, DL, VT);
    ReplaceNode(Node, Sbe);
    return true;
  }

  // Transform (sra (sext_inreg X, _), C) ->
  //           (SignedBitfieldExtract X, msb, lsb)
  if (N0.getOpcode() == ISD::SIGN_EXTEND_INREG) {
    unsigned ExtSize =
        cast<VTSDNode>(N0.getOperand(1))->getVT().getSizeInBits();

    // ExtSize of 32 should use sraiw via tablegen pattern.
    if (ExtSize == 32)
      return false;

    const unsigned Msb = ExtSize - 1;
    // If the shift-right amount is greater than Msb, it means that extracts
    // the X[Msb] bit and sign-extend it.
    const unsigned Lsb = RightShAmt > Msb ? Msb : RightShAmt;

    SDNode *Sbe = BitfieldExtract(N0, Msb, Lsb, DL, VT);
    ReplaceNode(Node, Sbe);
    return true;
  }

  return false;
}

bool CapstoneDAGToDAGISel::trySignedBitfieldInsertInMask(SDNode *Node) {
  // Supported only in Xqcibm for now.
  if (!Subtarget->hasVendorXqcibm())
    return false;

  using namespace SDPatternMatch;

  SDValue X;
  APInt MaskImm;
  if (!sd_match(Node, m_Or(m_OneUse(m_Value(X)), m_ConstInt(MaskImm))))
    return false;

  unsigned ShAmt, Width;
  if (!MaskImm.isShiftedMask(ShAmt, Width) || MaskImm.isSignedIntN(12))
    return false;

  // If Zbs is enabled and it is a single bit set we can use BSETI which
  // can be compressed to C_BSETI when Xqcibm in enabled.
  if (Width == 1 && Subtarget->hasStdExtZbs())
    return false;

  // If C1 is a shifted mask (but can't be formed as an ORI),
  // use a bitfield insert of -1.
  // Transform (or x, C1)
  //        -> (qc.insbi x, -1, width, shift)
  SDLoc DL(Node);
  MVT VT = Node->getSimpleValueType(0);

  SDValue Ops[] = {X, CurDAG->getSignedTargetConstant(-1, DL, VT),
                   CurDAG->getTargetConstant(Width, DL, VT),
                   CurDAG->getTargetConstant(ShAmt, DL, VT)};
  SDNode *BitIns = CurDAG->getMachineNode(Capstone::QC_INSBI, DL, VT, Ops);
  ReplaceNode(Node, BitIns);
  return true;
}

// Generate a QC_INSB/QC_INSBI from 'or (and X, MaskImm), OrImm' iff the value
// being inserted only sets known zero bits.
bool CapstoneDAGToDAGISel::tryBitfieldInsertOpFromOrAndImm(SDNode *Node) {
  // Supported only in Xqcibm for now.
  if (!Subtarget->hasVendorXqcibm())
    return false;

  using namespace SDPatternMatch;

  SDValue And;
  APInt MaskImm, OrImm;
  if (!sd_match(Node, m_Or(m_OneUse(m_And(m_Value(And), m_ConstInt(MaskImm))),
                           m_ConstInt(OrImm))))
    return false;

  // Compute the Known Zero for the AND as this allows us to catch more general
  // cases than just looking for AND with imm.
  KnownBits Known = CurDAG->computeKnownBits(Node->getOperand(0));

  // The bits being inserted must only set those bits that are known to be zero.
  if (!OrImm.isSubsetOf(Known.Zero)) {
    // FIXME:  It's okay if the OrImm sets NotKnownZero bits to 1, but we don't
    // currently handle this case.
    return false;
  }

  unsigned ShAmt, Width;
  // The KnownZero mask must be a shifted mask (e.g., 1110..011, 11100..00).
  if (!Known.Zero.isShiftedMask(ShAmt, Width))
    return false;

  // QC_INSB(I) dst, src, #width, #shamt.
  SDLoc DL(Node);
  MVT VT = Node->getSimpleValueType(0);
  SDValue ImmNode;
  auto Opc = Capstone::QC_INSB;

  int32_t LIImm = OrImm.getSExtValue() >> ShAmt;

  if (isInt<5>(LIImm)) {
    Opc = Capstone::QC_INSBI;
    ImmNode = CurDAG->getSignedTargetConstant(LIImm, DL, MVT::i32);
  } else {
    ImmNode = selectImm(CurDAG, DL, MVT::i32, LIImm, *Subtarget);
  }

  SDValue Ops[] = {And, ImmNode, CurDAG->getTargetConstant(Width, DL, VT),
                   CurDAG->getTargetConstant(ShAmt, DL, VT)};
  SDNode *BitIns = CurDAG->getMachineNode(Opc, DL, VT, Ops);
  ReplaceNode(Node, BitIns);
  return true;
}

bool CapstoneDAGToDAGISel::trySignedBitfieldInsertInSign(SDNode *Node) {
  // Only supported with XAndesPerf at the moment.
  if (!Subtarget->hasVendorXAndesPerf())
    return false;

  auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
  if (!N1C)
    return false;

  SDValue N0 = Node->getOperand(0);
  if (!N0.hasOneUse())
    return false;

  auto BitfieldInsert = [&](SDValue N0, unsigned Msb, unsigned Lsb,
                            const SDLoc &DL, MVT VT) {
    unsigned Opc = Capstone::NDS_BFOS;
    // If the Lsb is equal to the Msb, then the Lsb should be 0.
    if (Lsb == Msb)
      Lsb = 0;
    return CurDAG->getMachineNode(Opc, DL, VT, N0.getOperand(0),
                                  CurDAG->getTargetConstant(Lsb, DL, VT),
                                  CurDAG->getTargetConstant(Msb, DL, VT));
  };

  SDLoc DL(Node);
  MVT VT = Node->getSimpleValueType(0);
  const unsigned RightShAmt = N1C->getZExtValue();

  // Transform (sra (shl X, C1) C2) with C1 > C2
  //        -> (NDS.BFOS X, lsb, msb)
  if (N0.getOpcode() == ISD::SHL) {
    auto *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (!N01C)
      return false;

    const unsigned LeftShAmt = N01C->getZExtValue();
    // Make sure that this is a bitfield insertion (i.e., the shift-right
    // amount should be less than the left-shift).
    if (LeftShAmt <= RightShAmt)
      return false;

    const unsigned MsbPlusOne = VT.getSizeInBits() - RightShAmt;
    const unsigned Msb = MsbPlusOne - 1;
    const unsigned Lsb = LeftShAmt - RightShAmt;

    SDNode *Sbi = BitfieldInsert(N0, Msb, Lsb, DL, VT);
    ReplaceNode(Node, Sbi);
    return true;
  }

  return false;
}

bool CapstoneDAGToDAGISel::tryUnsignedBitfieldExtract(SDNode *Node,
                                                   const SDLoc &DL, MVT VT,
                                                   SDValue X, unsigned Msb,
                                                   unsigned Lsb) {
  unsigned Opc;

  if (Subtarget->hasVendorXTHeadBb()) {
    Opc = Capstone::TH_EXTU;
  } else if (Subtarget->hasVendorXAndesPerf()) {
    Opc = Capstone::NDS_BFOZ;
  } else if (Subtarget->hasVendorXqcibm()) {
    Opc = Capstone::QC_EXTU;
    // QC.EXTU X, width, shamt
    // shamt is the same as Lsb
    // width is the number of bits to extract from the Lsb
    Msb = Msb - Lsb + 1;
  } else {
    // Only supported with XTHeadBb/XAndesPerf/Xqcibm at the moment.
    return false;
  }

  SDNode *Ube = CurDAG->getMachineNode(Opc, DL, VT, X,
                                       CurDAG->getTargetConstant(Msb, DL, VT),
                                       CurDAG->getTargetConstant(Lsb, DL, VT));
  ReplaceNode(Node, Ube);
  return true;
}

bool CapstoneDAGToDAGISel::tryUnsignedBitfieldInsertInZero(SDNode *Node,
                                                        const SDLoc &DL, MVT VT,
                                                        SDValue X, unsigned Msb,
                                                        unsigned Lsb) {
  // Only supported with XAndesPerf at the moment.
  if (!Subtarget->hasVendorXAndesPerf())
    return false;

  unsigned Opc = Capstone::NDS_BFOZ;

  // If the Lsb is equal to the Msb, then the Lsb should be 0.
  if (Lsb == Msb)
    Lsb = 0;
  SDNode *Ubi = CurDAG->getMachineNode(Opc, DL, VT, X,
                                       CurDAG->getTargetConstant(Lsb, DL, VT),
                                       CurDAG->getTargetConstant(Msb, DL, VT));
  ReplaceNode(Node, Ubi);
  return true;
}

bool CapstoneDAGToDAGISel::tryIndexedLoad(SDNode *Node) {
  // Target does not support indexed loads.
  if (!Subtarget->hasVendorXTHeadMemIdx())
    return false;

  LoadSDNode *Ld = cast<LoadSDNode>(Node);
  ISD::MemIndexedMode AM = Ld->getAddressingMode();
  if (AM == ISD::UNINDEXED)
    return false;

  const ConstantSDNode *C = dyn_cast<ConstantSDNode>(Ld->getOffset());
  if (!C)
    return false;

  EVT LoadVT = Ld->getMemoryVT();
  assert((AM == ISD::PRE_INC || AM == ISD::POST_INC) &&
         "Unexpected addressing mode");
  bool IsPre = AM == ISD::PRE_INC;
  bool IsPost = AM == ISD::POST_INC;
  int64_t Offset = C->getSExtValue();

  // The constants that can be encoded in the THeadMemIdx instructions
  // are of the form (sign_extend(imm5) << imm2).
  unsigned Shift;
  for (Shift = 0; Shift < 4; Shift++)
    if (isInt<5>(Offset >> Shift) && ((Offset % (1LL << Shift)) == 0))
      break;

  // Constant cannot be encoded.
  if (Shift == 4)
    return false;

  bool IsZExt = (Ld->getExtensionType() == ISD::ZEXTLOAD);
  unsigned Opcode;
  if (LoadVT == MVT::i8 && IsPre)
    Opcode = IsZExt ? Capstone::TH_LBUIB : Capstone::TH_LBIB;
  else if (LoadVT == MVT::i8 && IsPost)
    Opcode = IsZExt ? Capstone::TH_LBUIA : Capstone::TH_LBIA;
  else if (LoadVT == MVT::i16 && IsPre)
    Opcode = IsZExt ? Capstone::TH_LHUIB : Capstone::TH_LHIB;
  else if (LoadVT == MVT::i16 && IsPost)
    Opcode = IsZExt ? Capstone::TH_LHUIA : Capstone::TH_LHIA;
  else if (LoadVT == MVT::i32 && IsPre)
    Opcode = IsZExt ? Capstone::TH_LWUIB : Capstone::TH_LWIB;
  else if (LoadVT == MVT::i32 && IsPost)
    Opcode = IsZExt ? Capstone::TH_LWUIA : Capstone::TH_LWIA;
  else if (LoadVT == MVT::i64 && IsPre)
    Opcode = Capstone::TH_LDIB;
  else if (LoadVT == MVT::i64 && IsPost)
    Opcode = Capstone::TH_LDIA;
  else
    return false;

  EVT Ty = Ld->getOffset().getValueType();
  SDValue Ops[] = {
      Ld->getBasePtr(),
      CurDAG->getSignedTargetConstant(Offset >> Shift, SDLoc(Node), Ty),
      CurDAG->getTargetConstant(Shift, SDLoc(Node), Ty), Ld->getChain()};
  SDNode *New = CurDAG->getMachineNode(Opcode, SDLoc(Node), Ld->getValueType(0),
                                       Ld->getValueType(1), MVT::Other, Ops);

  MachineMemOperand *MemOp = cast<MemSDNode>(Node)->getMemOperand();
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(New), {MemOp});

  ReplaceNode(Node, New);

  return true;
}

void CapstoneDAGToDAGISel::selectSF_VC_X_SE(SDNode *Node) {
  if (!Subtarget->hasVInstructions())
    return;

  assert(Node->getOpcode() == ISD::INTRINSIC_VOID && "Unexpected opcode");

  SDLoc DL(Node);
  unsigned IntNo = Node->getConstantOperandVal(1);

  assert((IntNo == Intrinsic::capstone_sf_vc_x_se ||
          IntNo == Intrinsic::capstone_sf_vc_i_se) &&
         "Unexpected vsetvli intrinsic");

  // imm, imm, imm, simm5/scalar, sew, log2lmul, vl
  unsigned Log2SEW = Log2_32(Node->getConstantOperandVal(6));
  SDValue SEWOp =
      CurDAG->getTargetConstant(Log2SEW, DL, Subtarget->getXLenVT());
  SmallVector<SDValue, 8> Operands = {Node->getOperand(2), Node->getOperand(3),
                                      Node->getOperand(4), Node->getOperand(5),
                                      Node->getOperand(8), SEWOp,
                                      Node->getOperand(0)};

  unsigned Opcode;
  auto *LMulSDNode = cast<ConstantSDNode>(Node->getOperand(7));
  switch (LMulSDNode->getSExtValue()) {
  case 5:
    Opcode = IntNo == Intrinsic::capstone_sf_vc_x_se ? Capstone::PseudoSF_VC_X_SE_MF8
                                                  : Capstone::PseudoSF_VC_I_SE_MF8;
    break;
  case 6:
    Opcode = IntNo == Intrinsic::capstone_sf_vc_x_se ? Capstone::PseudoSF_VC_X_SE_MF4
                                                  : Capstone::PseudoSF_VC_I_SE_MF4;
    break;
  case 7:
    Opcode = IntNo == Intrinsic::capstone_sf_vc_x_se ? Capstone::PseudoSF_VC_X_SE_MF2
                                                  : Capstone::PseudoSF_VC_I_SE_MF2;
    break;
  case 0:
    Opcode = IntNo == Intrinsic::capstone_sf_vc_x_se ? Capstone::PseudoSF_VC_X_SE_M1
                                                  : Capstone::PseudoSF_VC_I_SE_M1;
    break;
  case 1:
    Opcode = IntNo == Intrinsic::capstone_sf_vc_x_se ? Capstone::PseudoSF_VC_X_SE_M2
                                                  : Capstone::PseudoSF_VC_I_SE_M2;
    break;
  case 2:
    Opcode = IntNo == Intrinsic::capstone_sf_vc_x_se ? Capstone::PseudoSF_VC_X_SE_M4
                                                  : Capstone::PseudoSF_VC_I_SE_M4;
    break;
  case 3:
    Opcode = IntNo == Intrinsic::capstone_sf_vc_x_se ? Capstone::PseudoSF_VC_X_SE_M8
                                                  : Capstone::PseudoSF_VC_I_SE_M8;
    break;
  }

  ReplaceNode(Node, CurDAG->getMachineNode(
                        Opcode, DL, Node->getSimpleValueType(0), Operands));
}

static unsigned getSegInstNF(unsigned Intrinsic) {
#define INST_NF_CASE(NAME, NF)                                                 \
  case Intrinsic::capstone_##NAME##NF:                                            \
    return NF;
#define INST_NF_CASE_MASK(NAME, NF)                                            \
  case Intrinsic::capstone_##NAME##NF##_mask:                                     \
    return NF;
#define INST_NF_CASE_FF(NAME, NF)                                              \
  case Intrinsic::capstone_##NAME##NF##ff:                                        \
    return NF;
#define INST_NF_CASE_FF_MASK(NAME, NF)                                         \
  case Intrinsic::capstone_##NAME##NF##ff_mask:                                   \
    return NF;
#define INST_ALL_NF_CASE_BASE(MACRO_NAME, NAME)                                \
  MACRO_NAME(NAME, 2)                                                          \
  MACRO_NAME(NAME, 3)                                                          \
  MACRO_NAME(NAME, 4)                                                          \
  MACRO_NAME(NAME, 5)                                                          \
  MACRO_NAME(NAME, 6)                                                          \
  MACRO_NAME(NAME, 7)                                                          \
  MACRO_NAME(NAME, 8)
#define INST_ALL_NF_CASE(NAME)                                                 \
  INST_ALL_NF_CASE_BASE(INST_NF_CASE, NAME)                                    \
  INST_ALL_NF_CASE_BASE(INST_NF_CASE_MASK, NAME)
#define INST_ALL_NF_CASE_WITH_FF(NAME)                                         \
  INST_ALL_NF_CASE(NAME)                                                       \
  INST_ALL_NF_CASE_BASE(INST_NF_CASE_FF, NAME)                                 \
  INST_ALL_NF_CASE_BASE(INST_NF_CASE_FF_MASK, NAME)
  switch (Intrinsic) {
  default:
    llvm_unreachable("Unexpected segment load/store intrinsic");
    INST_ALL_NF_CASE_WITH_FF(vlseg)
    INST_ALL_NF_CASE(vlsseg)
    INST_ALL_NF_CASE(vloxseg)
    INST_ALL_NF_CASE(vluxseg)
    INST_ALL_NF_CASE(vsseg)
    INST_ALL_NF_CASE(vssseg)
    INST_ALL_NF_CASE(vsoxseg)
    INST_ALL_NF_CASE(vsuxseg)
  }
}

void CapstoneDAGToDAGISel::Select(SDNode *Node) {
  // If we have a custom node, we have already selected.
  if (Node->isMachineOpcode()) {
    LLVM_DEBUG(dbgs() << "== "; Node->dump(CurDAG); dbgs() << "\n");
    Node->setNodeId(-1);
    return;
  }

  // Instruction Selection not handled by the auto-generated tablegen selection
  // should be handled here.
  unsigned Opcode = Node->getOpcode();
  MVT XLenVT = Subtarget->getXLenVT();
  SDLoc DL(Node);
  MVT VT = Node->getSimpleValueType(0);

  bool HasBitTest = Subtarget->hasBEXTILike();

  switch (Opcode) {
  case ISD::Constant: {
    assert((VT == Subtarget->getXLenVT() || VT == MVT::i32) && "Unexpected VT");
    auto *ConstNode = cast<ConstantSDNode>(Node);
    if (ConstNode->isZero()) {
      SDValue New =
          CurDAG->getCopyFromReg(CurDAG->getEntryNode(), DL, Capstone::X0, VT);
      ReplaceNode(Node, New.getNode());
      return;
    }
    int64_t Imm = ConstNode->getSExtValue();
    // If only the lower 8 bits are used, try to convert this to a simm6 by
    // sign-extending bit 7. This is neutral without the C extension, and
    // allows C.LI to be used if C is present.
    if (isUInt<8>(Imm) && isInt<6>(SignExtend64<8>(Imm)) && hasAllBUsers(Node))
      Imm = SignExtend64<8>(Imm);
    // If the upper XLen-16 bits are not used, try to convert this to a simm12
    // by sign extending bit 15.
    if (isUInt<16>(Imm) && isInt<12>(SignExtend64<16>(Imm)) &&
        hasAllHUsers(Node))
      Imm = SignExtend64<16>(Imm);
    // If the upper 32-bits are not used try to convert this into a simm32 by
    // sign extending bit 32.
    if (!isInt<32>(Imm) && isUInt<32>(Imm) && hasAllWUsers(Node))
      Imm = SignExtend64<32>(Imm);

    ReplaceNode(Node, selectImm(CurDAG, DL, VT, Imm, *Subtarget).getNode());
    return;
  }
  case ISD::ConstantFP: {
    const APFloat &APF = cast<ConstantFPSDNode>(Node)->getValueAPF();

    bool Is64Bit = Subtarget->is64Bit();
    bool HasZdinx = Subtarget->hasStdExtZdinx();

    bool NegZeroF64 = APF.isNegZero() && VT == MVT::f64;
    SDValue Imm;
    // For +0.0 or f64 -0.0 we need to start from X0. For all others, we will
    // create an integer immediate.
    if (APF.isPosZero() || NegZeroF64) {
      if (VT == MVT::f64 && HasZdinx && !Is64Bit)
        Imm = CurDAG->getRegister(Capstone::X0_Pair, MVT::f64);
      else
        Imm = CurDAG->getRegister(Capstone::X0, XLenVT);
    } else {
      Imm = selectImm(CurDAG, DL, XLenVT, APF.bitcastToAPInt().getSExtValue(),
                      *Subtarget);
    }

    unsigned Opc;
    switch (VT.SimpleTy) {
    default:
      llvm_unreachable("Unexpected size");
    case MVT::bf16:
      assert(Subtarget->hasStdExtZfbfmin());
      Opc = Capstone::FMV_H_X;
      break;
    case MVT::f16:
      Opc = Subtarget->hasStdExtZhinxmin() ? Capstone::COPY : Capstone::FMV_H_X;
      break;
    case MVT::f32:
      Opc = Subtarget->hasStdExtZfinx() ? Capstone::COPY : Capstone::FMV_W_X;
      break;
    case MVT::f64:
      // For RV32, we can't move from a GPR, we need to convert instead. This
      // should only happen for +0.0 and -0.0.
      assert((Subtarget->is64Bit() || APF.isZero()) && "Unexpected constant");
      if (HasZdinx)
        Opc = Capstone::COPY;
      else
        Opc = Is64Bit ? Capstone::FMV_D_X : Capstone::FCVT_D_W;
      break;
    }

    SDNode *Res;
    if (VT.SimpleTy == MVT::f16 && Opc == Capstone::COPY) {
      Res =
          CurDAG->getTargetExtractSubreg(Capstone::sub_16, DL, VT, Imm).getNode();
    } else if (VT.SimpleTy == MVT::f32 && Opc == Capstone::COPY) {
      Res =
          CurDAG->getTargetExtractSubreg(Capstone::sub_32, DL, VT, Imm).getNode();
    } else if (Opc == Capstone::FCVT_D_W_IN32X || Opc == Capstone::FCVT_D_W)
      Res = CurDAG->getMachineNode(
          Opc, DL, VT, Imm,
          CurDAG->getTargetConstant(CapstoneFPRndMode::RNE, DL, XLenVT));
    else
      Res = CurDAG->getMachineNode(Opc, DL, VT, Imm);

    // For f64 -0.0, we need to insert a fneg.d idiom.
    if (NegZeroF64) {
      Opc = Capstone::FSGNJN_D;
      if (HasZdinx)
        Opc = Is64Bit ? Capstone::FSGNJN_D_INX : Capstone::FSGNJN_D_IN32X;
      Res =
          CurDAG->getMachineNode(Opc, DL, VT, SDValue(Res, 0), SDValue(Res, 0));
    }

    ReplaceNode(Node, Res);
    return;
  }
  case CapstoneISD::BuildGPRPair:
  case CapstoneISD::BuildPairF64: {
    if (Opcode == CapstoneISD::BuildPairF64 && !Subtarget->hasStdExtZdinx())
      break;

    assert((!Subtarget->is64Bit() || Opcode == CapstoneISD::BuildGPRPair) &&
           "BuildPairF64 only handled here on rv32i_zdinx");

    SDValue Ops[] = {
        CurDAG->getTargetConstant(Capstone::GPRPairRegClassID, DL, MVT::i32),
        Node->getOperand(0),
        CurDAG->getTargetConstant(Capstone::sub_gpr_even, DL, MVT::i32),
        Node->getOperand(1),
        CurDAG->getTargetConstant(Capstone::sub_gpr_odd, DL, MVT::i32)};

    SDNode *N = CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, DL, VT, Ops);
    ReplaceNode(Node, N);
    return;
  }
  case CapstoneISD::SplitGPRPair:
  case CapstoneISD::SplitF64: {
    if (Subtarget->hasStdExtZdinx() || Opcode != CapstoneISD::SplitF64) {
      assert((!Subtarget->is64Bit() || Opcode == CapstoneISD::SplitGPRPair) &&
             "SplitF64 only handled here on rv32i_zdinx");

      if (!SDValue(Node, 0).use_empty()) {
        SDValue Lo = CurDAG->getTargetExtractSubreg(Capstone::sub_gpr_even, DL,
                                                    Node->getValueType(0),
                                                    Node->getOperand(0));
        ReplaceUses(SDValue(Node, 0), Lo);
      }

      if (!SDValue(Node, 1).use_empty()) {
        SDValue Hi = CurDAG->getTargetExtractSubreg(
            Capstone::sub_gpr_odd, DL, Node->getValueType(1), Node->getOperand(0));
        ReplaceUses(SDValue(Node, 1), Hi);
      }

      CurDAG->RemoveDeadNode(Node);
      return;
    }

    assert(Opcode != CapstoneISD::SplitGPRPair &&
           "SplitGPRPair should already be handled");

    if (!Subtarget->hasStdExtZfa())
      break;
    assert(Subtarget->hasStdExtD() && !Subtarget->is64Bit() &&
           "Unexpected subtarget");

    // With Zfa, lower to fmv.x.w and fmvh.x.d.
    if (!SDValue(Node, 0).use_empty()) {
      SDNode *Lo = CurDAG->getMachineNode(Capstone::FMV_X_W_FPR64, DL, VT,
                                          Node->getOperand(0));
      ReplaceUses(SDValue(Node, 0), SDValue(Lo, 0));
    }
    if (!SDValue(Node, 1).use_empty()) {
      SDNode *Hi = CurDAG->getMachineNode(Capstone::FMVH_X_D, DL, VT,
                                          Node->getOperand(0));
      ReplaceUses(SDValue(Node, 1), SDValue(Hi, 0));
    }

    CurDAG->RemoveDeadNode(Node);
    return;
  }
  case ISD::SHL: {
    auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
    if (!N1C)
      break;
    SDValue N0 = Node->getOperand(0);
    if (N0.getOpcode() != ISD::AND || !N0.hasOneUse() ||
        !isa<ConstantSDNode>(N0.getOperand(1)))
      break;
    unsigned ShAmt = N1C->getZExtValue();
    uint64_t Mask = N0.getConstantOperandVal(1);

    if (isShiftedMask_64(Mask)) {
      unsigned XLen = Subtarget->getXLen();
      unsigned LeadingZeros = XLen - llvm::bit_width(Mask);
      unsigned TrailingZeros = llvm::countr_zero(Mask);
      if (ShAmt <= 32 && TrailingZeros > 0 && LeadingZeros == 32) {
        // Optimize (shl (and X, C2), C) -> (slli (srliw X, C3), C3+C)
        // where C2 has 32 leading zeros and C3 trailing zeros.
        SDNode *SRLIW = CurDAG->getMachineNode(
            Capstone::SRLIW, DL, VT, N0.getOperand(0),
            CurDAG->getTargetConstant(TrailingZeros, DL, VT));
        SDNode *SLLI = CurDAG->getMachineNode(
            Capstone::SLLI, DL, VT, SDValue(SRLIW, 0),
            CurDAG->getTargetConstant(TrailingZeros + ShAmt, DL, VT));
        ReplaceNode(Node, SLLI);
        return;
      }
      if (TrailingZeros == 0 && LeadingZeros > ShAmt &&
          XLen - LeadingZeros > 11 && LeadingZeros != 32) {
        // Optimize (shl (and X, C2), C) -> (srli (slli X, C4), C4-C)
        // where C2 has C4 leading zeros and no trailing zeros.
        // This is profitable if the "and" was to be lowered to
        // (srli (slli X, C4), C4) and not (andi X, C2).
        // For "LeadingZeros == 32":
        // - with Zba it's just (slli.uw X, C)
        // - without Zba a tablegen pattern applies the very same
        //   transform as we would have done here
        SDNode *SLLI = CurDAG->getMachineNode(
            Capstone::SLLI, DL, VT, N0.getOperand(0),
            CurDAG->getTargetConstant(LeadingZeros, DL, VT));
        SDNode *SRLI = CurDAG->getMachineNode(
            Capstone::SRLI, DL, VT, SDValue(SLLI, 0),
            CurDAG->getTargetConstant(LeadingZeros - ShAmt, DL, VT));
        ReplaceNode(Node, SRLI);
        return;
      }
    }
    break;
  }
  case ISD::SRL: {
    auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
    if (!N1C)
      break;
    SDValue N0 = Node->getOperand(0);
    if (N0.getOpcode() != ISD::AND || !isa<ConstantSDNode>(N0.getOperand(1)))
      break;
    unsigned ShAmt = N1C->getZExtValue();
    uint64_t Mask = N0.getConstantOperandVal(1);

    // Optimize (srl (and X, C2), C) -> (slli (srliw X, C3), C3-C) where C2 has
    // 32 leading zeros and C3 trailing zeros.
    if (isShiftedMask_64(Mask) && N0.hasOneUse()) {
      unsigned XLen = Subtarget->getXLen();
      unsigned LeadingZeros = XLen - llvm::bit_width(Mask);
      unsigned TrailingZeros = llvm::countr_zero(Mask);
      if (LeadingZeros == 32 && TrailingZeros > ShAmt) {
        SDNode *SRLIW = CurDAG->getMachineNode(
            Capstone::SRLIW, DL, VT, N0.getOperand(0),
            CurDAG->getTargetConstant(TrailingZeros, DL, VT));
        SDNode *SLLI = CurDAG->getMachineNode(
            Capstone::SLLI, DL, VT, SDValue(SRLIW, 0),
            CurDAG->getTargetConstant(TrailingZeros - ShAmt, DL, VT));
        ReplaceNode(Node, SLLI);
        return;
      }
    }

    // Optimize (srl (and X, C2), C) ->
    //          (srli (slli X, (XLen-C3), (XLen-C3) + C)
    // Where C2 is a mask with C3 trailing ones.
    // Taking into account that the C2 may have had lower bits unset by
    // SimplifyDemandedBits. This avoids materializing the C2 immediate.
    // This pattern occurs when type legalizing right shifts for types with
    // less than XLen bits.
    Mask |= maskTrailingOnes<uint64_t>(ShAmt);
    if (!isMask_64(Mask))
      break;
    unsigned TrailingOnes = llvm::countr_one(Mask);
    if (ShAmt >= TrailingOnes)
      break;
    // If the mask has 32 trailing ones, use SRLI on RV32 or SRLIW on RV64.
    if (TrailingOnes == 32) {
      SDNode *SRLI = CurDAG->getMachineNode(
          Subtarget->is64Bit() ? Capstone::SRLIW : Capstone::SRLI, DL, VT,
          N0.getOperand(0), CurDAG->getTargetConstant(ShAmt, DL, VT));
      ReplaceNode(Node, SRLI);
      return;
    }

    // Only do the remaining transforms if the AND has one use.
    if (!N0.hasOneUse())
      break;

    // If C2 is (1 << ShAmt) use bexti or th.tst if possible.
    if (HasBitTest && ShAmt + 1 == TrailingOnes) {
      SDNode *BEXTI = CurDAG->getMachineNode(
          Subtarget->hasStdExtZbs() ? Capstone::BEXTI : Capstone::TH_TST, DL, VT,
          N0.getOperand(0), CurDAG->getTargetConstant(ShAmt, DL, VT));
      ReplaceNode(Node, BEXTI);
      return;
    }

    const unsigned Msb = TrailingOnes - 1;
    const unsigned Lsb = ShAmt;
    if (tryUnsignedBitfieldExtract(Node, DL, VT, N0.getOperand(0), Msb, Lsb))
      return;

    unsigned LShAmt = Subtarget->getXLen() - TrailingOnes;
    SDNode *SLLI =
        CurDAG->getMachineNode(Capstone::SLLI, DL, VT, N0.getOperand(0),
                               CurDAG->getTargetConstant(LShAmt, DL, VT));
    SDNode *SRLI = CurDAG->getMachineNode(
        Capstone::SRLI, DL, VT, SDValue(SLLI, 0),
        CurDAG->getTargetConstant(LShAmt + ShAmt, DL, VT));
    ReplaceNode(Node, SRLI);
    return;
  }
  case ISD::SRA: {
    if (trySignedBitfieldExtract(Node))
      return;

    if (trySignedBitfieldInsertInSign(Node))
      return;

    // Optimize (sra (sext_inreg X, i16), C) ->
    //          (srai (slli X, (XLen-16), (XLen-16) + C)
    // And      (sra (sext_inreg X, i8), C) ->
    //          (srai (slli X, (XLen-8), (XLen-8) + C)
    // This can occur when Zbb is enabled, which makes sext_inreg i16/i8 legal.
    // This transform matches the code we get without Zbb. The shifts are more
    // compressible, and this can help expose CSE opportunities in the sdiv by
    // constant optimization.
    auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
    if (!N1C)
      break;
    SDValue N0 = Node->getOperand(0);
    if (N0.getOpcode() != ISD::SIGN_EXTEND_INREG || !N0.hasOneUse())
      break;
    unsigned ShAmt = N1C->getZExtValue();
    unsigned ExtSize =
        cast<VTSDNode>(N0.getOperand(1))->getVT().getSizeInBits();
    // ExtSize of 32 should use sraiw via tablegen pattern.
    if (ExtSize >= 32 || ShAmt >= ExtSize)
      break;
    unsigned LShAmt = Subtarget->getXLen() - ExtSize;
    SDNode *SLLI =
        CurDAG->getMachineNode(Capstone::SLLI, DL, VT, N0.getOperand(0),
                               CurDAG->getTargetConstant(LShAmt, DL, VT));
    SDNode *SRAI = CurDAG->getMachineNode(
        Capstone::SRAI, DL, VT, SDValue(SLLI, 0),
        CurDAG->getTargetConstant(LShAmt + ShAmt, DL, VT));
    ReplaceNode(Node, SRAI);
    return;
  }
  case ISD::OR: {
    if (trySignedBitfieldInsertInMask(Node))
      return;

    if (tryBitfieldInsertOpFromOrAndImm(Node))
      return;

    if (tryShrinkShlLogicImm(Node))
      return;

    break;
  }
  case ISD::XOR:
    if (tryShrinkShlLogicImm(Node))
      return;

    break;
  case ISD::AND: {
    auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
    if (!N1C)
      break;

    SDValue N0 = Node->getOperand(0);

    bool LeftShift = N0.getOpcode() == ISD::SHL;
    if (LeftShift || N0.getOpcode() == ISD::SRL) {
      auto *C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
      if (!C)
        break;
      unsigned C2 = C->getZExtValue();
      unsigned XLen = Subtarget->getXLen();
      assert((C2 > 0 && C2 < XLen) && "Unexpected shift amount!");

      // Keep track of whether this is a c.andi. If we can't use c.andi, the
      // shift pair might offer more compression opportunities.
      // TODO: We could check for C extension here, but we don't have many lit
      // tests with the C extension enabled so not checking gets better
      // coverage.
      // TODO: What if ANDI faster than shift?
      bool IsCANDI = isInt<6>(N1C->getSExtValue());

      uint64_t C1 = N1C->getZExtValue();

      // Clear irrelevant bits in the mask.
      if (LeftShift)
        C1 &= maskTrailingZeros<uint64_t>(C2);
      else
        C1 &= maskTrailingOnes<uint64_t>(XLen - C2);

      // Some transforms should only be done if the shift has a single use or
      // the AND would become (srli (slli X, 32), 32)
      bool OneUseOrZExtW = N0.hasOneUse() || C1 == UINT64_C(0xFFFFFFFF);

      SDValue X = N0.getOperand(0);

      // Turn (and (srl x, c2) c1) -> (srli (slli x, c3-c2), c3) if c1 is a mask
      // with c3 leading zeros.
      if (!LeftShift && isMask_64(C1)) {
        unsigned Leading = XLen - llvm::bit_width(C1);
        if (C2 < Leading) {
          // If the number of leading zeros is C2+32 this can be SRLIW.
          if (C2 + 32 == Leading) {
            SDNode *SRLIW = CurDAG->getMachineNode(
                Capstone::SRLIW, DL, VT, X, CurDAG->getTargetConstant(C2, DL, VT));
            ReplaceNode(Node, SRLIW);
            return;
          }

          // (and (srl (sexti32 Y), c2), c1) -> (srliw (sraiw Y, 31), c3 - 32)
          // if c1 is a mask with c3 leading zeros and c2 >= 32 and c3-c2==1.
          //
          // This pattern occurs when (i32 (srl (sra 31), c3 - 32)) is type
          // legalized and goes through DAG combine.
          if (C2 >= 32 && (Leading - C2) == 1 && N0.hasOneUse() &&
              X.getOpcode() == ISD::SIGN_EXTEND_INREG &&
              cast<VTSDNode>(X.getOperand(1))->getVT() == MVT::i32) {
            SDNode *SRAIW =
                CurDAG->getMachineNode(Capstone::SRAIW, DL, VT, X.getOperand(0),
                                       CurDAG->getTargetConstant(31, DL, VT));
            SDNode *SRLIW = CurDAG->getMachineNode(
                Capstone::SRLIW, DL, VT, SDValue(SRAIW, 0),
                CurDAG->getTargetConstant(Leading - 32, DL, VT));
            ReplaceNode(Node, SRLIW);
            return;
          }

          // Try to use an unsigned bitfield extract (e.g., th.extu) if
          // available.
          // Transform (and (srl x, C2), C1)
          //        -> (<bfextract> x, msb, lsb)
          //
          // Make sure to keep this below the SRLIW cases, as we always want to
          // prefer the more common instruction.
          const unsigned Msb = llvm::bit_width(C1) + C2 - 1;
          const unsigned Lsb = C2;
          if (tryUnsignedBitfieldExtract(Node, DL, VT, X, Msb, Lsb))
            return;

          // (srli (slli x, c3-c2), c3).
          // Skip if we could use (zext.w (sraiw X, C2)).
          bool Skip = Subtarget->hasStdExtZba() && Leading == 32 &&
                      X.getOpcode() == ISD::SIGN_EXTEND_INREG &&
                      cast<VTSDNode>(X.getOperand(1))->getVT() == MVT::i32;
          // Also Skip if we can use bexti or th.tst.
          Skip |= HasBitTest && Leading == XLen - 1;
          if (OneUseOrZExtW && !Skip) {
            SDNode *SLLI = CurDAG->getMachineNode(
                Capstone::SLLI, DL, VT, X,
                CurDAG->getTargetConstant(Leading - C2, DL, VT));
            SDNode *SRLI = CurDAG->getMachineNode(
                Capstone::SRLI, DL, VT, SDValue(SLLI, 0),
                CurDAG->getTargetConstant(Leading, DL, VT));
            ReplaceNode(Node, SRLI);
            return;
          }
        }
      }

      // Turn (and (shl x, c2), c1) -> (srli (slli c2+c3), c3) if c1 is a mask
      // shifted by c2 bits with c3 leading zeros.
      if (LeftShift && isShiftedMask_64(C1)) {
        unsigned Leading = XLen - llvm::bit_width(C1);

        if (C2 + Leading < XLen &&
            C1 == (maskTrailingOnes<uint64_t>(XLen - (C2 + Leading)) << C2)) {
          // Use slli.uw when possible.
          if ((XLen - (C2 + Leading)) == 32 && Subtarget->hasStdExtZba()) {
            SDNode *SLLI_UW =
                CurDAG->getMachineNode(Capstone::SLLI_UW, DL, VT, X,
                                       CurDAG->getTargetConstant(C2, DL, VT));
            ReplaceNode(Node, SLLI_UW);
            return;
          }

          // Try to use an unsigned bitfield insert (e.g., nds.bfoz) if
          // available.
          // Transform (and (shl x, c2), c1)
          //        -> (<bfinsert> x, msb, lsb)
          // e.g.
          //     (and (shl x, 12), 0x00fff000)
          //     If XLen = 32 and C2 = 12, then
          //     Msb = 32 - 8 - 1 = 23 and Lsb = 12
          const unsigned Msb = XLen - Leading - 1;
          const unsigned Lsb = C2;
          if (tryUnsignedBitfieldInsertInZero(Node, DL, VT, X, Msb, Lsb))
            return;

          // (srli (slli c2+c3), c3)
          if (OneUseOrZExtW && !IsCANDI) {
            SDNode *SLLI = CurDAG->getMachineNode(
                Capstone::SLLI, DL, VT, X,
                CurDAG->getTargetConstant(C2 + Leading, DL, VT));
            SDNode *SRLI = CurDAG->getMachineNode(
                Capstone::SRLI, DL, VT, SDValue(SLLI, 0),
                CurDAG->getTargetConstant(Leading, DL, VT));
            ReplaceNode(Node, SRLI);
            return;
          }
        }
      }

      // Turn (and (shr x, c2), c1) -> (slli (srli x, c2+c3), c3) if c1 is a
      // shifted mask with c2 leading zeros and c3 trailing zeros.
      if (!LeftShift && isShiftedMask_64(C1)) {
        unsigned Leading = XLen - llvm::bit_width(C1);
        unsigned Trailing = llvm::countr_zero(C1);
        if (Leading == C2 && C2 + Trailing < XLen && OneUseOrZExtW &&
            !IsCANDI) {
          unsigned SrliOpc = Capstone::SRLI;
          // If the input is zexti32 we should use SRLIW.
          if (X.getOpcode() == ISD::AND &&
              isa<ConstantSDNode>(X.getOperand(1)) &&
              X.getConstantOperandVal(1) == UINT64_C(0xFFFFFFFF)) {
            SrliOpc = Capstone::SRLIW;
            X = X.getOperand(0);
          }
          SDNode *SRLI = CurDAG->getMachineNode(
              SrliOpc, DL, VT, X,
              CurDAG->getTargetConstant(C2 + Trailing, DL, VT));
          SDNode *SLLI = CurDAG->getMachineNode(
              Capstone::SLLI, DL, VT, SDValue(SRLI, 0),
              CurDAG->getTargetConstant(Trailing, DL, VT));
          ReplaceNode(Node, SLLI);
          return;
        }
        // If the leading zero count is C2+32, we can use SRLIW instead of SRLI.
        if (Leading > 32 && (Leading - 32) == C2 && C2 + Trailing < 32 &&
            OneUseOrZExtW && !IsCANDI) {
          SDNode *SRLIW = CurDAG->getMachineNode(
              Capstone::SRLIW, DL, VT, X,
              CurDAG->getTargetConstant(C2 + Trailing, DL, VT));
          SDNode *SLLI = CurDAG->getMachineNode(
              Capstone::SLLI, DL, VT, SDValue(SRLIW, 0),
              CurDAG->getTargetConstant(Trailing, DL, VT));
          ReplaceNode(Node, SLLI);
          return;
        }
        // If we have 32 bits in the mask, we can use SLLI_UW instead of SLLI.
        if (Trailing > 0 && Leading + Trailing == 32 && C2 + Trailing < XLen &&
            OneUseOrZExtW && Subtarget->hasStdExtZba()) {
          SDNode *SRLI = CurDAG->getMachineNode(
              Capstone::SRLI, DL, VT, X,
              CurDAG->getTargetConstant(C2 + Trailing, DL, VT));
          SDNode *SLLI_UW = CurDAG->getMachineNode(
              Capstone::SLLI_UW, DL, VT, SDValue(SRLI, 0),
              CurDAG->getTargetConstant(Trailing, DL, VT));
          ReplaceNode(Node, SLLI_UW);
          return;
        }
      }

      // Turn (and (shl x, c2), c1) -> (slli (srli x, c3-c2), c3) if c1 is a
      // shifted mask with no leading zeros and c3 trailing zeros.
      if (LeftShift && isShiftedMask_64(C1)) {
        unsigned Leading = XLen - llvm::bit_width(C1);
        unsigned Trailing = llvm::countr_zero(C1);
        if (Leading == 0 && C2 < Trailing && OneUseOrZExtW && !IsCANDI) {
          SDNode *SRLI = CurDAG->getMachineNode(
              Capstone::SRLI, DL, VT, X,
              CurDAG->getTargetConstant(Trailing - C2, DL, VT));
          SDNode *SLLI = CurDAG->getMachineNode(
              Capstone::SLLI, DL, VT, SDValue(SRLI, 0),
              CurDAG->getTargetConstant(Trailing, DL, VT));
          ReplaceNode(Node, SLLI);
          return;
        }
        // If we have (32-C2) leading zeros, we can use SRLIW instead of SRLI.
        if (C2 < Trailing && Leading + C2 == 32 && OneUseOrZExtW && !IsCANDI) {
          SDNode *SRLIW = CurDAG->getMachineNode(
              Capstone::SRLIW, DL, VT, X,
              CurDAG->getTargetConstant(Trailing - C2, DL, VT));
          SDNode *SLLI = CurDAG->getMachineNode(
              Capstone::SLLI, DL, VT, SDValue(SRLIW, 0),
              CurDAG->getTargetConstant(Trailing, DL, VT));
          ReplaceNode(Node, SLLI);
          return;
        }

        // If we have 32 bits in the mask, we can use SLLI_UW instead of SLLI.
        if (C2 < Trailing && Leading + Trailing == 32 && OneUseOrZExtW &&
            Subtarget->hasStdExtZba()) {
          SDNode *SRLI = CurDAG->getMachineNode(
              Capstone::SRLI, DL, VT, X,
              CurDAG->getTargetConstant(Trailing - C2, DL, VT));
          SDNode *SLLI_UW = CurDAG->getMachineNode(
              Capstone::SLLI_UW, DL, VT, SDValue(SRLI, 0),
              CurDAG->getTargetConstant(Trailing, DL, VT));
          ReplaceNode(Node, SLLI_UW);
          return;
        }
      }
    }

    const uint64_t C1 = N1C->getZExtValue();

    if (N0.getOpcode() == ISD::SRA && isa<ConstantSDNode>(N0.getOperand(1)) &&
        N0.hasOneUse()) {
      unsigned C2 = N0.getConstantOperandVal(1);
      unsigned XLen = Subtarget->getXLen();
      assert((C2 > 0 && C2 < XLen) && "Unexpected shift amount!");

      SDValue X = N0.getOperand(0);

      // Prefer SRAIW + ANDI when possible.
      bool Skip = C2 > 32 && isInt<12>(N1C->getSExtValue()) &&
                  X.getOpcode() == ISD::SHL &&
                  isa<ConstantSDNode>(X.getOperand(1)) &&
                  X.getConstantOperandVal(1) == 32;
      // Turn (and (sra x, c2), c1) -> (srli (srai x, c2-c3), c3) if c1 is a
      // mask with c3 leading zeros and c2 is larger than c3.
      if (isMask_64(C1) && !Skip) {
        unsigned Leading = XLen - llvm::bit_width(C1);
        if (C2 > Leading) {
          SDNode *SRAI = CurDAG->getMachineNode(
              Capstone::SRAI, DL, VT, X,
              CurDAG->getTargetConstant(C2 - Leading, DL, VT));
          SDNode *SRLI = CurDAG->getMachineNode(
              Capstone::SRLI, DL, VT, SDValue(SRAI, 0),
              CurDAG->getTargetConstant(Leading, DL, VT));
          ReplaceNode(Node, SRLI);
          return;
        }
      }

      // Look for (and (sra y, c2), c1) where c1 is a shifted mask with c3
      // leading zeros and c4 trailing zeros. If c2 is greater than c3, we can
      // use (slli (srli (srai y, c2 - c3), c3 + c4), c4).
      if (isShiftedMask_64(C1) && !Skip) {
        unsigned Leading = XLen - llvm::bit_width(C1);
        unsigned Trailing = llvm::countr_zero(C1);
        if (C2 > Leading && Leading > 0 && Trailing > 0) {
          SDNode *SRAI = CurDAG->getMachineNode(
              Capstone::SRAI, DL, VT, N0.getOperand(0),
              CurDAG->getTargetConstant(C2 - Leading, DL, VT));
          SDNode *SRLI = CurDAG->getMachineNode(
              Capstone::SRLI, DL, VT, SDValue(SRAI, 0),
              CurDAG->getTargetConstant(Leading + Trailing, DL, VT));
          SDNode *SLLI = CurDAG->getMachineNode(
              Capstone::SLLI, DL, VT, SDValue(SRLI, 0),
              CurDAG->getTargetConstant(Trailing, DL, VT));
          ReplaceNode(Node, SLLI);
          return;
        }
      }
    }

    // If C1 masks off the upper bits only (but can't be formed as an
    // ANDI), use an unsigned bitfield extract (e.g., th.extu), if
    // available.
    // Transform (and x, C1)
    //        -> (<bfextract> x, msb, lsb)
    if (isMask_64(C1) && !isInt<12>(N1C->getSExtValue()) &&
        !(C1 == 0xffff && Subtarget->hasStdExtZbb()) &&
        !(C1 == 0xffffffff && Subtarget->hasStdExtZba())) {
      const unsigned Msb = llvm::bit_width(C1) - 1;
      if (tryUnsignedBitfieldExtract(Node, DL, VT, N0, Msb, 0))
        return;
    }

    if (tryShrinkShlLogicImm(Node))
      return;

    break;
  }
  case ISD::MUL: {
    // Special case for calculating (mul (and X, C2), C1) where the full product
    // fits in XLen bits. We can shift X left by the number of leading zeros in
    // C2 and shift C1 left by XLen-lzcnt(C2). This will ensure the final
    // product has XLen trailing zeros, putting it in the output of MULHU. This
    // can avoid materializing a constant in a register for C2.

    // RHS should be a constant.
    auto *N1C = dyn_cast<ConstantSDNode>(Node->getOperand(1));
    if (!N1C || !N1C->hasOneUse())
      break;

    // LHS should be an AND with constant.
    SDValue N0 = Node->getOperand(0);
    if (N0.getOpcode() != ISD::AND || !isa<ConstantSDNode>(N0.getOperand(1)))
      break;

    uint64_t C2 = N0.getConstantOperandVal(1);

    // Constant should be a mask.
    if (!isMask_64(C2))
      break;

    // If this can be an ANDI or ZEXT.H, don't do this if the ANDI/ZEXT has
    // multiple users or the constant is a simm12. This prevents inserting a
    // shift and still have uses of the AND/ZEXT. Shifting a simm12 will likely
    // make it more costly to materialize. Otherwise, using a SLLI might allow
    // it to be compressed.
    bool IsANDIOrZExt =
        isInt<12>(C2) ||
        (C2 == UINT64_C(0xFFFF) && Subtarget->hasStdExtZbb());
    // With XTHeadBb, we can use TH.EXTU.
    IsANDIOrZExt |= C2 == UINT64_C(0xFFFF) && Subtarget->hasVendorXTHeadBb();
    if (IsANDIOrZExt && (isInt<12>(N1C->getSExtValue()) || !N0.hasOneUse()))
      break;
    // If this can be a ZEXT.w, don't do this if the ZEXT has multiple users or
    // the constant is a simm32.
    bool IsZExtW = C2 == UINT64_C(0xFFFFFFFF) && Subtarget->hasStdExtZba();
    // With XTHeadBb, we can use TH.EXTU.
    IsZExtW |= C2 == UINT64_C(0xFFFFFFFF) && Subtarget->hasVendorXTHeadBb();
    if (IsZExtW && (isInt<32>(N1C->getSExtValue()) || !N0.hasOneUse()))
      break;

    // We need to shift left the AND input and C1 by a total of XLen bits.

    // How far left do we need to shift the AND input?
    unsigned XLen = Subtarget->getXLen();
    unsigned LeadingZeros = XLen - llvm::bit_width(C2);

    // The constant gets shifted by the remaining amount unless that would
    // shift bits out.
    uint64_t C1 = N1C->getZExtValue();
    unsigned ConstantShift = XLen - LeadingZeros;
    if (ConstantShift > (XLen - llvm::bit_width(C1)))
      break;

    uint64_t ShiftedC1 = C1 << ConstantShift;
    // If this RV32, we need to sign extend the constant.
    if (XLen == 32)
      ShiftedC1 = SignExtend64<32>(ShiftedC1);

    // Create (mulhu (slli X, lzcnt(C2)), C1 << (XLen - lzcnt(C2))).
    SDNode *Imm = selectImm(CurDAG, DL, VT, ShiftedC1, *Subtarget).getNode();
    SDNode *SLLI =
        CurDAG->getMachineNode(Capstone::SLLI, DL, VT, N0.getOperand(0),
                               CurDAG->getTargetConstant(LeadingZeros, DL, VT));
    SDNode *MULHU = CurDAG->getMachineNode(Capstone::MULHU, DL, VT,
                                           SDValue(SLLI, 0), SDValue(Imm, 0));
    ReplaceNode(Node, MULHU);
    return;
  }
  case ISD::LOAD: {
    if (tryIndexedLoad(Node))
      return;

    if (Subtarget->hasVendorXCVmem() && !Subtarget->is64Bit()) {
      // We match post-incrementing load here
      LoadSDNode *Load = cast<LoadSDNode>(Node);
      if (Load->getAddressingMode() != ISD::POST_INC)
        break;

      SDValue Chain = Node->getOperand(0);
      SDValue Base = Node->getOperand(1);
      SDValue Offset = Node->getOperand(2);

      bool Simm12 = false;
      bool SignExtend = Load->getExtensionType() == ISD::SEXTLOAD;

      if (auto ConstantOffset = dyn_cast<ConstantSDNode>(Offset)) {
        int ConstantVal = ConstantOffset->getSExtValue();
        Simm12 = isInt<12>(ConstantVal);
        if (Simm12)
          Offset = CurDAG->getTargetConstant(ConstantVal, SDLoc(Offset),
                                             Offset.getValueType());
      }

      unsigned Opcode = 0;
      switch (Load->getMemoryVT().getSimpleVT().SimpleTy) {
      case MVT::i8:
        if (Simm12 && SignExtend)
          Opcode = Capstone::CV_LB_ri_inc;
        else if (Simm12 && !SignExtend)
          Opcode = Capstone::CV_LBU_ri_inc;
        else if (!Simm12 && SignExtend)
          Opcode = Capstone::CV_LB_rr_inc;
        else
          Opcode = Capstone::CV_LBU_rr_inc;
        break;
      case MVT::i16:
        if (Simm12 && SignExtend)
          Opcode = Capstone::CV_LH_ri_inc;
        else if (Simm12 && !SignExtend)
          Opcode = Capstone::CV_LHU_ri_inc;
        else if (!Simm12 && SignExtend)
          Opcode = Capstone::CV_LH_rr_inc;
        else
          Opcode = Capstone::CV_LHU_rr_inc;
        break;
      case MVT::i32:
        if (Simm12)
          Opcode = Capstone::CV_LW_ri_inc;
        else
          Opcode = Capstone::CV_LW_rr_inc;
        break;
      default:
        break;
      }
      if (!Opcode)
        break;

      ReplaceNode(Node, CurDAG->getMachineNode(Opcode, DL, XLenVT, XLenVT,
                                               Chain.getSimpleValueType(), Base,
                                               Offset, Chain));
      return;
    }
    break;
  }
  case CapstoneISD::LD_RV32: {
    assert(Subtarget->hasStdExtZilsd() && "LD_RV32 is only used with Zilsd");

    SDValue Base, Offset;
    SDValue Chain = Node->getOperand(0);
    SDValue Addr = Node->getOperand(1);
    SelectAddrRegImm(Addr, Base, Offset);

    SDValue Ops[] = {Base, Offset, Chain};
    MachineSDNode *New = CurDAG->getMachineNode(
        Capstone::LD_RV32, DL, {MVT::Untyped, MVT::Other}, Ops);
    SDValue Lo = CurDAG->getTargetExtractSubreg(Capstone::sub_gpr_even, DL,
                                                MVT::i32, SDValue(New, 0));
    SDValue Hi = CurDAG->getTargetExtractSubreg(Capstone::sub_gpr_odd, DL,
                                                MVT::i32, SDValue(New, 0));
    CurDAG->setNodeMemRefs(New, {cast<MemSDNode>(Node)->getMemOperand()});
    ReplaceUses(SDValue(Node, 0), Lo);
    ReplaceUses(SDValue(Node, 1), Hi);
    ReplaceUses(SDValue(Node, 2), SDValue(New, 1));
    CurDAG->RemoveDeadNode(Node);
    return;
  }
  case CapstoneISD::SD_RV32: {
    SDValue Base, Offset;
    SDValue Chain = Node->getOperand(0);
    SDValue Addr = Node->getOperand(3);
    SelectAddrRegImm(Addr, Base, Offset);

    SDValue Lo = Node->getOperand(1);
    SDValue Hi = Node->getOperand(2);

    SDValue RegPair;
    // Peephole to use X0_Pair for storing zero.
    if (isNullConstant(Lo) && isNullConstant(Hi)) {
      RegPair = CurDAG->getRegister(Capstone::X0_Pair, MVT::Untyped);
    } else {
      SDValue Ops[] = {
          CurDAG->getTargetConstant(Capstone::GPRPairRegClassID, DL, MVT::i32), Lo,
          CurDAG->getTargetConstant(Capstone::sub_gpr_even, DL, MVT::i32), Hi,
          CurDAG->getTargetConstant(Capstone::sub_gpr_odd, DL, MVT::i32)};

      RegPair = SDValue(CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, DL,
                                               MVT::Untyped, Ops),
                        0);
    }

    MachineSDNode *New = CurDAG->getMachineNode(Capstone::SD_RV32, DL, MVT::Other,
                                                {RegPair, Base, Offset, Chain});
    CurDAG->setNodeMemRefs(New, {cast<MemSDNode>(Node)->getMemOperand()});
    ReplaceUses(SDValue(Node, 0), SDValue(New, 0));
    CurDAG->RemoveDeadNode(Node);
    return;
  }
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IntNo = Node->getConstantOperandVal(0);
    switch (IntNo) {
      // By default we do not custom select any intrinsic.
    default:
      break;
    case Intrinsic::capstone_vmsgeu:
    case Intrinsic::capstone_vmsge: {
      SDValue Src1 = Node->getOperand(1);
      SDValue Src2 = Node->getOperand(2);
      bool IsUnsigned = IntNo == Intrinsic::capstone_vmsgeu;
      bool IsCmpConstant = false;
      bool IsCmpMinimum = false;
      // Only custom select scalar second operand.
      if (Src2.getValueType() != XLenVT)
        break;
      // Small constants are handled with patterns.
      int64_t CVal = 0;
      MVT Src1VT = Src1.getSimpleValueType();
      if (auto *C = dyn_cast<ConstantSDNode>(Src2)) {
        IsCmpConstant = true;
        CVal = C->getSExtValue();
        if (CVal >= -15 && CVal <= 16) {
          if (!IsUnsigned || CVal != 0)
            break;
          IsCmpMinimum = true;
        } else if (!IsUnsigned && CVal == APInt::getSignedMinValue(
                                              Src1VT.getScalarSizeInBits())
                                              .getSExtValue()) {
          IsCmpMinimum = true;
        }
      }
      unsigned VMSLTOpcode, VMNANDOpcode, VMSetOpcode, VMSGTOpcode;
      switch (CapstoneTargetLowering::getLMUL(Src1VT)) {
      default:
        llvm_unreachable("Unexpected LMUL!");
#define CASE_VMSLT_OPCODES(lmulenum, suffix)                                   \
  case CapstoneVType::lmulenum:                                                   \
    VMSLTOpcode = IsUnsigned ? Capstone::PseudoVMSLTU_VX_##suffix                 \
                             : Capstone::PseudoVMSLT_VX_##suffix;                 \
    VMSGTOpcode = IsUnsigned ? Capstone::PseudoVMSGTU_VX_##suffix                 \
                             : Capstone::PseudoVMSGT_VX_##suffix;                 \
    break;
        CASE_VMSLT_OPCODES(LMUL_F8, MF8)
        CASE_VMSLT_OPCODES(LMUL_F4, MF4)
        CASE_VMSLT_OPCODES(LMUL_F2, MF2)
        CASE_VMSLT_OPCODES(LMUL_1, M1)
        CASE_VMSLT_OPCODES(LMUL_2, M2)
        CASE_VMSLT_OPCODES(LMUL_4, M4)
        CASE_VMSLT_OPCODES(LMUL_8, M8)
#undef CASE_VMSLT_OPCODES
      }
      // Mask operations use the LMUL from the mask type.
      switch (CapstoneTargetLowering::getLMUL(VT)) {
      default:
        llvm_unreachable("Unexpected LMUL!");
#define CASE_VMNAND_VMSET_OPCODES(lmulenum, suffix)                            \
  case CapstoneVType::lmulenum:                                                   \
    VMNANDOpcode = Capstone::PseudoVMNAND_MM_##suffix;                            \
    VMSetOpcode = Capstone::PseudoVMSET_M_##suffix;                               \
    break;
        CASE_VMNAND_VMSET_OPCODES(LMUL_F8, B64)
        CASE_VMNAND_VMSET_OPCODES(LMUL_F4, B32)
        CASE_VMNAND_VMSET_OPCODES(LMUL_F2, B16)
        CASE_VMNAND_VMSET_OPCODES(LMUL_1, B8)
        CASE_VMNAND_VMSET_OPCODES(LMUL_2, B4)
        CASE_VMNAND_VMSET_OPCODES(LMUL_4, B2)
        CASE_VMNAND_VMSET_OPCODES(LMUL_8, B1)
#undef CASE_VMNAND_VMSET_OPCODES
      }
      SDValue SEW = CurDAG->getTargetConstant(
          Log2_32(Src1VT.getScalarSizeInBits()), DL, XLenVT);
      SDValue MaskSEW = CurDAG->getTargetConstant(0, DL, XLenVT);
      SDValue VL;
      selectVLOp(Node->getOperand(3), VL);

      // If vmsge(u) with minimum value, expand it to vmset.
      if (IsCmpMinimum) {
        ReplaceNode(Node,
                    CurDAG->getMachineNode(VMSetOpcode, DL, VT, VL, MaskSEW));
        return;
      }

      if (IsCmpConstant) {
        SDValue Imm =
            selectImm(CurDAG, SDLoc(Src2), XLenVT, CVal - 1, *Subtarget);

        ReplaceNode(Node, CurDAG->getMachineNode(VMSGTOpcode, DL, VT,
                                                 {Src1, Imm, VL, SEW}));
        return;
      }

      // Expand to
      // vmslt{u}.vx vd, va, x; vmnand.mm vd, vd, vd
      SDValue Cmp = SDValue(
          CurDAG->getMachineNode(VMSLTOpcode, DL, VT, {Src1, Src2, VL, SEW}),
          0);
      ReplaceNode(Node, CurDAG->getMachineNode(VMNANDOpcode, DL, VT,
                                               {Cmp, Cmp, VL, MaskSEW}));
      return;
    }
    case Intrinsic::capstone_vmsgeu_mask:
    case Intrinsic::capstone_vmsge_mask: {
      SDValue Src1 = Node->getOperand(2);
      SDValue Src2 = Node->getOperand(3);
      bool IsUnsigned = IntNo == Intrinsic::capstone_vmsgeu_mask;
      bool IsCmpConstant = false;
      bool IsCmpMinimum = false;
      // Only custom select scalar second operand.
      if (Src2.getValueType() != XLenVT)
        break;
      // Small constants are handled with patterns.
      MVT Src1VT = Src1.getSimpleValueType();
      int64_t CVal = 0;
      if (auto *C = dyn_cast<ConstantSDNode>(Src2)) {
        IsCmpConstant = true;
        CVal = C->getSExtValue();
        if (CVal >= -15 && CVal <= 16) {
          if (!IsUnsigned || CVal != 0)
            break;
          IsCmpMinimum = true;
        } else if (!IsUnsigned && CVal == APInt::getSignedMinValue(
                                              Src1VT.getScalarSizeInBits())
                                              .getSExtValue()) {
          IsCmpMinimum = true;
        }
      }
      unsigned VMSLTOpcode, VMSLTMaskOpcode, VMXOROpcode, VMANDNOpcode,
          VMOROpcode, VMSGTMaskOpcode;
      switch (CapstoneTargetLowering::getLMUL(Src1VT)) {
      default:
        llvm_unreachable("Unexpected LMUL!");
#define CASE_VMSLT_OPCODES(lmulenum, suffix)                                   \
  case CapstoneVType::lmulenum:                                                   \
    VMSLTOpcode = IsUnsigned ? Capstone::PseudoVMSLTU_VX_##suffix                 \
                             : Capstone::PseudoVMSLT_VX_##suffix;                 \
    VMSLTMaskOpcode = IsUnsigned ? Capstone::PseudoVMSLTU_VX_##suffix##_MASK      \
                                 : Capstone::PseudoVMSLT_VX_##suffix##_MASK;      \
    VMSGTMaskOpcode = IsUnsigned ? Capstone::PseudoVMSGTU_VX_##suffix##_MASK      \
                                 : Capstone::PseudoVMSGT_VX_##suffix##_MASK;      \
    break;
        CASE_VMSLT_OPCODES(LMUL_F8, MF8)
        CASE_VMSLT_OPCODES(LMUL_F4, MF4)
        CASE_VMSLT_OPCODES(LMUL_F2, MF2)
        CASE_VMSLT_OPCODES(LMUL_1, M1)
        CASE_VMSLT_OPCODES(LMUL_2, M2)
        CASE_VMSLT_OPCODES(LMUL_4, M4)
        CASE_VMSLT_OPCODES(LMUL_8, M8)
#undef CASE_VMSLT_OPCODES
      }
      // Mask operations use the LMUL from the mask type.
      switch (CapstoneTargetLowering::getLMUL(VT)) {
      default:
        llvm_unreachable("Unexpected LMUL!");
#define CASE_VMXOR_VMANDN_VMOR_OPCODES(lmulenum, suffix)                       \
  case CapstoneVType::lmulenum:                                                   \
    VMXOROpcode = Capstone::PseudoVMXOR_MM_##suffix;                              \
    VMANDNOpcode = Capstone::PseudoVMANDN_MM_##suffix;                            \
    VMOROpcode = Capstone::PseudoVMOR_MM_##suffix;                                \
    break;
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_F8, B64)
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_F4, B32)
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_F2, B16)
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_1, B8)
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_2, B4)
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_4, B2)
        CASE_VMXOR_VMANDN_VMOR_OPCODES(LMUL_8, B1)
#undef CASE_VMXOR_VMANDN_VMOR_OPCODES
      }
      SDValue SEW = CurDAG->getTargetConstant(
          Log2_32(Src1VT.getScalarSizeInBits()), DL, XLenVT);
      SDValue MaskSEW = CurDAG->getTargetConstant(0, DL, XLenVT);
      SDValue VL;
      selectVLOp(Node->getOperand(5), VL);
      SDValue MaskedOff = Node->getOperand(1);
      SDValue Mask = Node->getOperand(4);

      // If vmsge(u) with minimum value, expand it to vmor mask, maskedoff.
      if (IsCmpMinimum) {
        // We don't need vmor if the MaskedOff and the Mask are the same
        // value.
        if (Mask == MaskedOff) {
          ReplaceUses(Node, Mask.getNode());
          return;
        }
        ReplaceNode(Node,
                    CurDAG->getMachineNode(VMOROpcode, DL, VT,
                                           {Mask, MaskedOff, VL, MaskSEW}));
        return;
      }

      // If the MaskedOff value and the Mask are the same value use
      // vmslt{u}.vx vt, va, x;  vmandn.mm vd, vd, vt
      // This avoids needing to copy v0 to vd before starting the next sequence.
      if (Mask == MaskedOff) {
        SDValue Cmp = SDValue(
            CurDAG->getMachineNode(VMSLTOpcode, DL, VT, {Src1, Src2, VL, SEW}),
            0);
        ReplaceNode(Node, CurDAG->getMachineNode(VMANDNOpcode, DL, VT,
                                                 {Mask, Cmp, VL, MaskSEW}));
        return;
      }

      SDValue PolicyOp =
          CurDAG->getTargetConstant(CapstoneVType::TAIL_AGNOSTIC, DL, XLenVT);

      if (IsCmpConstant) {
        SDValue Imm =
            selectImm(CurDAG, SDLoc(Src2), XLenVT, CVal - 1, *Subtarget);

        ReplaceNode(Node, CurDAG->getMachineNode(
                              VMSGTMaskOpcode, DL, VT,
                              {MaskedOff, Src1, Imm, Mask, VL, SEW, PolicyOp}));
        return;
      }

      // Otherwise use
      // vmslt{u}.vx vd, va, x, v0.t; vmxor.mm vd, vd, v0
      // The result is mask undisturbed.
      // We use the same instructions to emulate mask agnostic behavior, because
      // the agnostic result can be either undisturbed or all 1.
      SDValue Cmp = SDValue(CurDAG->getMachineNode(VMSLTMaskOpcode, DL, VT,
                                                   {MaskedOff, Src1, Src2, Mask,
                                                    VL, SEW, PolicyOp}),
                            0);
      // vmxor.mm vd, vd, v0 is used to update active value.
      ReplaceNode(Node, CurDAG->getMachineNode(VMXOROpcode, DL, VT,
                                               {Cmp, Mask, VL, MaskSEW}));
      return;
    }
    case Intrinsic::capstone_vsetvli:
    case Intrinsic::capstone_vsetvlimax:
      return selectVSETVLI(Node);
    }
    break;
  }
  case ISD::INTRINSIC_W_CHAIN: {
    unsigned IntNo = Node->getConstantOperandVal(1);
    switch (IntNo) {
      // By default we do not custom select any intrinsic.
    default:
      break;
    case Intrinsic::capstone_vlseg2:
    case Intrinsic::capstone_vlseg3:
    case Intrinsic::capstone_vlseg4:
    case Intrinsic::capstone_vlseg5:
    case Intrinsic::capstone_vlseg6:
    case Intrinsic::capstone_vlseg7:
    case Intrinsic::capstone_vlseg8: {
      selectVLSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ false,
                  /*IsStrided*/ false);
      return;
    }
    case Intrinsic::capstone_vlseg2_mask:
    case Intrinsic::capstone_vlseg3_mask:
    case Intrinsic::capstone_vlseg4_mask:
    case Intrinsic::capstone_vlseg5_mask:
    case Intrinsic::capstone_vlseg6_mask:
    case Intrinsic::capstone_vlseg7_mask:
    case Intrinsic::capstone_vlseg8_mask: {
      selectVLSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ true,
                  /*IsStrided*/ false);
      return;
    }
    case Intrinsic::capstone_vlsseg2:
    case Intrinsic::capstone_vlsseg3:
    case Intrinsic::capstone_vlsseg4:
    case Intrinsic::capstone_vlsseg5:
    case Intrinsic::capstone_vlsseg6:
    case Intrinsic::capstone_vlsseg7:
    case Intrinsic::capstone_vlsseg8: {
      selectVLSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ false,
                  /*IsStrided*/ true);
      return;
    }
    case Intrinsic::capstone_vlsseg2_mask:
    case Intrinsic::capstone_vlsseg3_mask:
    case Intrinsic::capstone_vlsseg4_mask:
    case Intrinsic::capstone_vlsseg5_mask:
    case Intrinsic::capstone_vlsseg6_mask:
    case Intrinsic::capstone_vlsseg7_mask:
    case Intrinsic::capstone_vlsseg8_mask: {
      selectVLSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ true,
                  /*IsStrided*/ true);
      return;
    }
    case Intrinsic::capstone_vloxseg2:
    case Intrinsic::capstone_vloxseg3:
    case Intrinsic::capstone_vloxseg4:
    case Intrinsic::capstone_vloxseg5:
    case Intrinsic::capstone_vloxseg6:
    case Intrinsic::capstone_vloxseg7:
    case Intrinsic::capstone_vloxseg8:
      selectVLXSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ false,
                   /*IsOrdered*/ true);
      return;
    case Intrinsic::capstone_vluxseg2:
    case Intrinsic::capstone_vluxseg3:
    case Intrinsic::capstone_vluxseg4:
    case Intrinsic::capstone_vluxseg5:
    case Intrinsic::capstone_vluxseg6:
    case Intrinsic::capstone_vluxseg7:
    case Intrinsic::capstone_vluxseg8:
      selectVLXSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ false,
                   /*IsOrdered*/ false);
      return;
    case Intrinsic::capstone_vloxseg2_mask:
    case Intrinsic::capstone_vloxseg3_mask:
    case Intrinsic::capstone_vloxseg4_mask:
    case Intrinsic::capstone_vloxseg5_mask:
    case Intrinsic::capstone_vloxseg6_mask:
    case Intrinsic::capstone_vloxseg7_mask:
    case Intrinsic::capstone_vloxseg8_mask:
      selectVLXSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ true,
                   /*IsOrdered*/ true);
      return;
    case Intrinsic::capstone_vluxseg2_mask:
    case Intrinsic::capstone_vluxseg3_mask:
    case Intrinsic::capstone_vluxseg4_mask:
    case Intrinsic::capstone_vluxseg5_mask:
    case Intrinsic::capstone_vluxseg6_mask:
    case Intrinsic::capstone_vluxseg7_mask:
    case Intrinsic::capstone_vluxseg8_mask:
      selectVLXSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ true,
                   /*IsOrdered*/ false);
      return;
    case Intrinsic::capstone_vlseg8ff:
    case Intrinsic::capstone_vlseg7ff:
    case Intrinsic::capstone_vlseg6ff:
    case Intrinsic::capstone_vlseg5ff:
    case Intrinsic::capstone_vlseg4ff:
    case Intrinsic::capstone_vlseg3ff:
    case Intrinsic::capstone_vlseg2ff: {
      selectVLSEGFF(Node, getSegInstNF(IntNo), /*IsMasked*/ false);
      return;
    }
    case Intrinsic::capstone_vlseg8ff_mask:
    case Intrinsic::capstone_vlseg7ff_mask:
    case Intrinsic::capstone_vlseg6ff_mask:
    case Intrinsic::capstone_vlseg5ff_mask:
    case Intrinsic::capstone_vlseg4ff_mask:
    case Intrinsic::capstone_vlseg3ff_mask:
    case Intrinsic::capstone_vlseg2ff_mask: {
      selectVLSEGFF(Node, getSegInstNF(IntNo), /*IsMasked*/ true);
      return;
    }
    case Intrinsic::capstone_vloxei:
    case Intrinsic::capstone_vloxei_mask:
    case Intrinsic::capstone_vluxei:
    case Intrinsic::capstone_vluxei_mask: {
      bool IsMasked = IntNo == Intrinsic::capstone_vloxei_mask ||
                      IntNo == Intrinsic::capstone_vluxei_mask;
      bool IsOrdered = IntNo == Intrinsic::capstone_vloxei ||
                       IntNo == Intrinsic::capstone_vloxei_mask;

      MVT VT = Node->getSimpleValueType(0);
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      unsigned CurOp = 2;
      SmallVector<SDValue, 8> Operands;
      Operands.push_back(Node->getOperand(CurOp++));

      MVT IndexVT;
      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                                 /*IsStridedOrIndexed*/ true, Operands,
                                 /*IsLoad=*/true, &IndexVT);

      assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
             "Element count mismatch");

      CapstoneVType::VLMUL LMUL = CapstoneTargetLowering::getLMUL(VT);
      CapstoneVType::VLMUL IndexLMUL = CapstoneTargetLowering::getLMUL(IndexVT);
      unsigned IndexLog2EEW = Log2_32(IndexVT.getScalarSizeInBits());
      if (IndexLog2EEW == 6 && !Subtarget->is64Bit()) {
        report_fatal_error("The V extension does not support EEW=64 for index "
                           "values when XLEN=32");
      }
      const Capstone::VLX_VSXPseudo *P = Capstone::getVLXPseudo(
          IsMasked, IsOrdered, IndexLog2EEW, static_cast<unsigned>(LMUL),
          static_cast<unsigned>(IndexLMUL));
      MachineSDNode *Load =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);

      CurDAG->setNodeMemRefs(Load, {cast<MemSDNode>(Node)->getMemOperand()});

      ReplaceNode(Node, Load);
      return;
    }
    case Intrinsic::capstone_vlm:
    case Intrinsic::capstone_vle:
    case Intrinsic::capstone_vle_mask:
    case Intrinsic::capstone_vlse:
    case Intrinsic::capstone_vlse_mask: {
      bool IsMasked = IntNo == Intrinsic::capstone_vle_mask ||
                      IntNo == Intrinsic::capstone_vlse_mask;
      bool IsStrided =
          IntNo == Intrinsic::capstone_vlse || IntNo == Intrinsic::capstone_vlse_mask;

      MVT VT = Node->getSimpleValueType(0);
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      // The capstone_vlm intrinsic are always tail agnostic and no passthru
      // operand at the IR level.  In pseudos, they have both policy and
      // passthru operand. The passthru operand is needed to track the
      // "tail undefined" state, and the policy is there just for
      // for consistency - it will always be "don't care" for the
      // unmasked form.
      bool HasPassthruOperand = IntNo != Intrinsic::capstone_vlm;
      unsigned CurOp = 2;
      SmallVector<SDValue, 8> Operands;
      if (HasPassthruOperand)
        Operands.push_back(Node->getOperand(CurOp++));
      else {
        // We eagerly lower to implicit_def (instead of undef), as we
        // otherwise fail to select nodes such as: nxv1i1 = undef
        SDNode *Passthru =
          CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF, DL, VT);
        Operands.push_back(SDValue(Passthru, 0));
      }
      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked, IsStrided,
                                 Operands, /*IsLoad=*/true);

      CapstoneVType::VLMUL LMUL = CapstoneTargetLowering::getLMUL(VT);
      const Capstone::VLEPseudo *P =
          Capstone::getVLEPseudo(IsMasked, IsStrided, /*FF*/ false, Log2SEW,
                              static_cast<unsigned>(LMUL));
      MachineSDNode *Load =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);

      CurDAG->setNodeMemRefs(Load, {cast<MemSDNode>(Node)->getMemOperand()});

      ReplaceNode(Node, Load);
      return;
    }
    case Intrinsic::capstone_vleff:
    case Intrinsic::capstone_vleff_mask: {
      bool IsMasked = IntNo == Intrinsic::capstone_vleff_mask;

      MVT VT = Node->getSimpleValueType(0);
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      unsigned CurOp = 2;
      SmallVector<SDValue, 7> Operands;
      Operands.push_back(Node->getOperand(CurOp++));
      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                                 /*IsStridedOrIndexed*/ false, Operands,
                                 /*IsLoad=*/true);

      CapstoneVType::VLMUL LMUL = CapstoneTargetLowering::getLMUL(VT);
      const Capstone::VLEPseudo *P =
          Capstone::getVLEPseudo(IsMasked, /*Strided*/ false, /*FF*/ true,
                              Log2SEW, static_cast<unsigned>(LMUL));
      MachineSDNode *Load = CurDAG->getMachineNode(
          P->Pseudo, DL, Node->getVTList(), Operands);
      CurDAG->setNodeMemRefs(Load, {cast<MemSDNode>(Node)->getMemOperand()});

      ReplaceNode(Node, Load);
      return;
    }
    case Intrinsic::capstone_nds_vln:
    case Intrinsic::capstone_nds_vln_mask:
    case Intrinsic::capstone_nds_vlnu:
    case Intrinsic::capstone_nds_vlnu_mask: {
      bool IsMasked = IntNo == Intrinsic::capstone_nds_vln_mask ||
                      IntNo == Intrinsic::capstone_nds_vlnu_mask;
      bool IsUnsigned = IntNo == Intrinsic::capstone_nds_vlnu ||
                        IntNo == Intrinsic::capstone_nds_vlnu_mask;

      MVT VT = Node->getSimpleValueType(0);
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());
      unsigned CurOp = 2;
      SmallVector<SDValue, 8> Operands;

      Operands.push_back(Node->getOperand(CurOp++));
      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                                 /*IsStridedOrIndexed=*/false, Operands,
                                 /*IsLoad=*/true);

      CapstoneVType::VLMUL LMUL = CapstoneTargetLowering::getLMUL(VT);
      const Capstone::NDSVLNPseudo *P = Capstone::getNDSVLNPseudo(
          IsMasked, IsUnsigned, Log2SEW, static_cast<unsigned>(LMUL));
      MachineSDNode *Load =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);

      if (auto *MemOp = dyn_cast<MemSDNode>(Node))
        CurDAG->setNodeMemRefs(Load, {MemOp->getMemOperand()});

      ReplaceNode(Node, Load);
      return;
    }
    }
    break;
  }
  case ISD::INTRINSIC_VOID: {
    unsigned IntNo = Node->getConstantOperandVal(1);
    switch (IntNo) {
    case Intrinsic::capstone_vsseg2:
    case Intrinsic::capstone_vsseg3:
    case Intrinsic::capstone_vsseg4:
    case Intrinsic::capstone_vsseg5:
    case Intrinsic::capstone_vsseg6:
    case Intrinsic::capstone_vsseg7:
    case Intrinsic::capstone_vsseg8: {
      selectVSSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ false,
                  /*IsStrided*/ false);
      return;
    }
    case Intrinsic::capstone_vsseg2_mask:
    case Intrinsic::capstone_vsseg3_mask:
    case Intrinsic::capstone_vsseg4_mask:
    case Intrinsic::capstone_vsseg5_mask:
    case Intrinsic::capstone_vsseg6_mask:
    case Intrinsic::capstone_vsseg7_mask:
    case Intrinsic::capstone_vsseg8_mask: {
      selectVSSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ true,
                  /*IsStrided*/ false);
      return;
    }
    case Intrinsic::capstone_vssseg2:
    case Intrinsic::capstone_vssseg3:
    case Intrinsic::capstone_vssseg4:
    case Intrinsic::capstone_vssseg5:
    case Intrinsic::capstone_vssseg6:
    case Intrinsic::capstone_vssseg7:
    case Intrinsic::capstone_vssseg8: {
      selectVSSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ false,
                  /*IsStrided*/ true);
      return;
    }
    case Intrinsic::capstone_vssseg2_mask:
    case Intrinsic::capstone_vssseg3_mask:
    case Intrinsic::capstone_vssseg4_mask:
    case Intrinsic::capstone_vssseg5_mask:
    case Intrinsic::capstone_vssseg6_mask:
    case Intrinsic::capstone_vssseg7_mask:
    case Intrinsic::capstone_vssseg8_mask: {
      selectVSSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ true,
                  /*IsStrided*/ true);
      return;
    }
    case Intrinsic::capstone_vsoxseg2:
    case Intrinsic::capstone_vsoxseg3:
    case Intrinsic::capstone_vsoxseg4:
    case Intrinsic::capstone_vsoxseg5:
    case Intrinsic::capstone_vsoxseg6:
    case Intrinsic::capstone_vsoxseg7:
    case Intrinsic::capstone_vsoxseg8:
      selectVSXSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ false,
                   /*IsOrdered*/ true);
      return;
    case Intrinsic::capstone_vsuxseg2:
    case Intrinsic::capstone_vsuxseg3:
    case Intrinsic::capstone_vsuxseg4:
    case Intrinsic::capstone_vsuxseg5:
    case Intrinsic::capstone_vsuxseg6:
    case Intrinsic::capstone_vsuxseg7:
    case Intrinsic::capstone_vsuxseg8:
      selectVSXSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ false,
                   /*IsOrdered*/ false);
      return;
    case Intrinsic::capstone_vsoxseg2_mask:
    case Intrinsic::capstone_vsoxseg3_mask:
    case Intrinsic::capstone_vsoxseg4_mask:
    case Intrinsic::capstone_vsoxseg5_mask:
    case Intrinsic::capstone_vsoxseg6_mask:
    case Intrinsic::capstone_vsoxseg7_mask:
    case Intrinsic::capstone_vsoxseg8_mask:
      selectVSXSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ true,
                   /*IsOrdered*/ true);
      return;
    case Intrinsic::capstone_vsuxseg2_mask:
    case Intrinsic::capstone_vsuxseg3_mask:
    case Intrinsic::capstone_vsuxseg4_mask:
    case Intrinsic::capstone_vsuxseg5_mask:
    case Intrinsic::capstone_vsuxseg6_mask:
    case Intrinsic::capstone_vsuxseg7_mask:
    case Intrinsic::capstone_vsuxseg8_mask:
      selectVSXSEG(Node, getSegInstNF(IntNo), /*IsMasked*/ true,
                   /*IsOrdered*/ false);
      return;
    case Intrinsic::capstone_vsoxei:
    case Intrinsic::capstone_vsoxei_mask:
    case Intrinsic::capstone_vsuxei:
    case Intrinsic::capstone_vsuxei_mask: {
      bool IsMasked = IntNo == Intrinsic::capstone_vsoxei_mask ||
                      IntNo == Intrinsic::capstone_vsuxei_mask;
      bool IsOrdered = IntNo == Intrinsic::capstone_vsoxei ||
                       IntNo == Intrinsic::capstone_vsoxei_mask;

      MVT VT = Node->getOperand(2)->getSimpleValueType(0);
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      unsigned CurOp = 2;
      SmallVector<SDValue, 8> Operands;
      Operands.push_back(Node->getOperand(CurOp++)); // Store value.

      MVT IndexVT;
      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked,
                                 /*IsStridedOrIndexed*/ true, Operands,
                                 /*IsLoad=*/false, &IndexVT);

      assert(VT.getVectorElementCount() == IndexVT.getVectorElementCount() &&
             "Element count mismatch");

      CapstoneVType::VLMUL LMUL = CapstoneTargetLowering::getLMUL(VT);
      CapstoneVType::VLMUL IndexLMUL = CapstoneTargetLowering::getLMUL(IndexVT);
      unsigned IndexLog2EEW = Log2_32(IndexVT.getScalarSizeInBits());
      if (IndexLog2EEW == 6 && !Subtarget->is64Bit()) {
        report_fatal_error("The V extension does not support EEW=64 for index "
                           "values when XLEN=32");
      }
      const Capstone::VLX_VSXPseudo *P = Capstone::getVSXPseudo(
          IsMasked, IsOrdered, IndexLog2EEW,
          static_cast<unsigned>(LMUL), static_cast<unsigned>(IndexLMUL));
      MachineSDNode *Store =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);

      CurDAG->setNodeMemRefs(Store, {cast<MemSDNode>(Node)->getMemOperand()});

      ReplaceNode(Node, Store);
      return;
    }
    case Intrinsic::capstone_vsm:
    case Intrinsic::capstone_vse:
    case Intrinsic::capstone_vse_mask:
    case Intrinsic::capstone_vsse:
    case Intrinsic::capstone_vsse_mask: {
      bool IsMasked = IntNo == Intrinsic::capstone_vse_mask ||
                      IntNo == Intrinsic::capstone_vsse_mask;
      bool IsStrided =
          IntNo == Intrinsic::capstone_vsse || IntNo == Intrinsic::capstone_vsse_mask;

      MVT VT = Node->getOperand(2)->getSimpleValueType(0);
      unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());

      unsigned CurOp = 2;
      SmallVector<SDValue, 8> Operands;
      Operands.push_back(Node->getOperand(CurOp++)); // Store value.

      addVectorLoadStoreOperands(Node, Log2SEW, DL, CurOp, IsMasked, IsStrided,
                                 Operands);

      CapstoneVType::VLMUL LMUL = CapstoneTargetLowering::getLMUL(VT);
      const Capstone::VSEPseudo *P = Capstone::getVSEPseudo(
          IsMasked, IsStrided, Log2SEW, static_cast<unsigned>(LMUL));
      MachineSDNode *Store =
          CurDAG->getMachineNode(P->Pseudo, DL, Node->getVTList(), Operands);
      CurDAG->setNodeMemRefs(Store, {cast<MemSDNode>(Node)->getMemOperand()});

      ReplaceNode(Node, Store);
      return;
    }
    case Intrinsic::capstone_sf_vc_x_se:
    case Intrinsic::capstone_sf_vc_i_se:
      selectSF_VC_X_SE(Node);
      return;
    }
    break;
  }
  case ISD::BITCAST: {
    MVT SrcVT = Node->getOperand(0).getSimpleValueType();
    // Just drop bitcasts between vectors if both are fixed or both are
    // scalable.
    if ((VT.isScalableVector() && SrcVT.isScalableVector()) ||
        (VT.isFixedLengthVector() && SrcVT.isFixedLengthVector())) {
      ReplaceUses(SDValue(Node, 0), Node->getOperand(0));
      CurDAG->RemoveDeadNode(Node);
      return;
    }
    break;
  }
  case ISD::INSERT_SUBVECTOR:
  case CapstoneISD::TUPLE_INSERT: {
    SDValue V = Node->getOperand(0);
    SDValue SubV = Node->getOperand(1);
    SDLoc DL(SubV);
    auto Idx = Node->getConstantOperandVal(2);
    MVT SubVecVT = SubV.getSimpleValueType();

    const CapstoneTargetLowering &TLI = *Subtarget->getTargetLowering();
    MVT SubVecContainerVT = SubVecVT;
    // Establish the correct scalable-vector types for any fixed-length type.
    if (SubVecVT.isFixedLengthVector()) {
      SubVecContainerVT = TLI.getContainerForFixedLengthVector(SubVecVT);
      TypeSize VecRegSize = TypeSize::getScalable(Capstone::RVVBitsPerBlock);
      [[maybe_unused]] bool ExactlyVecRegSized =
          Subtarget->expandVScale(SubVecVT.getSizeInBits())
              .isKnownMultipleOf(Subtarget->expandVScale(VecRegSize));
      assert(isPowerOf2_64(Subtarget->expandVScale(SubVecVT.getSizeInBits())
                               .getKnownMinValue()));
      assert(Idx == 0 && (ExactlyVecRegSized || V.isUndef()));
    }
    MVT ContainerVT = VT;
    if (VT.isFixedLengthVector())
      ContainerVT = TLI.getContainerForFixedLengthVector(VT);

    const auto *TRI = Subtarget->getRegisterInfo();
    unsigned SubRegIdx;
    std::tie(SubRegIdx, Idx) =
        CapstoneTargetLowering::decomposeSubvectorInsertExtractToSubRegs(
            ContainerVT, SubVecContainerVT, Idx, TRI);

    // If the Idx hasn't been completely eliminated then this is a subvector
    // insert which doesn't naturally align to a vector register. These must
    // be handled using instructions to manipulate the vector registers.
    if (Idx != 0)
      break;

    CapstoneVType::VLMUL SubVecLMUL =
        CapstoneTargetLowering::getLMUL(SubVecContainerVT);
    [[maybe_unused]] bool IsSubVecPartReg =
        SubVecLMUL == CapstoneVType::VLMUL::LMUL_F2 ||
        SubVecLMUL == CapstoneVType::VLMUL::LMUL_F4 ||
        SubVecLMUL == CapstoneVType::VLMUL::LMUL_F8;
    assert((V.getValueType().isCapstoneVectorTuple() || !IsSubVecPartReg ||
            V.isUndef()) &&
           "Expecting lowering to have created legal INSERT_SUBVECTORs when "
           "the subvector is smaller than a full-sized register");

    // If we haven't set a SubRegIdx, then we must be going between
    // equally-sized LMUL groups (e.g. VR -> VR). This can be done as a copy.
    if (SubRegIdx == Capstone::NoSubRegister) {
      unsigned InRegClassID =
          CapstoneTargetLowering::getRegClassIDForVecVT(ContainerVT);
      assert(CapstoneTargetLowering::getRegClassIDForVecVT(SubVecContainerVT) ==
                 InRegClassID &&
             "Unexpected subvector extraction");
      SDValue RC = CurDAG->getTargetConstant(InRegClassID, DL, XLenVT);
      SDNode *NewNode = CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS,
                                               DL, VT, SubV, RC);
      ReplaceNode(Node, NewNode);
      return;
    }

    SDValue Insert = CurDAG->getTargetInsertSubreg(SubRegIdx, DL, VT, V, SubV);
    ReplaceNode(Node, Insert.getNode());
    return;
  }
  case ISD::EXTRACT_SUBVECTOR:
  case CapstoneISD::TUPLE_EXTRACT: {
    SDValue V = Node->getOperand(0);
    auto Idx = Node->getConstantOperandVal(1);
    MVT InVT = V.getSimpleValueType();
    SDLoc DL(V);

    const CapstoneTargetLowering &TLI = *Subtarget->getTargetLowering();
    MVT SubVecContainerVT = VT;
    // Establish the correct scalable-vector types for any fixed-length type.
    if (VT.isFixedLengthVector()) {
      assert(Idx == 0);
      SubVecContainerVT = TLI.getContainerForFixedLengthVector(VT);
    }
    if (InVT.isFixedLengthVector())
      InVT = TLI.getContainerForFixedLengthVector(InVT);

    const auto *TRI = Subtarget->getRegisterInfo();
    unsigned SubRegIdx;
    std::tie(SubRegIdx, Idx) =
        CapstoneTargetLowering::decomposeSubvectorInsertExtractToSubRegs(
            InVT, SubVecContainerVT, Idx, TRI);

    // If the Idx hasn't been completely eliminated then this is a subvector
    // extract which doesn't naturally align to a vector register. These must
    // be handled using instructions to manipulate the vector registers.
    if (Idx != 0)
      break;

    // If we haven't set a SubRegIdx, then we must be going between
    // equally-sized LMUL types (e.g. VR -> VR). This can be done as a copy.
    if (SubRegIdx == Capstone::NoSubRegister) {
      unsigned InRegClassID = CapstoneTargetLowering::getRegClassIDForVecVT(InVT);
      assert(CapstoneTargetLowering::getRegClassIDForVecVT(SubVecContainerVT) ==
                 InRegClassID &&
             "Unexpected subvector extraction");
      SDValue RC = CurDAG->getTargetConstant(InRegClassID, DL, XLenVT);
      SDNode *NewNode =
          CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS, DL, VT, V, RC);
      ReplaceNode(Node, NewNode);
      return;
    }

    SDValue Extract = CurDAG->getTargetExtractSubreg(SubRegIdx, DL, VT, V);
    ReplaceNode(Node, Extract.getNode());
    return;
  }
  case CapstoneISD::VMV_S_X_VL:
  case CapstoneISD::VFMV_S_F_VL:
  case CapstoneISD::VMV_V_X_VL:
  case CapstoneISD::VFMV_V_F_VL: {
    // Try to match splat of a scalar load to a strided load with stride of x0.
    bool IsScalarMove = Node->getOpcode() == CapstoneISD::VMV_S_X_VL ||
                        Node->getOpcode() == CapstoneISD::VFMV_S_F_VL;
    if (!Node->getOperand(0).isUndef())
      break;
    SDValue Src = Node->getOperand(1);
    auto *Ld = dyn_cast<LoadSDNode>(Src);
    // Can't fold load update node because the second
    // output is used so that load update node can't be removed.
    if (!Ld || Ld->isIndexed())
      break;
    EVT MemVT = Ld->getMemoryVT();
    // The memory VT should be the same size as the element type.
    if (MemVT.getStoreSize() != VT.getVectorElementType().getStoreSize())
      break;
    if (!IsProfitableToFold(Src, Node, Node) ||
        !IsLegalToFold(Src, Node, Node, TM.getOptLevel()))
      break;

    SDValue VL;
    if (IsScalarMove) {
      // We could deal with more VL if we update the VSETVLI insert pass to
      // avoid introducing more VSETVLI.
      if (!isOneConstant(Node->getOperand(2)))
        break;
      selectVLOp(Node->getOperand(2), VL);
    } else
      selectVLOp(Node->getOperand(2), VL);

    unsigned Log2SEW = Log2_32(VT.getScalarSizeInBits());
    SDValue SEW = CurDAG->getTargetConstant(Log2SEW, DL, XLenVT);

    // If VL=1, then we don't need to do a strided load and can just do a
    // regular load.
    bool IsStrided = !isOneConstant(VL);

    // Only do a strided load if we have optimized zero-stride vector load.
    if (IsStrided && !Subtarget->hasOptimizedZeroStrideLoad())
      break;

    SmallVector<SDValue> Operands = {
        SDValue(CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF, DL, VT), 0),
        Ld->getBasePtr()};
    if (IsStrided)
      Operands.push_back(CurDAG->getRegister(Capstone::X0, XLenVT));
    uint64_t Policy = CapstoneVType::MASK_AGNOSTIC | CapstoneVType::TAIL_AGNOSTIC;
    SDValue PolicyOp = CurDAG->getTargetConstant(Policy, DL, XLenVT);
    Operands.append({VL, SEW, PolicyOp, Ld->getChain()});

    CapstoneVType::VLMUL LMUL = CapstoneTargetLowering::getLMUL(VT);
    const Capstone::VLEPseudo *P = Capstone::getVLEPseudo(
        /*IsMasked*/ false, IsStrided, /*FF*/ false,
        Log2SEW, static_cast<unsigned>(LMUL));
    MachineSDNode *Load =
        CurDAG->getMachineNode(P->Pseudo, DL, {VT, MVT::Other}, Operands);
    // Update the chain.
    ReplaceUses(Src.getValue(1), SDValue(Load, 1));
    // Record the mem-refs
    CurDAG->setNodeMemRefs(Load, {Ld->getMemOperand()});
    // Replace the splat with the vlse.
    ReplaceNode(Node, Load);
    return;
  }
  case ISD::PREFETCH:
    unsigned Locality = Node->getConstantOperandVal(3);
    if (Locality > 2)
      break;

    auto *LoadStoreMem = cast<MemSDNode>(Node);
    MachineMemOperand *MMO = LoadStoreMem->getMemOperand();
    MMO->setFlags(MachineMemOperand::MONonTemporal);

    int NontemporalLevel = 0;
    switch (Locality) {
    case 0:
      NontemporalLevel = 3; // NTL.ALL
      break;
    case 1:
      NontemporalLevel = 1; // NTL.PALL
      break;
    case 2:
      NontemporalLevel = 0; // NTL.P1
      break;
    default:
      llvm_unreachable("unexpected locality value.");
    }

    if (NontemporalLevel & 0b1)
      MMO->setFlags(MONontemporalBit0);
    if (NontemporalLevel & 0b10)
      MMO->setFlags(MONontemporalBit1);
    break;
  }

  // Select the default instruction.
  SelectCode(Node);
}

bool CapstoneDAGToDAGISel::SelectInlineAsmMemoryOperand(
    const SDValue &Op, InlineAsm::ConstraintCode ConstraintID,
    std::vector<SDValue> &OutOps) {
  // Always produce a register and immediate operand, as expected by
  // CapstoneAsmPrinter::PrintAsmMemoryOperand.
  switch (ConstraintID) {
  case InlineAsm::ConstraintCode::o:
  case InlineAsm::ConstraintCode::m: {
    SDValue Op0, Op1;
    [[maybe_unused]] bool Found = SelectAddrRegImm(Op, Op0, Op1);
    assert(Found && "SelectAddrRegImm should always succeed");
    OutOps.push_back(Op0);
    OutOps.push_back(Op1);
    return false;
  }
  case InlineAsm::ConstraintCode::A:
    OutOps.push_back(Op);
    OutOps.push_back(
        CurDAG->getTargetConstant(0, SDLoc(Op), Subtarget->getXLenVT()));
    return false;
  default:
    report_fatal_error("Unexpected asm memory constraint " +
                       InlineAsm::getMemConstraintName(ConstraintID));
  }

  return true;
}

bool CapstoneDAGToDAGISel::SelectAddrFrameIndex(SDValue Addr, SDValue &Base,
                                             SDValue &Offset) {
  if (auto *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), Subtarget->getXLenVT());
    Offset = CurDAG->getTargetConstant(0, SDLoc(Addr), Subtarget->getXLenVT());
    return true;
  }

  return false;
}

// Fold constant addresses.
static bool selectConstantAddr(SelectionDAG *CurDAG, const SDLoc &DL,
                               const MVT VT, const CapstoneSubtarget *Subtarget,
                               SDValue Addr, SDValue &Base, SDValue &Offset,
                               bool IsPrefetch = false) {
  if (!isa<ConstantSDNode>(Addr))
    return false;

  int64_t CVal = cast<ConstantSDNode>(Addr)->getSExtValue();

  // If the constant is a simm12, we can fold the whole constant and use X0 as
  // the base. If the constant can be materialized with LUI+simm12, use LUI as
  // the base. We can't use generateInstSeq because it favors LUI+ADDIW.
  int64_t Lo12 = SignExtend64<12>(CVal);
  int64_t Hi = (uint64_t)CVal - (uint64_t)Lo12;
  if (!Subtarget->is64Bit() || isInt<32>(Hi)) {
    if (IsPrefetch && (Lo12 & 0b11111) != 0)
      return false;
    if (Hi) {
      int64_t Hi20 = (Hi >> 12) & 0xfffff;
      Base = SDValue(
          CurDAG->getMachineNode(Capstone::LUI, DL, VT,
                                 CurDAG->getTargetConstant(Hi20, DL, VT)),
          0);
    } else {
      Base = CurDAG->getRegister(Capstone::X0, VT);
    }
    Offset = CurDAG->getSignedTargetConstant(Lo12, DL, VT);
    return true;
  }

  // Ask how constant materialization would handle this constant.
  CapstoneMatInt::InstSeq Seq = CapstoneMatInt::generateInstSeq(CVal, *Subtarget);

  // If the last instruction would be an ADDI, we can fold its immediate and
  // emit the rest of the sequence as the base.
  if (Seq.back().getOpcode() != Capstone::ADDI)
    return false;
  Lo12 = Seq.back().getImm();
  if (IsPrefetch && (Lo12 & 0b11111) != 0)
    return false;

  // Drop the last instruction.
  Seq.pop_back();
  assert(!Seq.empty() && "Expected more instructions in sequence");

  Base = selectImmSeq(CurDAG, DL, VT, Seq);
  Offset = CurDAG->getSignedTargetConstant(Lo12, DL, VT);
  return true;
}

// Is this ADD instruction only used as the base pointer of scalar loads and
// stores?
static bool isWorthFoldingAdd(SDValue Add) {
  for (auto *User : Add->users()) {
    if (User->getOpcode() != ISD::LOAD && User->getOpcode() != ISD::STORE &&
        User->getOpcode() != CapstoneISD::LD_RV32 &&
        User->getOpcode() != CapstoneISD::SD_RV32 &&
        User->getOpcode() != ISD::ATOMIC_LOAD &&
        User->getOpcode() != ISD::ATOMIC_STORE)
      return false;
    EVT VT = cast<MemSDNode>(User)->getMemoryVT();
    if (!VT.isScalarInteger() && VT != MVT::f16 && VT != MVT::f32 &&
        VT != MVT::f64)
      return false;
    // Don't allow stores of the value. It must be used as the address.
    if (User->getOpcode() == ISD::STORE &&
        cast<StoreSDNode>(User)->getValue() == Add)
      return false;
    if (User->getOpcode() == ISD::ATOMIC_STORE &&
        cast<AtomicSDNode>(User)->getVal() == Add)
      return false;
    if (User->getOpcode() == CapstoneISD::SD_RV32 &&
        (User->getOperand(0) == Add || User->getOperand(1) == Add))
      return false;
    if (isStrongerThanMonotonic(cast<MemSDNode>(User)->getSuccessOrdering()))
      return false;
  }

  return true;
}

bool isRegImmLoadOrStore(SDNode *User, SDValue Add) {
  switch (User->getOpcode()) {
  default:
    return false;
  case ISD::LOAD:
  case CapstoneISD::LD_RV32:
  case ISD::ATOMIC_LOAD:
    break;
  case ISD::STORE:
    // Don't allow stores of Add. It must only be used as the address.
    if (cast<StoreSDNode>(User)->getValue() == Add)
      return false;
    break;
  case CapstoneISD::SD_RV32:
    // Don't allow stores of Add. It must only be used as the address.
    if (User->getOperand(0) == Add || User->getOperand(1) == Add)
      return false;
    break;
  case ISD::ATOMIC_STORE:
    // Don't allow stores of Add. It must only be used as the address.
    if (cast<AtomicSDNode>(User)->getVal() == Add)
      return false;
    break;
  }

  return true;
}

// To prevent SelectAddrRegImm from folding offsets that conflict with the
// fusion of PseudoMovAddr, check if the offset of every use of a given address
// is within the alignment.
bool CapstoneDAGToDAGISel::areOffsetsWithinAlignment(SDValue Addr,
                                                  Align Alignment) {
  assert(Addr->getOpcode() == CapstoneISD::ADD_LO);
  for (auto *User : Addr->users()) {
    // If the user is a load or store, then the offset is 0 which is always
    // within alignment.
    if (isRegImmLoadOrStore(User, Addr))
      continue;

    if (CurDAG->isBaseWithConstantOffset(SDValue(User, 0))) {
      int64_t CVal = cast<ConstantSDNode>(User->getOperand(1))->getSExtValue();
      if (!isInt<12>(CVal) || Alignment <= CVal)
        return false;

      // Make sure all uses are foldable load/stores.
      for (auto *AddUser : User->users())
        if (!isRegImmLoadOrStore(AddUser, SDValue(User, 0)))
          return false;

      continue;
    }

    return false;
  }

  return true;
}

bool CapstoneDAGToDAGISel::SelectAddrRegImm(SDValue Addr, SDValue &Base,
                                         SDValue &Offset) {
  if (SelectAddrFrameIndex(Addr, Base, Offset))
    return true;

  SDLoc DL(Addr);
  MVT VT = Addr.getSimpleValueType();

  if (Addr.getOpcode() == CapstoneISD::ADD_LO) {
    bool CanFold = true;
    // Unconditionally fold if operand 1 is not a global address (e.g.
    // externsymbol)
    if (auto *GA = dyn_cast<GlobalAddressSDNode>(Addr.getOperand(1))) {
      const DataLayout &DL = CurDAG->getDataLayout();
      Align Alignment = commonAlignment(
          GA->getGlobal()->getPointerAlignment(DL), GA->getOffset());
      if (!areOffsetsWithinAlignment(Addr, Alignment))
        CanFold = false;
    }
    if (CanFold) {
      Base = Addr.getOperand(0);
      Offset = Addr.getOperand(1);
      return true;
    }
  }

  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    int64_t CVal = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    if (isInt<12>(CVal)) {
      Base = Addr.getOperand(0);
      if (Base.getOpcode() == CapstoneISD::ADD_LO) {
        SDValue LoOperand = Base.getOperand(1);
        if (auto *GA = dyn_cast<GlobalAddressSDNode>(LoOperand)) {
          // If the Lo in (ADD_LO hi, lo) is a global variable's address
          // (its low part, really), then we can rely on the alignment of that
          // variable to provide a margin of safety before low part can overflow
          // the 12 bits of the load/store offset. Check if CVal falls within
          // that margin; if so (low part + CVal) can't overflow.
          const DataLayout &DL = CurDAG->getDataLayout();
          Align Alignment = commonAlignment(
              GA->getGlobal()->getPointerAlignment(DL), GA->getOffset());
          if ((CVal == 0 || Alignment > CVal) &&
              areOffsetsWithinAlignment(Base, Alignment)) {
            int64_t CombinedOffset = CVal + GA->getOffset();
            Base = Base.getOperand(0);
            Offset = CurDAG->getTargetGlobalAddress(
                GA->getGlobal(), SDLoc(LoOperand), LoOperand.getValueType(),
                CombinedOffset, GA->getTargetFlags());
            return true;
          }
        }
      }

      if (auto *FIN = dyn_cast<FrameIndexSDNode>(Base))
        Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), VT);
      Offset = CurDAG->getSignedTargetConstant(CVal, DL, VT);
      return true;
    }
  }

  // Handle ADD with large immediates.
  if (Addr.getOpcode() == ISD::ADD && isa<ConstantSDNode>(Addr.getOperand(1))) {
    int64_t CVal = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    assert(!isInt<12>(CVal) && "simm12 not already handled?");

    // Handle immediates in the range [-4096,-2049] or [2048, 4094]. We can use
    // an ADDI for part of the offset and fold the rest into the load/store.
    // This mirrors the AddiPair PatFrag in CapstoneInstrInfo.td.
    if (CVal >= -4096 && CVal <= 4094) {
      int64_t Adj = CVal < 0 ? -2048 : 2047;
      Base = SDValue(
          CurDAG->getMachineNode(Capstone::ADDI, DL, VT, Addr.getOperand(0),
                                 CurDAG->getSignedTargetConstant(Adj, DL, VT)),
          0);
      Offset = CurDAG->getSignedTargetConstant(CVal - Adj, DL, VT);
      return true;
    }

    // For larger immediates, we might be able to save one instruction from
    // constant materialization by folding the Lo12 bits of the immediate into
    // the address. We should only do this if the ADD is only used by loads and
    // stores that can fold the lo12 bits. Otherwise, the ADD will get iseled
    // separately with the full materialized immediate creating extra
    // instructions.
    if (isWorthFoldingAdd(Addr) &&
        selectConstantAddr(CurDAG, DL, VT, Subtarget, Addr.getOperand(1), Base,
                           Offset, /*IsPrefetch=*/false)) {
      // Insert an ADD instruction with the materialized Hi52 bits.
      Base = SDValue(
          CurDAG->getMachineNode(Capstone::ADD, DL, VT, Addr.getOperand(0), Base),
          0);
      return true;
    }
  }

  if (selectConstantAddr(CurDAG, DL, VT, Subtarget, Addr, Base, Offset,
                         /*IsPrefetch=*/false))
    return true;

  Base = Addr;
  Offset = CurDAG->getTargetConstant(0, DL, VT);
  return true;
}

/// Similar to SelectAddrRegImm, except that the offset is restricted to uimm9.
bool CapstoneDAGToDAGISel::SelectAddrRegImm9(SDValue Addr, SDValue &Base,
                                          SDValue &Offset) {
  if (SelectAddrFrameIndex(Addr, Base, Offset))
    return true;

  SDLoc DL(Addr);
  MVT VT = Addr.getSimpleValueType();

  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    int64_t CVal = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    if (isUInt<9>(CVal)) {
      Base = Addr.getOperand(0);

      if (auto *FIN = dyn_cast<FrameIndexSDNode>(Base))
        Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), VT);
      Offset = CurDAG->getSignedTargetConstant(CVal, DL, VT);
      return true;
    }
  }

  Base = Addr;
  Offset = CurDAG->getTargetConstant(0, DL, VT);
  return true;
}

/// Similar to SelectAddrRegImm, except that the least significant 5 bits of
/// Offset should be all zeros.
bool CapstoneDAGToDAGISel::SelectAddrRegImmLsb00000(SDValue Addr, SDValue &Base,
                                                 SDValue &Offset) {
  if (SelectAddrFrameIndex(Addr, Base, Offset))
    return true;

  SDLoc DL(Addr);
  MVT VT = Addr.getSimpleValueType();

  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    int64_t CVal = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    if (isInt<12>(CVal)) {
      Base = Addr.getOperand(0);

      // Early-out if not a valid offset.
      if ((CVal & 0b11111) != 0) {
        Base = Addr;
        Offset = CurDAG->getTargetConstant(0, DL, VT);
        return true;
      }

      if (auto *FIN = dyn_cast<FrameIndexSDNode>(Base))
        Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), VT);
      Offset = CurDAG->getSignedTargetConstant(CVal, DL, VT);
      return true;
    }
  }

  // Handle ADD with large immediates.
  if (Addr.getOpcode() == ISD::ADD && isa<ConstantSDNode>(Addr.getOperand(1))) {
    int64_t CVal = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    assert(!isInt<12>(CVal) && "simm12 not already handled?");

    // Handle immediates in the range [-4096,-2049] or [2017, 4065]. We can save
    // one instruction by folding adjustment (-2048 or 2016) into the address.
    if ((-2049 >= CVal && CVal >= -4096) || (4065 >= CVal && CVal >= 2017)) {
      int64_t Adj = CVal < 0 ? -2048 : 2016;
      int64_t AdjustedOffset = CVal - Adj;
      Base =
          SDValue(CurDAG->getMachineNode(
                      Capstone::ADDI, DL, VT, Addr.getOperand(0),
                      CurDAG->getSignedTargetConstant(AdjustedOffset, DL, VT)),
                  0);
      Offset = CurDAG->getSignedTargetConstant(Adj, DL, VT);
      return true;
    }

    if (selectConstantAddr(CurDAG, DL, VT, Subtarget, Addr.getOperand(1), Base,
                           Offset, /*IsPrefetch=*/true)) {
      // Insert an ADD instruction with the materialized Hi52 bits.
      Base = SDValue(
          CurDAG->getMachineNode(Capstone::ADD, DL, VT, Addr.getOperand(0), Base),
          0);
      return true;
    }
  }

  if (selectConstantAddr(CurDAG, DL, VT, Subtarget, Addr, Base, Offset,
                         /*IsPrefetch=*/true))
    return true;

  Base = Addr;
  Offset = CurDAG->getTargetConstant(0, DL, VT);
  return true;
}

/// Return true if this a load/store that we have a RegRegScale instruction for.
static bool isRegRegScaleLoadOrStore(SDNode *User, SDValue Add,
                                     const CapstoneSubtarget &Subtarget) {
  if (User->getOpcode() != ISD::LOAD && User->getOpcode() != ISD::STORE)
    return false;
  EVT VT = cast<MemSDNode>(User)->getMemoryVT();
  if (!(VT.isScalarInteger() &&
        (Subtarget.hasVendorXTHeadMemIdx() || Subtarget.hasVendorXqcisls())) &&
      !((VT == MVT::f32 || VT == MVT::f64) &&
        Subtarget.hasVendorXTHeadFMemIdx()))
    return false;
  // Don't allow stores of the value. It must be used as the address.
  if (User->getOpcode() == ISD::STORE &&
      cast<StoreSDNode>(User)->getValue() == Add)
    return false;

  return true;
}

/// Is it profitable to fold this Add into RegRegScale load/store. If \p
/// Shift is non-null, then we have matched a shl+add. We allow reassociating
/// (add (add (shl A C2) B) C1) -> (add (add B C1) (shl A C2)) if there is a
/// single addi and we don't have a SHXADD instruction we could use.
/// FIXME: May still need to check how many and what kind of users the SHL has.
static bool isWorthFoldingIntoRegRegScale(const CapstoneSubtarget &Subtarget,
                                          SDValue Add,
                                          SDValue Shift = SDValue()) {
  bool FoundADDI = false;
  for (auto *User : Add->users()) {
    if (isRegRegScaleLoadOrStore(User, Add, Subtarget))
      continue;

    // Allow a single ADDI that is used by loads/stores if we matched a shift.
    if (!Shift || FoundADDI || User->getOpcode() != ISD::ADD ||
        !isa<ConstantSDNode>(User->getOperand(1)) ||
        !isInt<12>(cast<ConstantSDNode>(User->getOperand(1))->getSExtValue()))
      return false;

    FoundADDI = true;

    // If we have a SHXADD instruction, prefer that over reassociating an ADDI.
    assert(Shift.getOpcode() == ISD::SHL);
    unsigned ShiftAmt = Shift.getConstantOperandVal(1);
    if (Subtarget.hasShlAdd(ShiftAmt))
      return false;

    // All users of the ADDI should be load/store.
    for (auto *ADDIUser : User->users())
      if (!isRegRegScaleLoadOrStore(ADDIUser, SDValue(User, 0), Subtarget))
        return false;
  }

  return true;
}

bool CapstoneDAGToDAGISel::SelectAddrRegRegScale(SDValue Addr,
                                              unsigned MaxShiftAmount,
                                              SDValue &Base, SDValue &Index,
                                              SDValue &Scale) {
  if (Addr.getOpcode() != ISD::ADD)
    return false;
  SDValue LHS = Addr.getOperand(0);
  SDValue RHS = Addr.getOperand(1);

  EVT VT = Addr.getSimpleValueType();
  auto SelectShl = [this, VT, MaxShiftAmount](SDValue N, SDValue &Index,
                                              SDValue &Shift) {
    if (N.getOpcode() != ISD::SHL || !isa<ConstantSDNode>(N.getOperand(1)))
      return false;

    // Only match shifts by a value in range [0, MaxShiftAmount].
    unsigned ShiftAmt = N.getConstantOperandVal(1);
    if (ShiftAmt > MaxShiftAmount)
      return false;

    Index = N.getOperand(0);
    Shift = CurDAG->getTargetConstant(ShiftAmt, SDLoc(N), VT);
    return true;
  };

  if (auto *C1 = dyn_cast<ConstantSDNode>(RHS)) {
    // (add (add (shl A C2) B) C1) -> (add (add B C1) (shl A C2))
    if (LHS.getOpcode() == ISD::ADD &&
        !isa<ConstantSDNode>(LHS.getOperand(1)) &&
        isInt<12>(C1->getSExtValue())) {
      if (SelectShl(LHS.getOperand(1), Index, Scale) &&
          isWorthFoldingIntoRegRegScale(*Subtarget, LHS, LHS.getOperand(1))) {
        SDValue C1Val = CurDAG->getTargetConstant(*C1->getConstantIntValue(),
                                                  SDLoc(Addr), VT);
        Base = SDValue(CurDAG->getMachineNode(Capstone::ADDI, SDLoc(Addr), VT,
                                              LHS.getOperand(0), C1Val),
                       0);
        return true;
      }

      // Add is commutative so we need to check both operands.
      if (SelectShl(LHS.getOperand(0), Index, Scale) &&
          isWorthFoldingIntoRegRegScale(*Subtarget, LHS, LHS.getOperand(0))) {
        SDValue C1Val = CurDAG->getTargetConstant(*C1->getConstantIntValue(),
                                                  SDLoc(Addr), VT);
        Base = SDValue(CurDAG->getMachineNode(Capstone::ADDI, SDLoc(Addr), VT,
                                              LHS.getOperand(1), C1Val),
                       0);
        return true;
      }
    }

    // Don't match add with constants.
    // FIXME: Is this profitable for large constants that have 0s in the lower
    // 12 bits that we can materialize with LUI?
    return false;
  }

  // Try to match a shift on the RHS.
  if (SelectShl(RHS, Index, Scale)) {
    if (!isWorthFoldingIntoRegRegScale(*Subtarget, Addr, RHS))
      return false;
    Base = LHS;
    return true;
  }

  // Try to match a shift on the LHS.
  if (SelectShl(LHS, Index, Scale)) {
    if (!isWorthFoldingIntoRegRegScale(*Subtarget, Addr, LHS))
      return false;
    Base = RHS;
    return true;
  }

  if (!isWorthFoldingIntoRegRegScale(*Subtarget, Addr))
    return false;

  Base = LHS;
  Index = RHS;
  Scale = CurDAG->getTargetConstant(0, SDLoc(Addr), VT);
  return true;
}

bool CapstoneDAGToDAGISel::SelectAddrRegZextRegScale(SDValue Addr,
                                                  unsigned MaxShiftAmount,
                                                  unsigned Bits, SDValue &Base,
                                                  SDValue &Index,
                                                  SDValue &Scale) {
  if (!SelectAddrRegRegScale(Addr, MaxShiftAmount, Base, Index, Scale))
    return false;

  if (Index.getOpcode() == ISD::AND) {
    auto *C = dyn_cast<ConstantSDNode>(Index.getOperand(1));
    if (C && C->getZExtValue() == maskTrailingOnes<uint64_t>(Bits)) {
      Index = Index.getOperand(0);
      return true;
    }
  }

  return false;
}

bool CapstoneDAGToDAGISel::SelectAddrRegReg(SDValue Addr, SDValue &Base,
                                         SDValue &Offset) {
  if (Addr.getOpcode() != ISD::ADD)
    return false;

  if (isa<ConstantSDNode>(Addr.getOperand(1)))
    return false;

  Base = Addr.getOperand(0);
  Offset = Addr.getOperand(1);
  return true;
}

bool CapstoneDAGToDAGISel::selectShiftMask(SDValue N, unsigned ShiftWidth,
                                        SDValue &ShAmt) {
  ShAmt = N;

  // Peek through zext.
  if (ShAmt->getOpcode() == ISD::ZERO_EXTEND)
    ShAmt = ShAmt.getOperand(0);

  // Shift instructions on Capstone only read the lower 5 or 6 bits of the shift
  // amount. If there is an AND on the shift amount, we can bypass it if it
  // doesn't affect any of those bits.
  if (ShAmt.getOpcode() == ISD::AND &&
      isa<ConstantSDNode>(ShAmt.getOperand(1))) {
    const APInt &AndMask = ShAmt.getConstantOperandAPInt(1);

    // Since the max shift amount is a power of 2 we can subtract 1 to make a
    // mask that covers the bits needed to represent all shift amounts.
    assert(isPowerOf2_32(ShiftWidth) && "Unexpected max shift amount!");
    APInt ShMask(AndMask.getBitWidth(), ShiftWidth - 1);

    if (ShMask.isSubsetOf(AndMask)) {
      ShAmt = ShAmt.getOperand(0);
    } else {
      // SimplifyDemandedBits may have optimized the mask so try restoring any
      // bits that are known zero.
      KnownBits Known = CurDAG->computeKnownBits(ShAmt.getOperand(0));
      if (!ShMask.isSubsetOf(AndMask | Known.Zero))
        return true;
      ShAmt = ShAmt.getOperand(0);
    }
  }

  if (ShAmt.getOpcode() == ISD::ADD &&
      isa<ConstantSDNode>(ShAmt.getOperand(1))) {
    uint64_t Imm = ShAmt.getConstantOperandVal(1);
    // If we are shifting by X+N where N == 0 mod Size, then just shift by X
    // to avoid the ADD.
    if (Imm != 0 && Imm % ShiftWidth == 0) {
      ShAmt = ShAmt.getOperand(0);
      return true;
    }
  } else if (ShAmt.getOpcode() == ISD::SUB &&
             isa<ConstantSDNode>(ShAmt.getOperand(0))) {
    uint64_t Imm = ShAmt.getConstantOperandVal(0);
    // If we are shifting by N-X where N == 0 mod Size, then just shift by -X to
    // generate a NEG instead of a SUB of a constant.
    if (Imm != 0 && Imm % ShiftWidth == 0) {
      SDLoc DL(ShAmt);
      EVT VT = ShAmt.getValueType();
      SDValue Zero = CurDAG->getRegister(Capstone::X0, VT);
      unsigned NegOpc = VT == MVT::i64 ? Capstone::SUBW : Capstone::SUB;
      MachineSDNode *Neg = CurDAG->getMachineNode(NegOpc, DL, VT, Zero,
                                                  ShAmt.getOperand(1));
      ShAmt = SDValue(Neg, 0);
      return true;
    }
    // If we are shifting by N-X where N == -1 mod Size, then just shift by ~X
    // to generate a NOT instead of a SUB of a constant.
    if (Imm % ShiftWidth == ShiftWidth - 1) {
      SDLoc DL(ShAmt);
      EVT VT = ShAmt.getValueType();
      MachineSDNode *Not = CurDAG->getMachineNode(
          Capstone::XORI, DL, VT, ShAmt.getOperand(1),
          CurDAG->getAllOnesConstant(DL, VT, /*isTarget=*/true));
      ShAmt = SDValue(Not, 0);
      return true;
    }
  }

  return true;
}

/// Capstone doesn't have general instructions for integer setne/seteq, but we can
/// check for equality with 0. This function emits instructions that convert the
/// seteq/setne into something that can be compared with 0.
/// \p ExpectedCCVal indicates the condition code to attempt to match (e.g.
/// ISD::SETNE).
bool CapstoneDAGToDAGISel::selectSETCC(SDValue N, ISD::CondCode ExpectedCCVal,
                                    SDValue &Val) {
  assert(ISD::isIntEqualitySetCC(ExpectedCCVal) &&
         "Unexpected condition code!");

  // We're looking for a setcc.
  if (N->getOpcode() != ISD::SETCC)
    return false;

  // Must be an equality comparison.
  ISD::CondCode CCVal = cast<CondCodeSDNode>(N->getOperand(2))->get();
  if (CCVal != ExpectedCCVal)
    return false;

  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  if (!LHS.getValueType().isScalarInteger())
    return false;

  // If the RHS side is 0, we don't need any extra instructions, return the LHS.
  if (isNullConstant(RHS)) {
    Val = LHS;
    return true;
  }

  SDLoc DL(N);

  if (auto *C = dyn_cast<ConstantSDNode>(RHS)) {
    int64_t CVal = C->getSExtValue();
    // If the RHS is -2048, we can use xori to produce 0 if the LHS is -2048 and
    // non-zero otherwise.
    if (CVal == -2048) {
      Val = SDValue(
          CurDAG->getMachineNode(
              Capstone::XORI, DL, N->getValueType(0), LHS,
              CurDAG->getSignedTargetConstant(CVal, DL, N->getValueType(0))),
          0);
      return true;
    }
    // If the RHS is [-2047,2048], we can use addi with -RHS to produce 0 if the
    // LHS is equal to the RHS and non-zero otherwise.
    if (isInt<12>(CVal) || CVal == 2048) {
      Val = SDValue(
          CurDAG->getMachineNode(
              Capstone::ADDI, DL, N->getValueType(0), LHS,
              CurDAG->getSignedTargetConstant(-CVal, DL, N->getValueType(0))),
          0);
      return true;
    }
    if (isPowerOf2_64(CVal) && Subtarget->hasStdExtZbs()) {
      Val = SDValue(
          CurDAG->getMachineNode(
              Capstone::BINVI, DL, N->getValueType(0), LHS,
              CurDAG->getTargetConstant(Log2_64(CVal), DL, N->getValueType(0))),
          0);
      return true;
    }
    // Same as the addi case above but for larger immediates (signed 26-bit) use
    // the QC_E_ADDI instruction from the Xqcilia extension, if available. Avoid
    // anything which can be done with a single lui as it might be compressible.
    if (Subtarget->hasVendorXqcilia() && isInt<26>(CVal) &&
        (CVal & 0xFFF) != 0) {
      Val = SDValue(
          CurDAG->getMachineNode(
              Capstone::QC_E_ADDI, DL, N->getValueType(0), LHS,
              CurDAG->getSignedTargetConstant(-CVal, DL, N->getValueType(0))),
          0);
      return true;
    }
  }

  // If nothing else we can XOR the LHS and RHS to produce zero if they are
  // equal and a non-zero value if they aren't.
  Val = SDValue(
      CurDAG->getMachineNode(Capstone::XOR, DL, N->getValueType(0), LHS, RHS), 0);
  return true;
}

bool CapstoneDAGToDAGISel::selectSExtBits(SDValue N, unsigned Bits, SDValue &Val) {
  if (N.getOpcode() == ISD::SIGN_EXTEND_INREG &&
      cast<VTSDNode>(N.getOperand(1))->getVT().getSizeInBits() == Bits) {
    Val = N.getOperand(0);
    return true;
  }

  auto UnwrapShlSra = [](SDValue N, unsigned ShiftAmt) {
    if (N.getOpcode() != ISD::SRA || !isa<ConstantSDNode>(N.getOperand(1)))
      return N;

    SDValue N0 = N.getOperand(0);
    if (N0.getOpcode() == ISD::SHL && isa<ConstantSDNode>(N0.getOperand(1)) &&
        N.getConstantOperandVal(1) == ShiftAmt &&
        N0.getConstantOperandVal(1) == ShiftAmt)
      return N0.getOperand(0);

    return N;
  };

  MVT VT = N.getSimpleValueType();
  if (CurDAG->ComputeNumSignBits(N) > (VT.getSizeInBits() - Bits)) {
    Val = UnwrapShlSra(N, VT.getSizeInBits() - Bits);
    return true;
  }

  return false;
}

bool CapstoneDAGToDAGISel::selectZExtBits(SDValue N, unsigned Bits, SDValue &Val) {
  if (N.getOpcode() == ISD::AND) {
    auto *C = dyn_cast<ConstantSDNode>(N.getOperand(1));
    if (C && C->getZExtValue() == maskTrailingOnes<uint64_t>(Bits)) {
      Val = N.getOperand(0);
      return true;
    }
  }
  MVT VT = N.getSimpleValueType();
  APInt Mask = APInt::getBitsSetFrom(VT.getSizeInBits(), Bits);
  if (CurDAG->MaskedValueIsZero(N, Mask)) {
    Val = N;
    return true;
  }

  return false;
}

/// Look for various patterns that can be done with a SHL that can be folded
/// into a SHXADD. \p ShAmt contains 1, 2, or 3 and is set based on which
/// SHXADD we are trying to match.
bool CapstoneDAGToDAGISel::selectSHXADDOp(SDValue N, unsigned ShAmt,
                                       SDValue &Val) {
  if (N.getOpcode() == ISD::AND && isa<ConstantSDNode>(N.getOperand(1))) {
    SDValue N0 = N.getOperand(0);

    if (bool LeftShift = N0.getOpcode() == ISD::SHL;
        (LeftShift || N0.getOpcode() == ISD::SRL) &&
        isa<ConstantSDNode>(N0.getOperand(1))) {
      uint64_t Mask = N.getConstantOperandVal(1);
      unsigned C2 = N0.getConstantOperandVal(1);

      unsigned XLen = Subtarget->getXLen();
      if (LeftShift)
        Mask &= maskTrailingZeros<uint64_t>(C2);
      else
        Mask &= maskTrailingOnes<uint64_t>(XLen - C2);

      if (isShiftedMask_64(Mask)) {
        unsigned Leading = XLen - llvm::bit_width(Mask);
        unsigned Trailing = llvm::countr_zero(Mask);
        if (Trailing != ShAmt)
          return false;

        unsigned Opcode;
        // Look for (and (shl y, c2), c1) where c1 is a shifted mask with no
        // leading zeros and c3 trailing zeros. We can use an SRLI by c3-c2
        // followed by a SHXADD with c3 for the X amount.
        if (LeftShift && Leading == 0 && C2 < Trailing)
          Opcode = Capstone::SRLI;
        // Look for (and (shl y, c2), c1) where c1 is a shifted mask with 32-c2
        // leading zeros and c3 trailing zeros. We can use an SRLIW by c3-c2
        // followed by a SHXADD with c3 for the X amount.
        else if (LeftShift && Leading == 32 - C2 && C2 < Trailing)
          Opcode = Capstone::SRLIW;
        // Look for (and (shr y, c2), c1) where c1 is a shifted mask with c2
        // leading zeros and c3 trailing zeros. We can use an SRLI by c2+c3
        // followed by a SHXADD using c3 for the X amount.
        else if (!LeftShift && Leading == C2)
          Opcode = Capstone::SRLI;
        // Look for (and (shr y, c2), c1) where c1 is a shifted mask with 32+c2
        // leading zeros and c3 trailing zeros. We can use an SRLIW by c2+c3
        // followed by a SHXADD using c3 for the X amount.
        else if (!LeftShift && Leading == 32 + C2)
          Opcode = Capstone::SRLIW;
        else
          return false;

        SDLoc DL(N);
        EVT VT = N.getValueType();
        ShAmt = LeftShift ? Trailing - C2 : Trailing + C2;
        Val = SDValue(
            CurDAG->getMachineNode(Opcode, DL, VT, N0.getOperand(0),
                                   CurDAG->getTargetConstant(ShAmt, DL, VT)),
            0);
        return true;
      }
    } else if (N0.getOpcode() == ISD::SRA && N0.hasOneUse() &&
               isa<ConstantSDNode>(N0.getOperand(1))) {
      uint64_t Mask = N.getConstantOperandVal(1);
      unsigned C2 = N0.getConstantOperandVal(1);

      // Look for (and (sra y, c2), c1) where c1 is a shifted mask with c3
      // leading zeros and c4 trailing zeros. If c2 is greater than c3, we can
      // use (srli (srai y, c2 - c3), c3 + c4) followed by a SHXADD with c4 as
      // the X amount.
      if (isShiftedMask_64(Mask)) {
        unsigned XLen = Subtarget->getXLen();
        unsigned Leading = XLen - llvm::bit_width(Mask);
        unsigned Trailing = llvm::countr_zero(Mask);
        if (C2 > Leading && Leading > 0 && Trailing == ShAmt) {
          SDLoc DL(N);
          EVT VT = N.getValueType();
          Val = SDValue(CurDAG->getMachineNode(
                            Capstone::SRAI, DL, VT, N0.getOperand(0),
                            CurDAG->getTargetConstant(C2 - Leading, DL, VT)),
                        0);
          Val = SDValue(CurDAG->getMachineNode(
                            Capstone::SRLI, DL, VT, Val,
                            CurDAG->getTargetConstant(Leading + ShAmt, DL, VT)),
                        0);
          return true;
        }
      }
    }
  } else if (bool LeftShift = N.getOpcode() == ISD::SHL;
             (LeftShift || N.getOpcode() == ISD::SRL) &&
             isa<ConstantSDNode>(N.getOperand(1))) {
    SDValue N0 = N.getOperand(0);
    if (N0.getOpcode() == ISD::AND && N0.hasOneUse() &&
        isa<ConstantSDNode>(N0.getOperand(1))) {
      uint64_t Mask = N0.getConstantOperandVal(1);
      if (isShiftedMask_64(Mask)) {
        unsigned C1 = N.getConstantOperandVal(1);
        unsigned XLen = Subtarget->getXLen();
        unsigned Leading = XLen - llvm::bit_width(Mask);
        unsigned Trailing = llvm::countr_zero(Mask);
        // Look for (shl (and X, Mask), C1) where Mask has 32 leading zeros and
        // C3 trailing zeros. If C1+C3==ShAmt we can use SRLIW+SHXADD.
        if (LeftShift && Leading == 32 && Trailing > 0 &&
            (Trailing + C1) == ShAmt) {
          SDLoc DL(N);
          EVT VT = N.getValueType();
          Val = SDValue(CurDAG->getMachineNode(
                            Capstone::SRLIW, DL, VT, N0.getOperand(0),
                            CurDAG->getTargetConstant(Trailing, DL, VT)),
                        0);
          return true;
        }
        // Look for (srl (and X, Mask), C1) where Mask has 32 leading zeros and
        // C3 trailing zeros. If C3-C1==ShAmt we can use SRLIW+SHXADD.
        if (!LeftShift && Leading == 32 && Trailing > C1 &&
            (Trailing - C1) == ShAmt) {
          SDLoc DL(N);
          EVT VT = N.getValueType();
          Val = SDValue(CurDAG->getMachineNode(
                            Capstone::SRLIW, DL, VT, N0.getOperand(0),
                            CurDAG->getTargetConstant(Trailing, DL, VT)),
                        0);
          return true;
        }
      }
    }
  }

  return false;
}

/// Look for various patterns that can be done with a SHL that can be folded
/// into a SHXADD_UW. \p ShAmt contains 1, 2, or 3 and is set based on which
/// SHXADD_UW we are trying to match.
bool CapstoneDAGToDAGISel::selectSHXADD_UWOp(SDValue N, unsigned ShAmt,
                                          SDValue &Val) {
  if (N.getOpcode() == ISD::AND && isa<ConstantSDNode>(N.getOperand(1)) &&
      N.hasOneUse()) {
    SDValue N0 = N.getOperand(0);
    if (N0.getOpcode() == ISD::SHL && isa<ConstantSDNode>(N0.getOperand(1)) &&
        N0.hasOneUse()) {
      uint64_t Mask = N.getConstantOperandVal(1);
      unsigned C2 = N0.getConstantOperandVal(1);

      Mask &= maskTrailingZeros<uint64_t>(C2);

      // Look for (and (shl y, c2), c1) where c1 is a shifted mask with
      // 32-ShAmt leading zeros and c2 trailing zeros. We can use SLLI by
      // c2-ShAmt followed by SHXADD_UW with ShAmt for the X amount.
      if (isShiftedMask_64(Mask)) {
        unsigned Leading = llvm::countl_zero(Mask);
        unsigned Trailing = llvm::countr_zero(Mask);
        if (Leading == 32 - ShAmt && Trailing == C2 && Trailing > ShAmt) {
          SDLoc DL(N);
          EVT VT = N.getValueType();
          Val = SDValue(CurDAG->getMachineNode(
                            Capstone::SLLI, DL, VT, N0.getOperand(0),
                            CurDAG->getTargetConstant(C2 - ShAmt, DL, VT)),
                        0);
          return true;
        }
      }
    }
  }

  return false;
}

bool CapstoneDAGToDAGISel::orDisjoint(const SDNode *N) const {
  assert(N->getOpcode() == ISD::OR || N->getOpcode() == CapstoneISD::OR_VL);
  if (N->getFlags().hasDisjoint())
    return true;
  return CurDAG->haveNoCommonBitsSet(N->getOperand(0), N->getOperand(1));
}

bool CapstoneDAGToDAGISel::selectImm64IfCheaper(int64_t Imm, int64_t OrigImm,
                                             SDValue N, SDValue &Val) {
  int OrigCost = CapstoneMatInt::getIntMatCost(APInt(64, OrigImm), 64, *Subtarget,
                                            /*CompressionCost=*/true);
  int Cost = CapstoneMatInt::getIntMatCost(APInt(64, Imm), 64, *Subtarget,
                                        /*CompressionCost=*/true);
  if (OrigCost <= Cost)
    return false;

  Val = selectImm(CurDAG, SDLoc(N), N->getSimpleValueType(0), Imm, *Subtarget);
  return true;
}

bool CapstoneDAGToDAGISel::selectZExtImm32(SDValue N, SDValue &Val) {
  if (!isa<ConstantSDNode>(N))
    return false;
  int64_t Imm = cast<ConstantSDNode>(N)->getSExtValue();
  if ((Imm >> 31) != 1)
    return false;

  for (const SDNode *U : N->users()) {
    switch (U->getOpcode()) {
    case ISD::ADD:
      break;
    case ISD::OR:
      if (orDisjoint(U))
        break;
      return false;
    default:
      return false;
    }
  }

  return selectImm64IfCheaper(0xffffffff00000000 | Imm, Imm, N, Val);
}

bool CapstoneDAGToDAGISel::selectNegImm(SDValue N, SDValue &Val) {
  if (!isa<ConstantSDNode>(N))
    return false;
  int64_t Imm = cast<ConstantSDNode>(N)->getSExtValue();
  if (isInt<32>(Imm))
    return false;

  for (const SDNode *U : N->users()) {
    switch (U->getOpcode()) {
    case ISD::ADD:
      break;
    case CapstoneISD::VMV_V_X_VL:
      if (!all_of(U->users(), [](const SDNode *V) {
            return V->getOpcode() == ISD::ADD ||
                   V->getOpcode() == CapstoneISD::ADD_VL;
          }))
        return false;
      break;
    default:
      return false;
    }
  }

  return selectImm64IfCheaper(-Imm, Imm, N, Val);
}

bool CapstoneDAGToDAGISel::selectInvLogicImm(SDValue N, SDValue &Val) {
  if (!isa<ConstantSDNode>(N))
    return false;
  int64_t Imm = cast<ConstantSDNode>(N)->getSExtValue();

  // For 32-bit signed constants, we can only substitute LUI+ADDI with LUI.
  if (isInt<32>(Imm) && ((Imm & 0xfff) != 0xfff || Imm == -1))
    return false;

  // Abandon this transform if the constant is needed elsewhere.
  for (const SDNode *U : N->users()) {
    switch (U->getOpcode()) {
    case ISD::AND:
    case ISD::OR:
    case ISD::XOR:
      if (!(Subtarget->hasStdExtZbb() || Subtarget->hasStdExtZbkb()))
        return false;
      break;
    case CapstoneISD::VMV_V_X_VL:
      if (!Subtarget->hasStdExtZvkb())
        return false;
      if (!all_of(U->users(), [](const SDNode *V) {
            return V->getOpcode() == ISD::AND ||
                   V->getOpcode() == CapstoneISD::AND_VL;
          }))
        return false;
      break;
    default:
      return false;
    }
  }

  if (isInt<32>(Imm)) {
    Val =
        selectImm(CurDAG, SDLoc(N), N->getSimpleValueType(0), ~Imm, *Subtarget);
    return true;
  }

  // For 64-bit constants, the instruction sequences get complex,
  // so we select inverted only if it's cheaper.
  return selectImm64IfCheaper(~Imm, Imm, N, Val);
}

static bool vectorPseudoHasAllNBitUsers(SDNode *User, unsigned UserOpNo,
                                        unsigned Bits,
                                        const TargetInstrInfo *TII) {
  unsigned MCOpcode = Capstone::getRVVMCOpcode(User->getMachineOpcode());

  if (!MCOpcode)
    return false;

  const MCInstrDesc &MCID = TII->get(User->getMachineOpcode());
  const uint64_t TSFlags = MCID.TSFlags;
  if (!CapstoneII::hasSEWOp(TSFlags))
    return false;
  assert(CapstoneII::hasVLOp(TSFlags));

  unsigned ChainOpIdx = User->getNumOperands() - 1;
  bool HasChainOp = User->getOperand(ChainOpIdx).getValueType() == MVT::Other;
  bool HasVecPolicyOp = CapstoneII::hasVecPolicyOp(TSFlags);
  unsigned VLIdx = User->getNumOperands() - HasVecPolicyOp - HasChainOp - 2;
  const unsigned Log2SEW = User->getConstantOperandVal(VLIdx + 1);

  if (UserOpNo == VLIdx)
    return false;

  auto NumDemandedBits =
      Capstone::getVectorLowDemandedScalarBits(MCOpcode, Log2SEW);
  return NumDemandedBits && Bits >= *NumDemandedBits;
}

// Return true if all users of this SDNode* only consume the lower \p Bits.
// This can be used to form W instructions for add/sub/mul/shl even when the
// root isn't a sext_inreg. This can allow the ADDW/SUBW/MULW/SLLIW to CSE if
// SimplifyDemandedBits has made it so some users see a sext_inreg and some
// don't. The sext_inreg+add/sub/mul/shl will get selected, but still leave
// the add/sub/mul/shl to become non-W instructions. By checking the users we
// may be able to use a W instruction and CSE with the other instruction if
// this has happened. We could try to detect that the CSE opportunity exists
// before doing this, but that would be more complicated.
bool CapstoneDAGToDAGISel::hasAllNBitUsers(SDNode *Node, unsigned Bits,
                                        const unsigned Depth) const {
  assert((Node->getOpcode() == ISD::ADD || Node->getOpcode() == ISD::SUB ||
          Node->getOpcode() == ISD::MUL || Node->getOpcode() == ISD::SHL ||
          Node->getOpcode() == ISD::SRL || Node->getOpcode() == ISD::AND ||
          Node->getOpcode() == ISD::OR || Node->getOpcode() == ISD::XOR ||
          Node->getOpcode() == ISD::SIGN_EXTEND_INREG ||
          isa<ConstantSDNode>(Node) || Depth != 0) &&
         "Unexpected opcode");

  if (Depth >= SelectionDAG::MaxRecursionDepth)
    return false;

  // The PatFrags that call this may run before CapstoneGenDAGISel.inc has checked
  // the VT. Ensure the type is scalar to avoid wasting time on vectors.
  if (Depth == 0 && !Node->getValueType(0).isScalarInteger())
    return false;

  for (SDUse &Use : Node->uses()) {
    SDNode *User = Use.getUser();
    // Users of this node should have already been instruction selected
    if (!User->isMachineOpcode())
      return false;

    // TODO: Add more opcodes?
    switch (User->getMachineOpcode()) {
    default:
      if (vectorPseudoHasAllNBitUsers(User, Use.getOperandNo(), Bits, TII))
        break;
      return false;
    case Capstone::ADDW:
    case Capstone::ADDIW:
    case Capstone::SUBW:
    case Capstone::MULW:
    case Capstone::SLLW:
    case Capstone::SLLIW:
    case Capstone::SRAW:
    case Capstone::SRAIW:
    case Capstone::SRLW:
    case Capstone::SRLIW:
    case Capstone::DIVW:
    case Capstone::DIVUW:
    case Capstone::REMW:
    case Capstone::REMUW:
    case Capstone::ROLW:
    case Capstone::RORW:
    case Capstone::RORIW:
    case Capstone::CLZW:
    case Capstone::CTZW:
    case Capstone::CPOPW:
    case Capstone::SLLI_UW:
    case Capstone::FMV_W_X:
    case Capstone::FCVT_H_W:
    case Capstone::FCVT_H_W_INX:
    case Capstone::FCVT_H_WU:
    case Capstone::FCVT_H_WU_INX:
    case Capstone::FCVT_S_W:
    case Capstone::FCVT_S_W_INX:
    case Capstone::FCVT_S_WU:
    case Capstone::FCVT_S_WU_INX:
    case Capstone::FCVT_D_W:
    case Capstone::FCVT_D_W_INX:
    case Capstone::FCVT_D_WU:
    case Capstone::FCVT_D_WU_INX:
    case Capstone::TH_REVW:
    case Capstone::TH_SRRIW:
      if (Bits >= 32)
        break;
      return false;
    case Capstone::SLL:
    case Capstone::SRA:
    case Capstone::SRL:
    case Capstone::ROL:
    case Capstone::ROR:
    case Capstone::BSET:
    case Capstone::BCLR:
    case Capstone::BINV:
      // Shift amount operands only use log2(Xlen) bits.
      if (Use.getOperandNo() == 1 && Bits >= Log2_32(Subtarget->getXLen()))
        break;
      return false;
    case Capstone::SLLI:
      // SLLI only uses the lower (XLen - ShAmt) bits.
      if (Bits >= Subtarget->getXLen() - User->getConstantOperandVal(1))
        break;
      return false;
    case Capstone::ANDI:
      if (Bits >= (unsigned)llvm::bit_width(User->getConstantOperandVal(1)))
        break;
      goto RecCheck;
    case Capstone::ORI: {
      uint64_t Imm = cast<ConstantSDNode>(User->getOperand(1))->getSExtValue();
      if (Bits >= (unsigned)llvm::bit_width<uint64_t>(~Imm))
        break;
      [[fallthrough]];
    }
    case Capstone::AND:
    case Capstone::OR:
    case Capstone::XOR:
    case Capstone::XORI:
    case Capstone::ANDN:
    case Capstone::ORN:
    case Capstone::XNOR:
    case Capstone::SH1ADD:
    case Capstone::SH2ADD:
    case Capstone::SH3ADD:
    RecCheck:
      if (hasAllNBitUsers(User, Bits, Depth + 1))
        break;
      return false;
    case Capstone::SRLI: {
      unsigned ShAmt = User->getConstantOperandVal(1);
      // If we are shifting right by less than Bits, and users don't demand any
      // bits that were shifted into [Bits-1:0], then we can consider this as an
      // N-Bit user.
      if (Bits > ShAmt && hasAllNBitUsers(User, Bits - ShAmt, Depth + 1))
        break;
      return false;
    }
    case Capstone::SEXT_B:
    case Capstone::PACKH:
      if (Bits >= 8)
        break;
      return false;
    case Capstone::SEXT_H:
    case Capstone::FMV_H_X:
    case Capstone::ZEXT_H_RV32:
    case Capstone::ZEXT_H_RV64:
    case Capstone::PACKW:
      if (Bits >= 16)
        break;
      return false;
    case Capstone::PACK:
      if (Bits >= (Subtarget->getXLen() / 2))
        break;
      return false;
    case Capstone::ADD_UW:
    case Capstone::SH1ADD_UW:
    case Capstone::SH2ADD_UW:
    case Capstone::SH3ADD_UW:
      // The first operand to add.uw/shXadd.uw is implicitly zero extended from
      // 32 bits.
      if (Use.getOperandNo() == 0 && Bits >= 32)
        break;
      return false;
    case Capstone::SB:
      if (Use.getOperandNo() == 0 && Bits >= 8)
        break;
      return false;
    case Capstone::SH:
      if (Use.getOperandNo() == 0 && Bits >= 16)
        break;
      return false;
    case Capstone::SW:
      if (Use.getOperandNo() == 0 && Bits >= 32)
        break;
      return false;
    case Capstone::TH_EXT:
    case Capstone::TH_EXTU: {
      unsigned Msb = User->getConstantOperandVal(1);
      unsigned Lsb = User->getConstantOperandVal(2);
      // Behavior of Msb < Lsb is not well documented.
      if (Msb >= Lsb && Bits > Msb)
        break;
      return false;
    }
    }
  }

  return true;
}

// Select a constant that can be represented as (sign_extend(imm5) << imm2).
bool CapstoneDAGToDAGISel::selectSimm5Shl2(SDValue N, SDValue &Simm5,
                                        SDValue &Shl2) {
  auto *C = dyn_cast<ConstantSDNode>(N);
  if (!C)
    return false;

  int64_t Offset = C->getSExtValue();
  for (unsigned Shift = 0; Shift < 4; Shift++) {
    if (isInt<5>(Offset >> Shift) && ((Offset % (1LL << Shift)) == 0)) {
      EVT VT = N->getValueType(0);
      Simm5 = CurDAG->getSignedTargetConstant(Offset >> Shift, SDLoc(N), VT);
      Shl2 = CurDAG->getTargetConstant(Shift, SDLoc(N), VT);
      return true;
    }
  }

  return false;
}

// Select VL as a 5 bit immediate or a value that will become a register. This
// allows us to choose between VSETIVLI or VSETVLI later.
bool CapstoneDAGToDAGISel::selectVLOp(SDValue N, SDValue &VL) {
  auto *C = dyn_cast<ConstantSDNode>(N);
  if (C && isUInt<5>(C->getZExtValue())) {
    VL = CurDAG->getTargetConstant(C->getZExtValue(), SDLoc(N),
                                   N->getValueType(0));
  } else if (C && C->isAllOnes()) {
    // Treat all ones as VLMax.
    VL = CurDAG->getSignedTargetConstant(Capstone::VLMaxSentinel, SDLoc(N),
                                         N->getValueType(0));
  } else if (isa<RegisterSDNode>(N) &&
             cast<RegisterSDNode>(N)->getReg() == Capstone::X0) {
    // All our VL operands use an operand that allows GPRNoX0 or an immediate
    // as the register class. Convert X0 to a special immediate to pass the
    // MachineVerifier. This is recognized specially by the vsetvli insertion
    // pass.
    VL = CurDAG->getSignedTargetConstant(Capstone::VLMaxSentinel, SDLoc(N),
                                         N->getValueType(0));
  } else {
    VL = N;
  }

  return true;
}

static SDValue findVSplat(SDValue N) {
  if (N.getOpcode() == ISD::INSERT_SUBVECTOR) {
    if (!N.getOperand(0).isUndef())
      return SDValue();
    N = N.getOperand(1);
  }
  SDValue Splat = N;
  if ((Splat.getOpcode() != CapstoneISD::VMV_V_X_VL &&
       Splat.getOpcode() != CapstoneISD::VMV_S_X_VL) ||
      !Splat.getOperand(0).isUndef())
    return SDValue();
  assert(Splat.getNumOperands() == 3 && "Unexpected number of operands");
  return Splat;
}

bool CapstoneDAGToDAGISel::selectVSplat(SDValue N, SDValue &SplatVal) {
  SDValue Splat = findVSplat(N);
  if (!Splat)
    return false;

  SplatVal = Splat.getOperand(1);
  return true;
}

static bool selectVSplatImmHelper(SDValue N, SDValue &SplatVal,
                                  SelectionDAG &DAG,
                                  const CapstoneSubtarget &Subtarget,
                                  std::function<bool(int64_t)> ValidateImm,
                                  bool Decrement = false) {
  SDValue Splat = findVSplat(N);
  if (!Splat || !isa<ConstantSDNode>(Splat.getOperand(1)))
    return false;

  const unsigned SplatEltSize = Splat.getScalarValueSizeInBits();
  assert(Subtarget.getXLenVT() == Splat.getOperand(1).getSimpleValueType() &&
         "Unexpected splat operand type");

  // The semantics of CapstoneISD::VMV_V_X_VL is that when the operand
  // type is wider than the resulting vector element type: an implicit
  // truncation first takes place. Therefore, perform a manual
  // truncation/sign-extension in order to ignore any truncated bits and catch
  // any zero-extended immediate.
  // For example, we wish to match (i8 -1) -> (XLenVT 255) as a simm5 by first
  // sign-extending to (XLenVT -1).
  APInt SplatConst = Splat.getConstantOperandAPInt(1).sextOrTrunc(SplatEltSize);

  int64_t SplatImm = SplatConst.getSExtValue();

  if (!ValidateImm(SplatImm))
    return false;

  if (Decrement)
    SplatImm -= 1;

  SplatVal =
      DAG.getSignedTargetConstant(SplatImm, SDLoc(N), Subtarget.getXLenVT());
  return true;
}

bool CapstoneDAGToDAGISel::selectVSplatSimm5(SDValue N, SDValue &SplatVal) {
  return selectVSplatImmHelper(N, SplatVal, *CurDAG, *Subtarget,
                               [](int64_t Imm) { return isInt<5>(Imm); });
}

bool CapstoneDAGToDAGISel::selectVSplatSimm5Plus1(SDValue N, SDValue &SplatVal) {
  return selectVSplatImmHelper(
      N, SplatVal, *CurDAG, *Subtarget,
      [](int64_t Imm) { return (isInt<5>(Imm) && Imm != -16) || Imm == 16; },
      /*Decrement=*/true);
}

bool CapstoneDAGToDAGISel::selectVSplatSimm5Plus1NoDec(SDValue N, SDValue &SplatVal) {
  return selectVSplatImmHelper(
      N, SplatVal, *CurDAG, *Subtarget,
      [](int64_t Imm) { return (isInt<5>(Imm) && Imm != -16) || Imm == 16; },
      /*Decrement=*/false);
}

bool CapstoneDAGToDAGISel::selectVSplatSimm5Plus1NonZero(SDValue N,
                                                      SDValue &SplatVal) {
  return selectVSplatImmHelper(
      N, SplatVal, *CurDAG, *Subtarget,
      [](int64_t Imm) {
        return Imm != 0 && ((isInt<5>(Imm) && Imm != -16) || Imm == 16);
      },
      /*Decrement=*/true);
}

bool CapstoneDAGToDAGISel::selectVSplatUimm(SDValue N, unsigned Bits,
                                         SDValue &SplatVal) {
  return selectVSplatImmHelper(
      N, SplatVal, *CurDAG, *Subtarget,
      [Bits](int64_t Imm) { return isUIntN(Bits, Imm); });
}

bool CapstoneDAGToDAGISel::selectVSplatImm64Neg(SDValue N, SDValue &SplatVal) {
  SDValue Splat = findVSplat(N);
  return Splat && selectNegImm(Splat.getOperand(1), SplatVal);
}

bool CapstoneDAGToDAGISel::selectLow8BitsVSplat(SDValue N, SDValue &SplatVal) {
  auto IsExtOrTrunc = [](SDValue N) {
    switch (N->getOpcode()) {
    case ISD::SIGN_EXTEND:
    case ISD::ZERO_EXTEND:
    // There's no passthru on these _VL nodes so any VL/mask is ok, since any
    // inactive elements will be undef.
    case CapstoneISD::TRUNCATE_VECTOR_VL:
    case CapstoneISD::VSEXT_VL:
    case CapstoneISD::VZEXT_VL:
      return true;
    default:
      return false;
    }
  };

  // We can have multiple nested nodes, so unravel them all if needed.
  while (IsExtOrTrunc(N)) {
    if (!N.hasOneUse() || N.getScalarValueSizeInBits() < 8)
      return false;
    N = N->getOperand(0);
  }

  return selectVSplat(N, SplatVal);
}

bool CapstoneDAGToDAGISel::selectScalarFPAsInt(SDValue N, SDValue &Imm) {
  // Allow bitcasts from XLenVT -> FP.
  if (N.getOpcode() == ISD::BITCAST &&
      N.getOperand(0).getValueType() == Subtarget->getXLenVT()) {
    Imm = N.getOperand(0);
    return true;
  }
  // Allow moves from XLenVT to FP.
  if (N.getOpcode() == CapstoneISD::FMV_H_X ||
      N.getOpcode() == CapstoneISD::FMV_W_X_RV64) {
    Imm = N.getOperand(0);
    return true;
  }

  // Otherwise, look for FP constants that can materialized with scalar int.
  ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N.getNode());
  if (!CFP)
    return false;
  const APFloat &APF = CFP->getValueAPF();
  // td can handle +0.0 already.
  if (APF.isPosZero())
    return false;

  MVT VT = CFP->getSimpleValueType(0);

  MVT XLenVT = Subtarget->getXLenVT();
  if (VT == MVT::f64 && !Subtarget->is64Bit()) {
    assert(APF.isNegZero() && "Unexpected constant.");
    return false;
  }
  SDLoc DL(N);
  Imm = selectImm(CurDAG, DL, XLenVT, APF.bitcastToAPInt().getSExtValue(),
                  *Subtarget);
  return true;
}

bool CapstoneDAGToDAGISel::selectRVVSimm5(SDValue N, unsigned Width,
                                       SDValue &Imm) {
  if (auto *C = dyn_cast<ConstantSDNode>(N)) {
    int64_t ImmVal = SignExtend64(C->getSExtValue(), Width);

    if (!isInt<5>(ImmVal))
      return false;

    Imm = CurDAG->getSignedTargetConstant(ImmVal, SDLoc(N),
                                          Subtarget->getXLenVT());
    return true;
  }

  return false;
}

// Try to remove sext.w if the input is a W instruction or can be made into
// a W instruction cheaply.
bool CapstoneDAGToDAGISel::doPeepholeSExtW(SDNode *N) {
  // Look for the sext.w pattern, addiw rd, rs1, 0.
  if (N->getMachineOpcode() != Capstone::ADDIW ||
      !isNullConstant(N->getOperand(1)))
    return false;

  SDValue N0 = N->getOperand(0);
  if (!N0.isMachineOpcode())
    return false;

  switch (N0.getMachineOpcode()) {
  default:
    break;
  case Capstone::ADD:
  case Capstone::ADDI:
  case Capstone::SUB:
  case Capstone::MUL:
  case Capstone::SLLI: {
    // Convert sext.w+add/sub/mul to their W instructions. This will create
    // a new independent instruction. This improves latency.
    unsigned Opc;
    switch (N0.getMachineOpcode()) {
    default:
      llvm_unreachable("Unexpected opcode!");
    case Capstone::ADD:  Opc = Capstone::ADDW;  break;
    case Capstone::ADDI: Opc = Capstone::ADDIW; break;
    case Capstone::SUB:  Opc = Capstone::SUBW;  break;
    case Capstone::MUL:  Opc = Capstone::MULW;  break;
    case Capstone::SLLI: Opc = Capstone::SLLIW; break;
    }

    SDValue N00 = N0.getOperand(0);
    SDValue N01 = N0.getOperand(1);

    // Shift amount needs to be uimm5.
    if (N0.getMachineOpcode() == Capstone::SLLI &&
        !isUInt<5>(cast<ConstantSDNode>(N01)->getSExtValue()))
      break;

    SDNode *Result =
        CurDAG->getMachineNode(Opc, SDLoc(N), N->getValueType(0),
                               N00, N01);
    ReplaceUses(N, Result);
    return true;
  }
  case Capstone::ADDW:
  case Capstone::ADDIW:
  case Capstone::SUBW:
  case Capstone::MULW:
  case Capstone::SLLIW:
  case Capstone::PACKW:
  case Capstone::TH_MULAW:
  case Capstone::TH_MULAH:
  case Capstone::TH_MULSW:
  case Capstone::TH_MULSH:
    if (N0.getValueType() == MVT::i32)
      break;

    // Result is already sign extended just remove the sext.w.
    // NOTE: We only handle the nodes that are selected with hasAllWUsers.
    ReplaceUses(N, N0.getNode());
    return true;
  }

  return false;
}

static bool usesAllOnesMask(SDValue MaskOp) {
  const auto IsVMSet = [](unsigned Opc) {
    return Opc == Capstone::PseudoVMSET_M_B1 || Opc == Capstone::PseudoVMSET_M_B16 ||
           Opc == Capstone::PseudoVMSET_M_B2 || Opc == Capstone::PseudoVMSET_M_B32 ||
           Opc == Capstone::PseudoVMSET_M_B4 || Opc == Capstone::PseudoVMSET_M_B64 ||
           Opc == Capstone::PseudoVMSET_M_B8;
  };

  // TODO: Check that the VMSET is the expected bitwidth? The pseudo has
  // undefined behaviour if it's the wrong bitwidth, so we could choose to
  // assume that it's all-ones? Same applies to its VL.
  return MaskOp->isMachineOpcode() && IsVMSet(MaskOp.getMachineOpcode());
}

static bool isImplicitDef(SDValue V) {
  if (!V.isMachineOpcode())
    return false;
  if (V.getMachineOpcode() == TargetOpcode::REG_SEQUENCE) {
    for (unsigned I = 1; I < V.getNumOperands(); I += 2)
      if (!isImplicitDef(V.getOperand(I)))
        return false;
    return true;
  }
  return V.getMachineOpcode() == TargetOpcode::IMPLICIT_DEF;
}

// Optimize masked RVV pseudo instructions with a known all-ones mask to their
// corresponding "unmasked" pseudo versions.
bool CapstoneDAGToDAGISel::doPeepholeMaskedRVV(MachineSDNode *N) {
  const Capstone::CapstoneMaskedPseudoInfo *I =
      Capstone::getMaskedPseudoInfo(N->getMachineOpcode());
  if (!I)
    return false;

  unsigned MaskOpIdx = I->MaskOpIdx;
  if (!usesAllOnesMask(N->getOperand(MaskOpIdx)))
    return false;

  // There are two classes of pseudos in the table - compares and
  // everything else.  See the comment on CapstoneMaskedPseudo for details.
  const unsigned Opc = I->UnmaskedPseudo;
  const MCInstrDesc &MCID = TII->get(Opc);
  const bool HasPassthru = CapstoneII::isFirstDefTiedToFirstUse(MCID);

  const MCInstrDesc &MaskedMCID = TII->get(N->getMachineOpcode());
  const bool MaskedHasPassthru = CapstoneII::isFirstDefTiedToFirstUse(MaskedMCID);

  assert((CapstoneII::hasVecPolicyOp(MaskedMCID.TSFlags) ||
          !CapstoneII::hasVecPolicyOp(MCID.TSFlags)) &&
         "Unmasked pseudo has policy but masked pseudo doesn't?");
  assert(CapstoneII::hasVecPolicyOp(MCID.TSFlags) == HasPassthru &&
         "Unexpected pseudo structure");
  assert(!(HasPassthru && !MaskedHasPassthru) &&
         "Unmasked pseudo has passthru but masked pseudo doesn't?");

  SmallVector<SDValue, 8> Ops;
  // Skip the passthru operand at index 0 if the unmasked don't have one.
  bool ShouldSkip = !HasPassthru && MaskedHasPassthru;
  bool DropPolicy = !CapstoneII::hasVecPolicyOp(MCID.TSFlags) &&
                    CapstoneII::hasVecPolicyOp(MaskedMCID.TSFlags);
  bool HasChainOp =
      N->getOperand(N->getNumOperands() - 1).getValueType() == MVT::Other;
  unsigned LastOpNum = N->getNumOperands() - 1 - HasChainOp;
  for (unsigned I = ShouldSkip, E = N->getNumOperands(); I != E; I++) {
    // Skip the mask
    SDValue Op = N->getOperand(I);
    if (I == MaskOpIdx)
      continue;
    if (DropPolicy && I == LastOpNum)
      continue;
    Ops.push_back(Op);
  }

  MachineSDNode *Result =
      CurDAG->getMachineNode(Opc, SDLoc(N), N->getVTList(), Ops);

  if (!N->memoperands_empty())
    CurDAG->setNodeMemRefs(Result, N->memoperands());

  Result->setFlags(N->getFlags());
  ReplaceUses(N, Result);

  return true;
}

/// If our passthru is an implicit_def, use noreg instead.  This side
/// steps issues with MachineCSE not being able to CSE expressions with
/// IMPLICIT_DEF operands while preserving the semantic intent. See
/// pr64282 for context. Note that this transform is the last one
/// performed at ISEL DAG to DAG.
bool CapstoneDAGToDAGISel::doPeepholeNoRegPassThru() {
  bool MadeChange = false;
  SelectionDAG::allnodes_iterator Position = CurDAG->allnodes_end();

  while (Position != CurDAG->allnodes_begin()) {
    SDNode *N = &*--Position;
    if (N->use_empty() || !N->isMachineOpcode())
      continue;

    const unsigned Opc = N->getMachineOpcode();
    if (!CapstoneVPseudosTable::getPseudoInfo(Opc) ||
        !CapstoneII::isFirstDefTiedToFirstUse(TII->get(Opc)) ||
        !isImplicitDef(N->getOperand(0)))
      continue;

    SmallVector<SDValue> Ops;
    Ops.push_back(CurDAG->getRegister(Capstone::NoRegister, N->getValueType(0)));
    for (unsigned I = 1, E = N->getNumOperands(); I != E; I++) {
      SDValue Op = N->getOperand(I);
      Ops.push_back(Op);
    }

    MachineSDNode *Result =
      CurDAG->getMachineNode(Opc, SDLoc(N), N->getVTList(), Ops);
    Result->setFlags(N->getFlags());
    CurDAG->setNodeMemRefs(Result, cast<MachineSDNode>(N)->memoperands());
    ReplaceUses(N, Result);
    MadeChange = true;
  }
  return MadeChange;
}


// This pass converts a legalized DAG into a Capstone-specific DAG, ready
// for instruction scheduling.
FunctionPass *llvm::createCapstoneISelDag(CapstoneTargetMachine &TM,
                                       CodeGenOptLevel OptLevel) {
  return new CapstoneDAGToDAGISelLegacy(TM, OptLevel);
}

char CapstoneDAGToDAGISelLegacy::ID = 0;

CapstoneDAGToDAGISelLegacy::CapstoneDAGToDAGISelLegacy(CapstoneTargetMachine &TM,
                                                 CodeGenOptLevel OptLevel)
    : SelectionDAGISelLegacy(
          ID, std::make_unique<CapstoneDAGToDAGISel>(TM, OptLevel)) {}

INITIALIZE_PASS(CapstoneDAGToDAGISelLegacy, DEBUG_TYPE, PASS_NAME, false, false)
