//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Capstone_CapstoneSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_Capstone_CapstoneSELECTIONDAGINFO_H

#include "llvm/CodeGen/SDNodeInfo.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "CapstoneGenSDNodeInfo.inc"

namespace llvm {

namespace CapstoneISD {
// CapstoneISD Node TSFlags
enum : llvm::SDNodeTSFlags {
  HasPassthruOpMask = 1 << 0,
  HasMaskOpMask = 1 << 1,
};
} // namespace CapstoneISD

class CapstoneSelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  CapstoneSelectionDAGInfo();

  ~CapstoneSelectionDAGInfo() override;

  void verifyTargetNode(const SelectionDAG &DAG,
                        const SDNode *N) const override;

  SDValue EmitTargetCodeForMemset(SelectionDAG &DAG, const SDLoc &dl,
                                  SDValue Chain, SDValue Dst, SDValue Src,
                                  SDValue Size, Align Alignment,
                                  bool isVolatile, bool AlwaysInline,
                                  MachinePointerInfo DstPtrInfo) const override;

  bool hasPassthruOp(unsigned Opcode) const {
    return GenNodeInfo.getDesc(Opcode).TSFlags & CapstoneISD::HasPassthruOpMask;
  }

  bool hasMaskOp(unsigned Opcode) const {
    return GenNodeInfo.getDesc(Opcode).TSFlags & CapstoneISD::HasMaskOpMask;
  }

  unsigned getMAccOpcode(unsigned MulOpcode) const {
    switch (static_cast<CapstoneISD::GenNodeType>(MulOpcode)) {
    default:
      llvm_unreachable("Unexpected opcode");
    case CapstoneISD::VWMUL_VL:
      return CapstoneISD::VWMACC_VL;
    case CapstoneISD::VWMULU_VL:
      return CapstoneISD::VWMACCU_VL;
    case CapstoneISD::VWMULSU_VL:
      return CapstoneISD::VWMACCSU_VL;
    }
  }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_Capstone_CapstoneSELECTIONDAGINFO_H
