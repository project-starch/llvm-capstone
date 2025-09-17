//===-- CapstoneInstrGISel.td - Capstone GISel target pseudos ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
// Capstone GlobalISel target pseudo instruction definitions. This is kept
// separately from the other tablegen files for organizational purposes, but
// share the same infrastructure.
//
//===----------------------------------------------------------------------===//

class CapstoneGenericInstruction : GenericInstruction {
  let Namespace = "Capstone";
}

// Pseudo equivalent to a CapstoneISD::SRAW.
def G_SRAW : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_SRAW, capstone_sraw>;

// Pseudo equivalent to a CapstoneISD::SRLW.
def G_SRLW : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_SRLW, capstone_srlw>;

// Pseudo equivalent to a CapstoneISD::SLLW.
def G_SLLW : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_SLLW, capstone_sllw>;

// Pseudo equivalent to a CapstoneISD::DIVW.
def G_DIVW : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_DIVW, capstone_divw>;

// Pseudo equivalent to a CapstoneISD::DIVUW.
def G_DIVUW : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_DIVUW, capstone_divuw>;

// Pseudo equivalent to a CapstoneISD::REMUW.
def G_REMUW : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_REMUW, capstone_remuw>;

// Pseudo equivalent to a CapstoneISD::RORW.
def G_RORW : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_RORW, capstone_rorw>;

// Pseudo equivalent to a CapstoneISD::ROLW.
def G_ROLW : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_ROLW, capstone_rolw>;

// Pseudo equivalent to a CapstoneISD::CLZW.
def G_CLZW : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_CLZW, capstone_clzw>;

// Pseudo equivalent to a CapstoneISD::CTZW.
def G_CTZW : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_CTZW, capstone_ctzw>;

// Pseudo equivalent to a CapstoneISD::FCVT_W_RV64.
def G_FCVT_W_RV64 : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src, untyped_imm_0:$frm);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_FCVT_W_RV64, capstone_fcvt_w_rv64>;

// Pseudo equivalent to a CapstoneISD::FCVT_WU_RV64.
def G_FCVT_WU_RV64 : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src, untyped_imm_0:$frm);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_FCVT_WU_RV64, capstone_fcvt_wu_rv64>;

// Pseudo equivalent to a CapstoneISD::FCLASS.
def G_FCLASS : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_FCLASS, capstone_fclass>;

// Pseudo equivalent to a CapstoneISD::READ_VLENB.
def G_READ_VLENB : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_READ_VLENB, capstone_read_vlenb>;

// Pseudo equivalent to a CapstoneISD::VMCLR_VL
def G_VMCLR_VL : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$vl);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_VMCLR_VL, capstone_vmclr_vl>;

// Pseudo equivalent to a CapstoneISD::VMSET_VL
def G_VMSET_VL : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$vl);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_VMSET_VL, capstone_vmset_vl>;

def G_SPLAT_VECTOR_SPLIT_I64_VL : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$passthru, type1:$hi, type1:$lo, type2:$vl);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_SPLAT_VECTOR_SPLIT_I64_VL, capstone_splat_vector_split_i64_vl>;

// Pseudo equivalent to a CapstoneISD::VSLIDEDOWN_VL
def G_VSLIDEDOWN_VL : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$merge, type0:$vec, type1:$idx, type2:$mask,
                       type1:$vl, type1:$policy);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_VSLIDEDOWN_VL, capstone_slidedown_vl>;

// Pseudo equivalent to a CapstoneISD::VMV_V_V_VL
def G_VMV_V_V_VL : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$passthru, type0:$vec, type1:$vl);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_VMV_V_V_VL, capstone_vmv_v_v_vl>;

// Pseudo equivalent to a CapstoneISD::VSLIDEUP_VL
def G_VSLIDEUP_VL : CapstoneGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$merge, type0:$vec, type1:$idx, type2:$mask,
                       type3:$vl, type4:$policy);
  let hasSideEffects = false;
}
def : GINodeEquiv<G_VSLIDEUP_VL, capstone_slideup_vl>;

