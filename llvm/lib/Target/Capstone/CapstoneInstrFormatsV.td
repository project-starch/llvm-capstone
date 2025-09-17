//===-- CapstoneInstrFormatsV.td - Capstone V Instruction Formats -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file describes the Capstone V extension instruction formats.
//
//===----------------------------------------------------------------------===//

class CapstoneVFormat<bits<3> val> {
  bits<3> Value = val;
}
def OPIVV : CapstoneVFormat<0b000>;
def OPFVV : CapstoneVFormat<0b001>;
def OPMVV : CapstoneVFormat<0b010>;
def OPIVI : CapstoneVFormat<0b011>;
def OPIVX : CapstoneVFormat<0b100>;
def OPFVF : CapstoneVFormat<0b101>;
def OPMVX : CapstoneVFormat<0b110>;
def OPCFG : CapstoneVFormat<0b111>;

class CapstoneMOP<bits<2> val> {
  bits<2> Value = val;
}
def MOPLDUnitStride   : CapstoneMOP<0b00>;
def MOPLDIndexedUnord : CapstoneMOP<0b01>;
def MOPLDStrided      : CapstoneMOP<0b10>;
def MOPLDIndexedOrder : CapstoneMOP<0b11>;

def MOPSTUnitStride   : CapstoneMOP<0b00>;
def MOPSTIndexedUnord : CapstoneMOP<0b01>;
def MOPSTStrided      : CapstoneMOP<0b10>;
def MOPSTIndexedOrder : CapstoneMOP<0b11>;

class CapstoneLSUMOP<bits<5> val> {
  bits<5> Value = val;
}
def LUMOPUnitStride  : CapstoneLSUMOP<0b00000>;
def LUMOPUnitStrideMask : CapstoneLSUMOP<0b01011>;
def LUMOPUnitStrideWholeReg : CapstoneLSUMOP<0b01000>;
def LUMOPUnitStrideFF: CapstoneLSUMOP<0b10000>;
def SUMOPUnitStride  : CapstoneLSUMOP<0b00000>;
def SUMOPUnitStrideMask : CapstoneLSUMOP<0b01011>;
def SUMOPUnitStrideWholeReg : CapstoneLSUMOP<0b01000>;

class CapstoneWidth<bits<4> val> {
  bits<4> Value = val;
}
def LSWidth8     : CapstoneWidth<0b0000>;
def LSWidth16    : CapstoneWidth<0b0101>;
def LSWidth32    : CapstoneWidth<0b0110>;
def LSWidth64    : CapstoneWidth<0b0111>;

class RVInstSetiVLi<dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatI> {
  bits<5> uimm;
  bits<5> rd;
  bits<10> vtypei;

  let Inst{31} = 1;
  let Inst{30} = 1;
  let Inst{29-20} = vtypei{9-0};
  let Inst{19-15} = uimm;
  let Inst{14-12} = OPCFG.Value;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_OP_V.Value;

  let Defs = [VL, VTYPE];
}

class RVInstSetVLi<dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatI> {
  bits<5> rs1;
  bits<5> rd;
  bits<11> vtypei;

  let Inst{31} = 0;
  let Inst{30-20} = vtypei;
  let Inst{19-15} = rs1;
  let Inst{14-12} = OPCFG.Value;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_OP_V.Value;

  let Defs = [VL, VTYPE];
}

class RVInstSetVL<dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> rd;

  let Inst{31} = 1;
  let Inst{30-25} = 0b000000;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = OPCFG.Value;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_OP_V.Value;

  let Defs = [VL, VTYPE];
}

class RVInstVV<bits<6> funct6, CapstoneVFormat opv, dag outs, dag ins,
               string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> vs1;
  bits<5> vd;
  bit vm;

  let Inst{31-26} = funct6;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = vs1;
  let Inst{14-12} = opv.Value;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_OP_V.Value;

  let Uses = [VL, VTYPE];
  let RVVConstraint = VMConstraint;
}

class RVInstVX<bits<6> funct6, CapstoneVFormat opv, dag outs, dag ins,
                string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> rs1;
  bits<5> vd;
  bit vm;

  let Inst{31-26} = funct6;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = opv.Value;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_OP_V.Value;

  let Uses = [VL, VTYPE];
  let RVVConstraint = VMConstraint;
}

class RVInstV2<bits<6> funct6, bits<5> vs2, CapstoneVFormat opv, dag outs, dag ins,
                string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs1;
  bits<5> vd;
  bit vm;

  let Inst{31-26} = funct6;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = opv.Value;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_OP_V.Value;

  let Uses = [VL, VTYPE];
  let RVVConstraint = VMConstraint;
}

class RVInstIVI<bits<6> funct6, dag outs, dag ins, string opcodestr,
                string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> imm;
  bits<5> vd;
  bit vm;

  let Inst{31-26} = funct6;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = imm;
  let Inst{14-12} = OPIVI.Value;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_OP_V.Value;

  let Uses = [VL, VTYPE];
  let RVVConstraint = VMConstraint;
}

class RVInstV<bits<6> funct6, bits<5> vs1, CapstoneVFormat opv, dag outs,
              dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> vd;
  bit vm;

  let Inst{31-26} = funct6;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = vs1;
  let Inst{14-12} = opv.Value;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_OP_V.Value;

  let Uses = [VL, VTYPE];
  let RVVConstraint = VMConstraint;
}

class RVInstVLU<bits<3> nf, bit mew, CapstoneLSUMOP lumop,
                bits<3> width, dag outs, dag ins, string opcodestr,
                string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs1;
  bits<5> vd;
  bit vm;

  let Inst{31-29} = nf;
  let Inst{28} = mew;
  let Inst{27-26} = MOPLDUnitStride.Value;
  let Inst{25} = vm;
  let Inst{24-20} = lumop.Value;
  let Inst{19-15} = rs1;
  let Inst{14-12} = width;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_LOAD_FP.Value;

  let Uses = [VL, VTYPE];
  let RVVConstraint = VMConstraint;
}

class RVInstVLS<bits<3> nf, bit mew, bits<3> width,
                dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> vd;
  bit vm;

  let Inst{31-29} = nf;
  let Inst{28} = mew;
  let Inst{27-26} = MOPLDStrided.Value;
  let Inst{25} = vm;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = width;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_LOAD_FP.Value;

  let Uses = [VL, VTYPE];
  let RVVConstraint = VMConstraint;
}

class RVInstVLX<bits<3> nf, bit mew, CapstoneMOP mop, bits<3> width,
                dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> rs1;
  bits<5> vd;
  bit vm;

  let Inst{31-29} = nf;
  let Inst{28} = mew;
  let Inst{27-26} = mop.Value;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = width;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_LOAD_FP.Value;

  let Uses = [VL, VTYPE];
  let RVVConstraint = VMConstraint;
}

class RVInstVSU<bits<3> nf, bit mew, CapstoneLSUMOP sumop,
                bits<3> width, dag outs, dag ins, string opcodestr,
                string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs1;
  bits<5> vs3;
  bit vm;

  let Inst{31-29} = nf;
  let Inst{28} = mew;
  let Inst{27-26} = MOPSTUnitStride.Value;
  let Inst{25} = vm;
  let Inst{24-20} = sumop.Value;
  let Inst{19-15} = rs1;
  let Inst{14-12} = width;
  let Inst{11-7} = vs3;
  let Inst{6-0} = OPC_STORE_FP.Value;

  let Uses = [VL, VTYPE];
}

class RVInstVSS<bits<3> nf, bit mew, bits<3> width,
                dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> vs3;
  bit vm;

  let Inst{31-29} = nf;
  let Inst{28} = mew;
  let Inst{27-26} = MOPSTStrided.Value;
  let Inst{25} = vm;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = width;
  let Inst{11-7} = vs3;
  let Inst{6-0} = OPC_STORE_FP.Value;

  let Uses = [VL, VTYPE];
}

class RVInstVSX<bits<3> nf, bit mew, CapstoneMOP mop, bits<3> width,
                dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  bits<5> vs2;
  bits<5> rs1;
  bits<5> vs3;
  bit vm;

  let Inst{31-29} = nf;
  let Inst{28} = mew;
  let Inst{27-26} = mop.Value;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = width;
  let Inst{11-7} = vs3;
  let Inst{6-0} = OPC_STORE_FP.Value;

  let Uses = [VL, VTYPE];
}
