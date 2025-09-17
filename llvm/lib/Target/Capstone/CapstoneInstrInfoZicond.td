//===-- CapstoneInstrInfoZicond.td ----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the Capstone instructions from the standard Integer
// Conditional operations extension (Zicond).
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Capstone specific DAG Nodes.
//===----------------------------------------------------------------------===//

// Branchless select operations, matching the semantics of the instructions
// defined in Zicond or XVentanaCondOps
def capstone_czero_eqz : RVSDNode<"CZERO_EQZ", SDTIntBinOp>; // vt.maskc for XVentanaCondOps.
def capstone_czero_nez : RVSDNode<"CZERO_NEZ", SDTIntBinOp>; // vt.maskcn for XVentanaCondOps.

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZicond] in {
def CZERO_EQZ : ALU_rr<0b0000111, 0b101, "czero.eqz">,
                Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def CZERO_NEZ : ALU_rr<0b0000111, 0b111, "czero.nez">,
                Sched<[WriteIALU, ReadIALU, ReadIALU]>;
} // Predicates = [HasStdExtZicond]

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZicond] in {
def : Pat<(XLenVT (capstone_czero_eqz GPR:$rs1, GPR:$rc)),
          (CZERO_EQZ GPR:$rs1, GPR:$rc)>;
def : Pat<(XLenVT (capstone_czero_nez GPR:$rs1, GPR:$rc)),
          (CZERO_NEZ GPR:$rs1, GPR:$rc)>;

def : Pat<(XLenVT (capstone_czero_eqz GPR:$rs1, (capstone_setne (XLenVT GPR:$rc)))),
          (CZERO_EQZ GPR:$rs1, GPR:$rc)>;
def : Pat<(XLenVT (capstone_czero_eqz GPR:$rs1, (capstone_seteq (XLenVT GPR:$rc)))),
          (CZERO_NEZ GPR:$rs1, GPR:$rc)>;
def : Pat<(XLenVT (capstone_czero_nez GPR:$rs1, (capstone_setne (XLenVT GPR:$rc)))),
          (CZERO_NEZ GPR:$rs1, GPR:$rc)>;
def : Pat<(XLenVT (capstone_czero_nez GPR:$rs1, (capstone_seteq (XLenVT GPR:$rc)))),
          (CZERO_EQZ GPR:$rs1, GPR:$rc)>;
} // Predicates = [HasStdExtZicond]
