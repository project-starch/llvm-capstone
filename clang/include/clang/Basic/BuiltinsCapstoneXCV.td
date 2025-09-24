//==- BuiltinsCapstoneXCV.td - Capstone CORE-V Builtin database    ----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CORE-V-specific builtin function database.  Users of
// this file must define the BUILTIN macro to make use of this information.
//
//===----------------------------------------------------------------------===//

class CapstoneXCVBuiltin<string prototype, string features = ""> : TargetBuiltin {
  let Spellings = ["__builtin_capstone_cv_" # NAME];
  let Prototype = prototype;
  let Features = features;
}

let Attributes = [NoThrow, Const] in {
//===----------------------------------------------------------------------===//
// XCValu extension.
//===----------------------------------------------------------------------===//
def alu_sle   : CapstoneXCVBuiltin<"int(int, int)", "xcvalu">;
def alu_sleu  : CapstoneXCVBuiltin<"int(unsigned int, unsigned int)", "xcvalu">;
def alu_exths : CapstoneXCVBuiltin<"int(int)", "xcvalu">;
def alu_exthz : CapstoneXCVBuiltin<"unsigned int(unsigned int)", "xcvalu">;
def alu_extbs : CapstoneXCVBuiltin<"int(int)", "xcvalu">;
def alu_extbz : CapstoneXCVBuiltin<"unsigned int(unsigned int)", "xcvalu">;

def alu_clip   : CapstoneXCVBuiltin<"int(int, int)", "xcvalu">;
def alu_clipu  : CapstoneXCVBuiltin<"unsigned int(unsigned int, unsigned int)", "xcvalu">;
def alu_addN   : CapstoneXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_adduN  : CapstoneXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
def alu_addRN  : CapstoneXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_adduRN : CapstoneXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
def alu_subN   : CapstoneXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_subuN  : CapstoneXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
def alu_subRN  : CapstoneXCVBuiltin<"int(int, int, unsigned int)", "xcvalu">;
def alu_subuRN : CapstoneXCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)", "xcvalu">;
} // Attributes = [NoThrow, Const]
