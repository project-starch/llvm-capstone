//===-- CapstoneCallingConv.h - Capstone Custom CC Routines ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the custom routines for the Capstone Calling Convention.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/CapstoneBaseInfo.h"
#include "llvm/CodeGen/CallingConvLower.h"

namespace llvm {

/// CapstoneCCAssignFn - This target-specific function extends the default
/// CCValAssign with additional information used to lower Capstone calling
/// conventions.
typedef bool CapstoneCCAssignFn(unsigned ValNo, MVT ValVT, MVT LocVT,
                             CCValAssign::LocInfo LocInfo,
                             ISD::ArgFlagsTy ArgFlags, CCState &State,
                             bool IsRet, Type *OrigTy);

bool CC_Capstone(unsigned ValNo, MVT ValVT, MVT LocVT,
              CCValAssign::LocInfo LocInfo, ISD::ArgFlagsTy ArgFlags,
              CCState &State, bool IsRet, Type *OrigTy);

bool CC_Capstone_FastCC(unsigned ValNo, MVT ValVT, MVT LocVT,
                     CCValAssign::LocInfo LocInfo, ISD::ArgFlagsTy ArgFlags,
                     CCState &State, bool IsRet, Type *OrigTy);

bool CC_Capstone_GHC(unsigned ValNo, MVT ValVT, MVT LocVT,
                  CCValAssign::LocInfo LocInfo, ISD::ArgFlagsTy ArgFlags,
                  Type *OrigTy, CCState &State);

namespace Capstone {

ArrayRef<MCPhysReg> getArgGPRs(const CapstoneABI::ABI ABI);

} // end namespace Capstone

} // end namespace llvm
