//===-- CapstoneAsmPrinter.cpp - Capstone LLVM assembly writer -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to the Capstone assembly language.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/CapstoneBaseInfo.h"
#include "MCTargetDesc/CapstoneInstPrinter.h"
#include "MCTargetDesc/CapstoneMCAsmInfo.h"
#include "MCTargetDesc/CapstoneMatInt.h"
#include "MCTargetDesc/CapstoneTargetStreamer.h"
#include "Capstone.h"
#include "CapstoneConstantPoolValue.h"
#include "CapstoneMachineFunctionInfo.h"
#include "CapstoneRegisterInfo.h"
#include "TargetInfo/CapstoneTargetInfo.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/CapstoneISAInfo.h"
#include "llvm/Transforms/Instrumentation/HWAddressSanitizer.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

STATISTIC(CapstoneNumInstrsCompressed,
          "Number of Capstone Compressed instructions emitted");

namespace llvm {
extern const SubtargetFeatureKV CapstoneFeatureKV[Capstone::NumSubtargetFeatures];
} // namespace llvm

namespace {
class CapstoneAsmPrinter : public AsmPrinter {
public:
  static char ID;

private:
  const CapstoneSubtarget *STI;

public:
  explicit CapstoneAsmPrinter(TargetMachine &TM,
                           std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer), ID) {}

  StringRef getPassName() const override { return "Capstone Assembly Printer"; }

  void LowerSTACKMAP(MCStreamer &OutStreamer, StackMaps &SM,
                     const MachineInstr &MI);

  void LowerPATCHPOINT(MCStreamer &OutStreamer, StackMaps &SM,
                       const MachineInstr &MI);

  void LowerSTATEPOINT(MCStreamer &OutStreamer, StackMaps &SM,
                       const MachineInstr &MI);

  bool runOnMachineFunction(MachineFunction &MF) override;

  void emitInstruction(const MachineInstr *MI) override;

  void emitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) override;

  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &OS) override;
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                             const char *ExtraCode, raw_ostream &OS) override;

  // Returns whether Inst is compressed.
  bool EmitToStreamer(MCStreamer &S, const MCInst &Inst,
                      const MCSubtargetInfo &SubtargetInfo);
  bool EmitToStreamer(MCStreamer &S, const MCInst &Inst) {
    return EmitToStreamer(S, Inst, *STI);
  }

  bool lowerPseudoInstExpansion(const MachineInstr *MI, MCInst &Inst);

  typedef std::tuple<unsigned, uint32_t> HwasanMemaccessTuple;
  std::map<HwasanMemaccessTuple, MCSymbol *> HwasanMemaccessSymbols;
  void LowerHWASAN_CHECK_MEMACCESS(const MachineInstr &MI);
  void LowerKCFI_CHECK(const MachineInstr &MI);
  void EmitHwasanMemaccessSymbols(Module &M);

  // Wrapper needed for tblgenned pseudo lowering.
  bool lowerOperand(const MachineOperand &MO, MCOperand &MCOp) const;

  void emitStartOfAsmFile(Module &M) override;
  void emitEndOfAsmFile(Module &M) override;

  void emitFunctionEntryLabel() override;
  bool emitDirectiveOptionArch();

  void emitNoteGnuProperty(const Module &M);

private:
  void emitAttributes(const MCSubtargetInfo &SubtargetInfo);

  void emitNTLHint(const MachineInstr *MI);

  // XRay Support
  void LowerPATCHABLE_FUNCTION_ENTER(const MachineInstr *MI);
  void LowerPATCHABLE_FUNCTION_EXIT(const MachineInstr *MI);
  void LowerPATCHABLE_TAIL_CALL(const MachineInstr *MI);
  void emitSled(const MachineInstr *MI, SledKind Kind);

  void lowerToMCInst(const MachineInstr *MI, MCInst &OutMI);
};
}

void CapstoneAsmPrinter::LowerSTACKMAP(MCStreamer &OutStreamer, StackMaps &SM,
                                    const MachineInstr &MI) {
  unsigned NOPBytes = STI->hasStdExtZca() ? 2 : 4;
  unsigned NumNOPBytes = StackMapOpers(&MI).getNumPatchBytes();

  auto &Ctx = OutStreamer.getContext();
  MCSymbol *MILabel = Ctx.createTempSymbol();
  OutStreamer.emitLabel(MILabel);

  SM.recordStackMap(*MILabel, MI);
  assert(NumNOPBytes % NOPBytes == 0 &&
         "Invalid number of NOP bytes requested!");

  // Scan ahead to trim the shadow.
  const MachineBasicBlock &MBB = *MI.getParent();
  MachineBasicBlock::const_iterator MII(MI);
  ++MII;
  while (NumNOPBytes > 0) {
    if (MII == MBB.end() || MII->isCall() ||
        MII->getOpcode() == Capstone::DBG_VALUE ||
        MII->getOpcode() == TargetOpcode::PATCHPOINT ||
        MII->getOpcode() == TargetOpcode::STACKMAP)
      break;
    ++MII;
    NumNOPBytes -= 4;
  }

  // Emit nops.
  emitNops(NumNOPBytes / NOPBytes);
}

// Lower a patchpoint of the form:
// [<def>], <id>, <numBytes>, <target>, <numArgs>
void CapstoneAsmPrinter::LowerPATCHPOINT(MCStreamer &OutStreamer, StackMaps &SM,
                                      const MachineInstr &MI) {
  unsigned NOPBytes = STI->hasStdExtZca() ? 2 : 4;

  auto &Ctx = OutStreamer.getContext();
  MCSymbol *MILabel = Ctx.createTempSymbol();
  OutStreamer.emitLabel(MILabel);
  SM.recordPatchPoint(*MILabel, MI);

  PatchPointOpers Opers(&MI);

  const MachineOperand &CalleeMO = Opers.getCallTarget();
  unsigned EncodedBytes = 0;

  if (CalleeMO.isImm()) {
    uint64_t CallTarget = CalleeMO.getImm();
    if (CallTarget) {
      assert((CallTarget & 0xFFFF'FFFF'FFFF) == CallTarget &&
             "High 16 bits of call target should be zero.");
      // Materialize the jump address:
      SmallVector<MCInst, 8> Seq;
      CapstoneMatInt::generateMCInstSeq(CallTarget, *STI, Capstone::X1, Seq);
      for (MCInst &Inst : Seq) {
        bool Compressed = EmitToStreamer(OutStreamer, Inst);
        EncodedBytes += Compressed ? 2 : 4;
      }
      bool Compressed = EmitToStreamer(OutStreamer, MCInstBuilder(Capstone::JALR)
                                                        .addReg(Capstone::X1)
                                                        .addReg(Capstone::X1)
                                                        .addImm(0));
      EncodedBytes += Compressed ? 2 : 4;
    }
  } else if (CalleeMO.isGlobal()) {
    MCOperand CallTargetMCOp;
    lowerOperand(CalleeMO, CallTargetMCOp);
    EmitToStreamer(OutStreamer,
                   MCInstBuilder(Capstone::PseudoCALL).addOperand(CallTargetMCOp));
    EncodedBytes += 8;
  }

  // Emit padding.
  unsigned NumBytes = Opers.getNumPatchBytes();
  assert(NumBytes >= EncodedBytes &&
         "Patchpoint can't request size less than the length of a call.");
  assert((NumBytes - EncodedBytes) % NOPBytes == 0 &&
         "Invalid number of NOP bytes requested!");
  emitNops((NumBytes - EncodedBytes) / NOPBytes);
}

void CapstoneAsmPrinter::LowerSTATEPOINT(MCStreamer &OutStreamer, StackMaps &SM,
                                      const MachineInstr &MI) {
  unsigned NOPBytes = STI->hasStdExtZca() ? 2 : 4;

  StatepointOpers SOpers(&MI);
  if (unsigned PatchBytes = SOpers.getNumPatchBytes()) {
    assert(PatchBytes % NOPBytes == 0 &&
           "Invalid number of NOP bytes requested!");
    emitNops(PatchBytes / NOPBytes);
  } else {
    // Lower call target and choose correct opcode
    const MachineOperand &CallTarget = SOpers.getCallTarget();
    MCOperand CallTargetMCOp;
    switch (CallTarget.getType()) {
    case MachineOperand::MO_GlobalAddress:
    case MachineOperand::MO_ExternalSymbol:
      lowerOperand(CallTarget, CallTargetMCOp);
      EmitToStreamer(
          OutStreamer,
          MCInstBuilder(Capstone::PseudoCALL).addOperand(CallTargetMCOp));
      break;
    case MachineOperand::MO_Immediate:
      CallTargetMCOp = MCOperand::createImm(CallTarget.getImm());
      EmitToStreamer(OutStreamer, MCInstBuilder(Capstone::JAL)
                                      .addReg(Capstone::X1)
                                      .addOperand(CallTargetMCOp));
      break;
    case MachineOperand::MO_Register:
      CallTargetMCOp = MCOperand::createReg(CallTarget.getReg());
      EmitToStreamer(OutStreamer, MCInstBuilder(Capstone::JALR)
                                      .addReg(Capstone::X1)
                                      .addOperand(CallTargetMCOp)
                                      .addImm(0));
      break;
    default:
      llvm_unreachable("Unsupported operand type in statepoint call target");
      break;
    }
  }

  auto &Ctx = OutStreamer.getContext();
  MCSymbol *MILabel = Ctx.createTempSymbol();
  OutStreamer.emitLabel(MILabel);
  SM.recordStatepoint(*MILabel, MI);
}

bool CapstoneAsmPrinter::EmitToStreamer(MCStreamer &S, const MCInst &Inst,
                                     const MCSubtargetInfo &SubtargetInfo) {
  MCInst CInst;
  bool Res = CapstoneRVC::compress(CInst, Inst, SubtargetInfo);
  if (Res)
    ++CapstoneNumInstrsCompressed;
  S.emitInstruction(Res ? CInst : Inst, SubtargetInfo);
  return Res;
}

// Simple pseudo-instructions have their lowering (with expansion to real
// instructions) auto-generated.
#include "CapstoneGenMCPseudoLowering.inc"

// If the target supports Zihintntl and the instruction has a nontemporal
// MachineMemOperand, emit an NTLH hint instruction before it.
void CapstoneAsmPrinter::emitNTLHint(const MachineInstr *MI) {
  if (!STI->hasStdExtZihintntl())
    return;

  if (MI->memoperands_empty())
    return;

  MachineMemOperand *MMO = *(MI->memoperands_begin());
  if (!MMO->isNonTemporal())
    return;

  unsigned NontemporalMode = 0;
  if (MMO->getFlags() & MONontemporalBit0)
    NontemporalMode += 0b1;
  if (MMO->getFlags() & MONontemporalBit1)
    NontemporalMode += 0b10;

  MCInst Hint;
  if (STI->hasStdExtZca())
    Hint.setOpcode(Capstone::C_ADD);
  else
    Hint.setOpcode(Capstone::ADD);

  Hint.addOperand(MCOperand::createReg(Capstone::X0));
  Hint.addOperand(MCOperand::createReg(Capstone::X0));
  Hint.addOperand(MCOperand::createReg(Capstone::X2 + NontemporalMode));

  EmitToStreamer(*OutStreamer, Hint);
}

void CapstoneAsmPrinter::emitInstruction(const MachineInstr *MI) {
  Capstone_MC::verifyInstructionPredicates(MI->getOpcode(), STI->getFeatureBits());

  emitNTLHint(MI);

  // Do any auto-generated pseudo lowerings.
  if (MCInst OutInst; lowerPseudoInstExpansion(MI, OutInst)) {
    EmitToStreamer(*OutStreamer, OutInst);
    return;
  }

  switch (MI->getOpcode()) {
  case Capstone::HWASAN_CHECK_MEMACCESS_SHORTGRANULES:
    LowerHWASAN_CHECK_MEMACCESS(*MI);
    return;
  case Capstone::KCFI_CHECK:
    LowerKCFI_CHECK(*MI);
    return;
  case TargetOpcode::STACKMAP:
    return LowerSTACKMAP(*OutStreamer, SM, *MI);
  case TargetOpcode::PATCHPOINT:
    return LowerPATCHPOINT(*OutStreamer, SM, *MI);
  case TargetOpcode::STATEPOINT:
    return LowerSTATEPOINT(*OutStreamer, SM, *MI);
  case TargetOpcode::PATCHABLE_FUNCTION_ENTER: {
    const Function &F = MI->getParent()->getParent()->getFunction();
    if (F.hasFnAttribute("patchable-function-entry")) {
      unsigned Num;
      [[maybe_unused]] bool Result =
          F.getFnAttribute("patchable-function-entry")
              .getValueAsString()
              .getAsInteger(10, Num);
      assert(!Result && "Enforced by the verifier");
      emitNops(Num);
      return;
    }
    LowerPATCHABLE_FUNCTION_ENTER(MI);
    return;
  }
  case TargetOpcode::PATCHABLE_FUNCTION_EXIT:
    LowerPATCHABLE_FUNCTION_EXIT(MI);
    return;
  case TargetOpcode::PATCHABLE_TAIL_CALL:
    LowerPATCHABLE_TAIL_CALL(MI);
    return;
  }

  MCInst OutInst;
  lowerToMCInst(MI, OutInst);
  EmitToStreamer(*OutStreamer, OutInst);
}

bool CapstoneAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                      const char *ExtraCode, raw_ostream &OS) {
  // First try the generic code, which knows about modifiers like 'c' and 'n'.
  if (!AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, OS))
    return false;

  const MachineOperand &MO = MI->getOperand(OpNo);
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0)
      return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default:
      return true; // Unknown modifier.
    case 'z':      // Print zero register if zero, regular printing otherwise.
      if (MO.isImm() && MO.getImm() == 0) {
        OS << CapstoneInstPrinter::getRegisterName(Capstone::X0);
        return false;
      }
      break;
    case 'i': // Literal 'i' if operand is not a register.
      if (!MO.isReg())
        OS << 'i';
      return false;
    case 'N': // Print the register encoding as an integer (0-31)
      if (!MO.isReg())
        return true;

      const CapstoneRegisterInfo *TRI = STI->getRegisterInfo();
      OS << TRI->getEncodingValue(MO.getReg());
      return false;
    }
  }

  switch (MO.getType()) {
  case MachineOperand::MO_Immediate:
    OS << MO.getImm();
    return false;
  case MachineOperand::MO_Register:
    OS << CapstoneInstPrinter::getRegisterName(MO.getReg());
    return false;
  case MachineOperand::MO_GlobalAddress:
    PrintSymbolOperand(MO, OS);
    return false;
  case MachineOperand::MO_BlockAddress: {
    MCSymbol *Sym = GetBlockAddressSymbol(MO.getBlockAddress());
    Sym->print(OS, MAI);
    return false;
  }
  default:
    break;
  }

  return true;
}

bool CapstoneAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                            unsigned OpNo,
                                            const char *ExtraCode,
                                            raw_ostream &OS) {
  if (ExtraCode)
    return AsmPrinter::PrintAsmMemoryOperand(MI, OpNo, ExtraCode, OS);

  const MachineOperand &AddrReg = MI->getOperand(OpNo);
  assert(MI->getNumOperands() > OpNo + 1 && "Expected additional operand");
  const MachineOperand &Offset = MI->getOperand(OpNo + 1);
  // All memory operands should have a register and an immediate operand (see
  // CapstoneDAGToDAGISel::SelectInlineAsmMemoryOperand).
  if (!AddrReg.isReg())
    return true;
  if (!Offset.isImm() && !Offset.isGlobal() && !Offset.isBlockAddress() &&
      !Offset.isMCSymbol())
    return true;

  MCOperand MCO;
  if (!lowerOperand(Offset, MCO))
    return true;

  if (Offset.isImm())
    OS << MCO.getImm();
  else if (Offset.isGlobal() || Offset.isBlockAddress() || Offset.isMCSymbol())
    MAI->printExpr(OS, *MCO.getExpr());

  if (Offset.isMCSymbol())
    MMI->getContext().registerInlineAsmLabel(Offset.getMCSymbol());
  if (Offset.isBlockAddress()) {
    const BlockAddress *BA = Offset.getBlockAddress();
    MCSymbol *Sym = GetBlockAddressSymbol(BA);
    MMI->getContext().registerInlineAsmLabel(Sym);
  }

  OS << "(" << CapstoneInstPrinter::getRegisterName(AddrReg.getReg()) << ")";
  return false;
}

bool CapstoneAsmPrinter::emitDirectiveOptionArch() {
  CapstoneTargetStreamer &RTS =
      static_cast<CapstoneTargetStreamer &>(*OutStreamer->getTargetStreamer());
  SmallVector<CapstoneOptionArchArg> NeedEmitStdOptionArgs;
  const MCSubtargetInfo &MCSTI = *TM.getMCSubtargetInfo();
  for (const auto &Feature : CapstoneFeatureKV) {
    if (STI->hasFeature(Feature.Value) == MCSTI.hasFeature(Feature.Value))
      continue;

    if (!llvm::CapstoneISAInfo::isSupportedExtensionFeature(Feature.Key))
      continue;

    auto Delta = STI->hasFeature(Feature.Value) ? CapstoneOptionArchArgType::Plus
                                                : CapstoneOptionArchArgType::Minus;
    NeedEmitStdOptionArgs.emplace_back(Delta, Feature.Key);
  }
  if (!NeedEmitStdOptionArgs.empty()) {
    RTS.emitDirectiveOptionPush();
    RTS.emitDirectiveOptionArch(NeedEmitStdOptionArgs);
    return true;
  }

  return false;
}

bool CapstoneAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<CapstoneSubtarget>();
  CapstoneTargetStreamer &RTS =
      static_cast<CapstoneTargetStreamer &>(*OutStreamer->getTargetStreamer());

  bool EmittedOptionArch = emitDirectiveOptionArch();

  SetupMachineFunction(MF);
  emitFunctionBody();

  // Emit the XRay table
  emitXRayTable();

  if (EmittedOptionArch)
    RTS.emitDirectiveOptionPop();
  return false;
}

void CapstoneAsmPrinter::LowerPATCHABLE_FUNCTION_ENTER(const MachineInstr *MI) {
  emitSled(MI, SledKind::FUNCTION_ENTER);
}

void CapstoneAsmPrinter::LowerPATCHABLE_FUNCTION_EXIT(const MachineInstr *MI) {
  emitSled(MI, SledKind::FUNCTION_EXIT);
}

void CapstoneAsmPrinter::LowerPATCHABLE_TAIL_CALL(const MachineInstr *MI) {
  emitSled(MI, SledKind::TAIL_CALL);
}

void CapstoneAsmPrinter::emitSled(const MachineInstr *MI, SledKind Kind) {
  // We want to emit the jump instruction and the nops constituting the sled.
  // The format is as follows:
  // .Lxray_sled_N
  //   ALIGN
  //   J .tmpN
  //   21 or 33 C.NOP instructions
  // .tmpN

  // The following variable holds the count of the number of NOPs to be patched
  // in for XRay instrumentation during compilation.
  // Note that RV64 and RV32 each has a sled of 68 and 44 bytes, respectively.
  // Assuming we're using JAL to jump to .tmpN, then we only need
  // (68 - 4)/2 = 32 NOPs for RV64 and (44 - 4)/2 = 20 for RV32. However, there
  // is a chance that we'll use C.JAL instead, so an additional NOP is needed.
  const uint8_t NoopsInSledCount = STI->is64Bit() ? 33 : 21;

  OutStreamer->emitCodeAlignment(Align(4), STI);
  auto CurSled = OutContext.createTempSymbol("xray_sled_", true);
  OutStreamer->emitLabel(CurSled);
  auto Target = OutContext.createTempSymbol();

  const MCExpr *TargetExpr = MCSymbolRefExpr::create(Target, OutContext);

  // Emit "J bytes" instruction, which jumps over the nop sled to the actual
  // start of function.
  EmitToStreamer(
      *OutStreamer,
      MCInstBuilder(Capstone::JAL).addReg(Capstone::X0).addExpr(TargetExpr));

  // Emit NOP instructions
  for (int8_t I = 0; I < NoopsInSledCount; ++I)
    EmitToStreamer(*OutStreamer, MCInstBuilder(Capstone::ADDI)
                                     .addReg(Capstone::X0)
                                     .addReg(Capstone::X0)
                                     .addImm(0));

  OutStreamer->emitLabel(Target);
  recordSled(CurSled, *MI, Kind, 2);
}

void CapstoneAsmPrinter::emitStartOfAsmFile(Module &M) {
  CapstoneTargetStreamer &RTS =
      static_cast<CapstoneTargetStreamer &>(*OutStreamer->getTargetStreamer());
  if (const MDString *ModuleTargetABI =
          dyn_cast_or_null<MDString>(M.getModuleFlag("target-abi")))
    RTS.setTargetABI(CapstoneABI::getTargetABI(ModuleTargetABI->getString()));

  MCSubtargetInfo SubtargetInfo = *TM.getMCSubtargetInfo();

  // Use module flag to update feature bits.
  if (auto *MD = dyn_cast_or_null<MDNode>(M.getModuleFlag("capstone-isa"))) {
    for (auto &ISA : MD->operands()) {
      if (auto *ISAString = dyn_cast_or_null<MDString>(ISA)) {
        auto ParseResult = llvm::CapstoneISAInfo::parseArchString(
            ISAString->getString(), /*EnableExperimentalExtension=*/true,
            /*ExperimentalExtensionVersionCheck=*/true);
        if (!errorToBool(ParseResult.takeError())) {
          auto &ISAInfo = *ParseResult;
          for (const auto &Feature : CapstoneFeatureKV) {
            if (ISAInfo->hasExtension(Feature.Key) &&
                !SubtargetInfo.hasFeature(Feature.Value))
              SubtargetInfo.ToggleFeature(Feature.Key);
          }
        }
      }
    }

    RTS.setFlagsFromFeatures(SubtargetInfo);
  }

  if (TM.getTargetTriple().isOSBinFormatELF())
    emitAttributes(SubtargetInfo);
}

void CapstoneAsmPrinter::emitEndOfAsmFile(Module &M) {
  CapstoneTargetStreamer &RTS =
      static_cast<CapstoneTargetStreamer &>(*OutStreamer->getTargetStreamer());

  if (TM.getTargetTriple().isOSBinFormatELF()) {
    RTS.finishAttributeSection();
    emitNoteGnuProperty(M);
  }
  EmitHwasanMemaccessSymbols(M);
}

void CapstoneAsmPrinter::emitAttributes(const MCSubtargetInfo &SubtargetInfo) {
  CapstoneTargetStreamer &RTS =
      static_cast<CapstoneTargetStreamer &>(*OutStreamer->getTargetStreamer());
  // Use MCSubtargetInfo from TargetMachine. Individual functions may have
  // attributes that differ from other functions in the module and we have no
  // way to know which function is correct.
  RTS.emitTargetAttributes(SubtargetInfo, /*EmitStackAlign*/ true);
}

void CapstoneAsmPrinter::emitFunctionEntryLabel() {
  const auto *RMFI = MF->getInfo<CapstoneMachineFunctionInfo>();
  if (RMFI->isVectorCall()) {
    auto &RTS =
        static_cast<CapstoneTargetStreamer &>(*OutStreamer->getTargetStreamer());
    RTS.emitDirectiveVariantCC(*CurrentFnSym);
  }
  return AsmPrinter::emitFunctionEntryLabel();
}

// Force static initialization.
extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeCapstoneAsmPrinter() {
  RegisterAsmPrinter<CapstoneAsmPrinter> X(getTheCapstone32Target());
  RegisterAsmPrinter<CapstoneAsmPrinter> Y(getTheCapstone64Target());
  RegisterAsmPrinter<CapstoneAsmPrinter> A(getTheCapstone32beTarget());
  RegisterAsmPrinter<CapstoneAsmPrinter> B(getTheCapstone64beTarget());
}

void CapstoneAsmPrinter::LowerHWASAN_CHECK_MEMACCESS(const MachineInstr &MI) {
  Register Reg = MI.getOperand(0).getReg();
  uint32_t AccessInfo = MI.getOperand(1).getImm();
  MCSymbol *&Sym =
      HwasanMemaccessSymbols[HwasanMemaccessTuple(Reg, AccessInfo)];
  if (!Sym) {
    // FIXME: Make this work on non-ELF.
    if (!TM.getTargetTriple().isOSBinFormatELF())
      report_fatal_error("llvm.hwasan.check.memaccess only supported on ELF");

    std::string SymName = "__hwasan_check_x" + utostr(Reg - Capstone::X0) + "_" +
                          utostr(AccessInfo) + "_short";
    Sym = OutContext.getOrCreateSymbol(SymName);
  }
  auto Res = MCSymbolRefExpr::create(Sym, OutContext);
  auto Expr = MCSpecifierExpr::create(Res, ELF::R_Capstone_CALL_PLT, OutContext);

  EmitToStreamer(*OutStreamer, MCInstBuilder(Capstone::PseudoCALL).addExpr(Expr));
}

void CapstoneAsmPrinter::LowerKCFI_CHECK(const MachineInstr &MI) {
  Register AddrReg = MI.getOperand(0).getReg();
  assert(std::next(MI.getIterator())->isCall() &&
         "KCFI_CHECK not followed by a call instruction");
  assert(std::next(MI.getIterator())->getOperand(0).getReg() == AddrReg &&
         "KCFI_CHECK call target doesn't match call operand");

  // Temporary registers for comparing the hashes. If a register is used
  // for the call target, or reserved by the user, we can clobber another
  // temporary register as the check is immediately followed by the
  // call. The check defaults to X6/X7, but can fall back to X28-X31 if
  // needed.
  unsigned ScratchRegs[] = {Capstone::X6, Capstone::X7};
  unsigned NextReg = Capstone::X28;
  auto isRegAvailable = [&](unsigned Reg) {
    return Reg != AddrReg && !STI->isRegisterReservedByUser(Reg);
  };
  for (auto &Reg : ScratchRegs) {
    if (isRegAvailable(Reg))
      continue;
    while (!isRegAvailable(NextReg))
      ++NextReg;
    Reg = NextReg++;
    if (Reg > Capstone::X31)
      report_fatal_error("Unable to find scratch registers for KCFI_CHECK");
  }

  if (AddrReg == Capstone::X0) {
    // Checking X0 makes no sense. Instead of emitting a load, zero
    // ScratchRegs[0].
    EmitToStreamer(*OutStreamer, MCInstBuilder(Capstone::ADDI)
                                     .addReg(ScratchRegs[0])
                                     .addReg(Capstone::X0)
                                     .addImm(0));
  } else {
    // Adjust the offset for patchable-function-prefix. This assumes that
    // patchable-function-prefix is the same for all functions.
    int NopSize = STI->hasStdExtZca() ? 2 : 4;
    int64_t PrefixNops = 0;
    (void)MI.getMF()
        ->getFunction()
        .getFnAttribute("patchable-function-prefix")
        .getValueAsString()
        .getAsInteger(10, PrefixNops);

    // Load the target function type hash.
    EmitToStreamer(*OutStreamer, MCInstBuilder(Capstone::LW)
                                     .addReg(ScratchRegs[0])
                                     .addReg(AddrReg)
                                     .addImm(-(PrefixNops * NopSize + 4)));
  }

  // Load the expected 32-bit type hash.
  const int64_t Type = MI.getOperand(1).getImm();
  const int64_t Hi20 = ((Type + 0x800) >> 12) & 0xFFFFF;
  const int64_t Lo12 = SignExtend64<12>(Type);
  if (Hi20) {
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(Capstone::LUI).addReg(ScratchRegs[1]).addImm(Hi20));
  }
  if (Lo12 || Hi20 == 0) {
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder((STI->hasFeature(Capstone::Feature64Bit) && Hi20)
                                     ? Capstone::ADDIW
                                     : Capstone::ADDI)
                       .addReg(ScratchRegs[1])
                       .addReg(ScratchRegs[1])
                       .addImm(Lo12));
  }

  // Compare the hashes and trap if there's a mismatch.
  MCSymbol *Pass = OutContext.createTempSymbol();
  EmitToStreamer(*OutStreamer,
                 MCInstBuilder(Capstone::BEQ)
                     .addReg(ScratchRegs[0])
                     .addReg(ScratchRegs[1])
                     .addExpr(MCSymbolRefExpr::create(Pass, OutContext)));

  MCSymbol *Trap = OutContext.createTempSymbol();
  OutStreamer->emitLabel(Trap);
  EmitToStreamer(*OutStreamer, MCInstBuilder(Capstone::EBREAK));
  emitKCFITrapEntry(*MI.getMF(), Trap);
  OutStreamer->emitLabel(Pass);
}

void CapstoneAsmPrinter::EmitHwasanMemaccessSymbols(Module &M) {
  if (HwasanMemaccessSymbols.empty())
    return;

  assert(TM.getTargetTriple().isOSBinFormatELF());
  // Use MCSubtargetInfo from TargetMachine. Individual functions may have
  // attributes that differ from other functions in the module and we have no
  // way to know which function is correct.
  const MCSubtargetInfo &MCSTI = *TM.getMCSubtargetInfo();

  MCSymbol *HwasanTagMismatchV2Sym =
      OutContext.getOrCreateSymbol("__hwasan_tag_mismatch_v2");
  // Annotate symbol as one having incompatible calling convention, so
  // run-time linkers can instead eagerly bind this function.
  auto &RTS =
      static_cast<CapstoneTargetStreamer &>(*OutStreamer->getTargetStreamer());
  RTS.emitDirectiveVariantCC(*HwasanTagMismatchV2Sym);

  const MCSymbolRefExpr *HwasanTagMismatchV2Ref =
      MCSymbolRefExpr::create(HwasanTagMismatchV2Sym, OutContext);
  auto Expr = MCSpecifierExpr::create(HwasanTagMismatchV2Ref,
                                      ELF::R_Capstone_CALL_PLT, OutContext);

  for (auto &P : HwasanMemaccessSymbols) {
    unsigned Reg = std::get<0>(P.first);
    uint32_t AccessInfo = std::get<1>(P.first);
    MCSymbol *Sym = P.second;

    unsigned Size =
        1 << ((AccessInfo >> HWASanAccessInfo::AccessSizeShift) & 0xf);
    OutStreamer->switchSection(OutContext.getELFSection(
        ".text.hot", ELF::SHT_PROGBITS,
        ELF::SHF_EXECINSTR | ELF::SHF_ALLOC | ELF::SHF_GROUP, 0, Sym->getName(),
        /*IsComdat=*/true));

    OutStreamer->emitSymbolAttribute(Sym, MCSA_ELF_TypeFunction);
    OutStreamer->emitSymbolAttribute(Sym, MCSA_Weak);
    OutStreamer->emitSymbolAttribute(Sym, MCSA_Hidden);
    OutStreamer->emitLabel(Sym);

    // Extract shadow offset from ptr
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(Capstone::SLLI).addReg(Capstone::X6).addReg(Reg).addImm(8),
        MCSTI);
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(Capstone::SRLI)
                       .addReg(Capstone::X6)
                       .addReg(Capstone::X6)
                       .addImm(12),
                   MCSTI);
    // load shadow tag in X6, X5 contains shadow base
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(Capstone::ADD)
                       .addReg(Capstone::X6)
                       .addReg(Capstone::X5)
                       .addReg(Capstone::X6),
                   MCSTI);
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(Capstone::LBU).addReg(Capstone::X6).addReg(Capstone::X6).addImm(0),
        MCSTI);
    // Extract tag from pointer and compare it with loaded tag from shadow
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(Capstone::SRLI).addReg(Capstone::X7).addReg(Reg).addImm(56),
        MCSTI);
    MCSymbol *HandleMismatchOrPartialSym = OutContext.createTempSymbol();
    // X7 contains tag from the pointer, while X6 contains tag from memory
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(Capstone::BNE)
                       .addReg(Capstone::X7)
                       .addReg(Capstone::X6)
                       .addExpr(MCSymbolRefExpr::create(
                           HandleMismatchOrPartialSym, OutContext)),
                   MCSTI);
    MCSymbol *ReturnSym = OutContext.createTempSymbol();
    OutStreamer->emitLabel(ReturnSym);
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(Capstone::JALR)
                       .addReg(Capstone::X0)
                       .addReg(Capstone::X1)
                       .addImm(0),
                   MCSTI);
    OutStreamer->emitLabel(HandleMismatchOrPartialSym);

    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(Capstone::ADDI)
                       .addReg(Capstone::X28)
                       .addReg(Capstone::X0)
                       .addImm(16),
                   MCSTI);
    MCSymbol *HandleMismatchSym = OutContext.createTempSymbol();
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(Capstone::BGEU)
            .addReg(Capstone::X6)
            .addReg(Capstone::X28)
            .addExpr(MCSymbolRefExpr::create(HandleMismatchSym, OutContext)),
        MCSTI);

    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(Capstone::ANDI).addReg(Capstone::X28).addReg(Reg).addImm(0xF),
        MCSTI);

    if (Size != 1)
      EmitToStreamer(*OutStreamer,
                     MCInstBuilder(Capstone::ADDI)
                         .addReg(Capstone::X28)
                         .addReg(Capstone::X28)
                         .addImm(Size - 1),
                     MCSTI);
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(Capstone::BGE)
            .addReg(Capstone::X28)
            .addReg(Capstone::X6)
            .addExpr(MCSymbolRefExpr::create(HandleMismatchSym, OutContext)),
        MCSTI);

    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(Capstone::ORI).addReg(Capstone::X6).addReg(Reg).addImm(0xF),
        MCSTI);
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(Capstone::LBU).addReg(Capstone::X6).addReg(Capstone::X6).addImm(0),
        MCSTI);
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(Capstone::BEQ)
                       .addReg(Capstone::X6)
                       .addReg(Capstone::X7)
                       .addExpr(MCSymbolRefExpr::create(ReturnSym, OutContext)),
                   MCSTI);

    OutStreamer->emitLabel(HandleMismatchSym);

    // | Previous stack frames...        |
    // +=================================+ <-- [SP + 256]
    // |              ...                |
    // |                                 |
    // | Stack frame space for x12 - x31.|
    // |                                 |
    // |              ...                |
    // +---------------------------------+ <-- [SP + 96]
    // | Saved x11(arg1), as             |
    // | __hwasan_check_* clobbers it.   |
    // +---------------------------------+ <-- [SP + 88]
    // | Saved x10(arg0), as             |
    // | __hwasan_check_* clobbers it.   |
    // +---------------------------------+ <-- [SP + 80]
    // |                                 |
    // | Stack frame space for x9.       |
    // +---------------------------------+ <-- [SP + 72]
    // |                                 |
    // | Saved x8(fp), as                |
    // | __hwasan_check_* clobbers it.   |
    // +---------------------------------+ <-- [SP + 64]
    // |              ...                |
    // |                                 |
    // | Stack frame space for x2 - x7.  |
    // |                                 |
    // |              ...                |
    // +---------------------------------+ <-- [SP + 16]
    // | Return address (x1) for caller  |
    // | of __hwasan_check_*.            |
    // +---------------------------------+ <-- [SP + 8]
    // | Reserved place for x0, possibly |
    // | junk, since we don't save it.   |
    // +---------------------------------+ <-- [x2 / SP]

    // Adjust sp
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(Capstone::ADDI)
                       .addReg(Capstone::X2)
                       .addReg(Capstone::X2)
                       .addImm(-256),
                   MCSTI);

    // store x10(arg0) by new sp
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(Capstone::SD)
                       .addReg(Capstone::X10)
                       .addReg(Capstone::X2)
                       .addImm(8 * 10),
                   MCSTI);
    // store x11(arg1) by new sp
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(Capstone::SD)
                       .addReg(Capstone::X11)
                       .addReg(Capstone::X2)
                       .addImm(8 * 11),
                   MCSTI);

    // store x8(fp) by new sp
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(Capstone::SD).addReg(Capstone::X8).addReg(Capstone::X2).addImm(8 *
                                                                            8),
        MCSTI);
    // store x1(ra) by new sp
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(Capstone::SD).addReg(Capstone::X1).addReg(Capstone::X2).addImm(1 *
                                                                            8),
        MCSTI);
    if (Reg != Capstone::X10)
      EmitToStreamer(
          *OutStreamer,
          MCInstBuilder(Capstone::ADDI).addReg(Capstone::X10).addReg(Reg).addImm(0),
          MCSTI);
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(Capstone::ADDI)
                       .addReg(Capstone::X11)
                       .addReg(Capstone::X0)
                       .addImm(AccessInfo & HWASanAccessInfo::RuntimeMask),
                   MCSTI);

    EmitToStreamer(*OutStreamer, MCInstBuilder(Capstone::PseudoCALL).addExpr(Expr),
                   MCSTI);
  }
}

void CapstoneAsmPrinter::emitNoteGnuProperty(const Module &M) {
  if (const Metadata *const Flag = M.getModuleFlag("cf-protection-return");
      Flag && !mdconst::extract<ConstantInt>(Flag)->isZero()) {
    CapstoneTargetStreamer &RTS =
        static_cast<CapstoneTargetStreamer &>(*OutStreamer->getTargetStreamer());
    RTS.emitNoteGnuPropertySection(ELF::GNU_PROPERTY_Capstone_FEATURE_1_CFI_SS);
  }
}

static MCOperand lowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym,
                                    const AsmPrinter &AP) {
  MCContext &Ctx = AP.OutContext;
  Capstone::Specifier Kind;

  switch (MO.getTargetFlags()) {
  default:
    llvm_unreachable("Unknown target flag on GV operand");
  case CapstoneII::MO_None:
    Kind = Capstone::S_None;
    break;
  case CapstoneII::MO_CALL:
    Kind = ELF::R_Capstone_CALL_PLT;
    break;
  case CapstoneII::MO_LO:
    Kind = Capstone::S_LO;
    break;
  case CapstoneII::MO_HI:
    Kind = ELF::R_Capstone_HI20;
    break;
  case CapstoneII::MO_PCREL_LO:
    Kind = Capstone::S_PCREL_LO;
    break;
  case CapstoneII::MO_PCREL_HI:
    Kind = ELF::R_Capstone_PCREL_HI20;
    break;
  case CapstoneII::MO_GOT_HI:
    Kind = ELF::R_Capstone_GOT_HI20;
    break;
  case CapstoneII::MO_TPREL_LO:
    Kind = Capstone::S_TPREL_LO;
    break;
  case CapstoneII::MO_TPREL_HI:
    Kind = ELF::R_Capstone_TPREL_HI20;
    break;
  case CapstoneII::MO_TPREL_ADD:
    Kind = ELF::R_Capstone_TPREL_ADD;
    break;
  case CapstoneII::MO_TLS_GOT_HI:
    Kind = ELF::R_Capstone_TLS_GOT_HI20;
    break;
  case CapstoneII::MO_TLS_GD_HI:
    Kind = ELF::R_Capstone_TLS_GD_HI20;
    break;
  case CapstoneII::MO_TLSDESC_HI:
    Kind = ELF::R_Capstone_TLSDESC_HI20;
    break;
  case CapstoneII::MO_TLSDESC_LOAD_LO:
    Kind = ELF::R_Capstone_TLSDESC_LOAD_LO12;
    break;
  case CapstoneII::MO_TLSDESC_ADD_LO:
    Kind = ELF::R_Capstone_TLSDESC_ADD_LO12;
    break;
  case CapstoneII::MO_TLSDESC_CALL:
    Kind = ELF::R_Capstone_TLSDESC_CALL;
    break;
  }

  const MCExpr *ME = MCSymbolRefExpr::create(Sym, Ctx);

  if (!MO.isJTI() && !MO.isMBB() && MO.getOffset())
    ME = MCBinaryExpr::createAdd(
        ME, MCConstantExpr::create(MO.getOffset(), Ctx), Ctx);

  if (Kind != Capstone::S_None)
    ME = MCSpecifierExpr::create(ME, Kind, Ctx);
  return MCOperand::createExpr(ME);
}

bool CapstoneAsmPrinter::lowerOperand(const MachineOperand &MO,
                                   MCOperand &MCOp) const {
  switch (MO.getType()) {
  default:
    report_fatal_error("lowerOperand: unknown operand type");
  case MachineOperand::MO_Register:
    // Ignore all implicit register operands.
    if (MO.isImplicit())
      return false;
    MCOp = MCOperand::createReg(MO.getReg());
    break;
  case MachineOperand::MO_RegisterMask:
    // Regmasks are like implicit defs.
    return false;
  case MachineOperand::MO_Immediate:
    MCOp = MCOperand::createImm(MO.getImm());
    break;
  case MachineOperand::MO_MachineBasicBlock:
    MCOp = lowerSymbolOperand(MO, MO.getMBB()->getSymbol(), *this);
    break;
  case MachineOperand::MO_GlobalAddress:
    MCOp = lowerSymbolOperand(MO, getSymbolPreferLocal(*MO.getGlobal()), *this);
    break;
  case MachineOperand::MO_BlockAddress:
    MCOp = lowerSymbolOperand(MO, GetBlockAddressSymbol(MO.getBlockAddress()),
                              *this);
    break;
  case MachineOperand::MO_ExternalSymbol:
    MCOp = lowerSymbolOperand(MO, GetExternalSymbolSymbol(MO.getSymbolName()),
                              *this);
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    MCOp = lowerSymbolOperand(MO, GetCPISymbol(MO.getIndex()), *this);
    break;
  case MachineOperand::MO_JumpTableIndex:
    MCOp = lowerSymbolOperand(MO, GetJTISymbol(MO.getIndex()), *this);
    break;
  case MachineOperand::MO_MCSymbol:
    MCOp = lowerSymbolOperand(MO, MO.getMCSymbol(), *this);
    break;
  }
  return true;
}

static bool lowerCapstoneVMachineInstrToMCInst(const MachineInstr *MI,
                                            MCInst &OutMI,
                                            const CapstoneSubtarget *STI) {
  const CapstoneVPseudosTable::PseudoInfo *RVV =
      CapstoneVPseudosTable::getPseudoInfo(MI->getOpcode());
  if (!RVV)
    return false;

  OutMI.setOpcode(RVV->BaseInstr);

  const TargetInstrInfo *TII = STI->getInstrInfo();
  const TargetRegisterInfo *TRI = STI->getRegisterInfo();
  assert(TRI && "TargetRegisterInfo expected");

  const MCInstrDesc &MCID = MI->getDesc();
  uint64_t TSFlags = MCID.TSFlags;
  unsigned NumOps = MI->getNumExplicitOperands();

  // Skip policy, SEW, VL, VXRM/FRM operands which are the last operands if
  // present.
  if (CapstoneII::hasVecPolicyOp(TSFlags))
    --NumOps;
  if (CapstoneII::hasSEWOp(TSFlags))
    --NumOps;
  if (CapstoneII::hasVLOp(TSFlags))
    --NumOps;
  if (CapstoneII::hasRoundModeOp(TSFlags))
    --NumOps;

  bool hasVLOutput = CapstoneInstrInfo::isFaultOnlyFirstLoad(*MI);
  for (unsigned OpNo = 0; OpNo != NumOps; ++OpNo) {
    const MachineOperand &MO = MI->getOperand(OpNo);
    // Skip vl output. It should be the second output.
    if (hasVLOutput && OpNo == 1)
      continue;

    // Skip passthru op. It should be the first operand after the defs.
    if (OpNo == MI->getNumExplicitDefs() && MO.isReg() && MO.isTied()) {
      assert(MCID.getOperandConstraint(OpNo, MCOI::TIED_TO) == 0 &&
             "Expected tied to first def.");
      const MCInstrDesc &OutMCID = TII->get(OutMI.getOpcode());
      // Skip if the next operand in OutMI is not supposed to be tied. Unless it
      // is a _TIED instruction.
      if (OutMCID.getOperandConstraint(OutMI.getNumOperands(), MCOI::TIED_TO) <
              0 &&
          !CapstoneII::isTiedPseudo(TSFlags))
        continue;
    }

    MCOperand MCOp;
    switch (MO.getType()) {
    default:
      llvm_unreachable("Unknown operand type");
    case MachineOperand::MO_Register: {
      Register Reg = MO.getReg();

      if (Capstone::VRM2RegClass.contains(Reg) ||
          Capstone::VRM4RegClass.contains(Reg) ||
          Capstone::VRM8RegClass.contains(Reg)) {
        Reg = TRI->getSubReg(Reg, Capstone::sub_vrm1_0);
        assert(Reg && "Subregister does not exist");
      } else if (Capstone::FPR16RegClass.contains(Reg)) {
        Reg =
            TRI->getMatchingSuperReg(Reg, Capstone::sub_16, &Capstone::FPR32RegClass);
        assert(Reg && "Subregister does not exist");
      } else if (Capstone::FPR64RegClass.contains(Reg)) {
        Reg = TRI->getSubReg(Reg, Capstone::sub_32);
        assert(Reg && "Superregister does not exist");
      } else if (Capstone::VRN2M1RegClass.contains(Reg) ||
                 Capstone::VRN2M2RegClass.contains(Reg) ||
                 Capstone::VRN2M4RegClass.contains(Reg) ||
                 Capstone::VRN3M1RegClass.contains(Reg) ||
                 Capstone::VRN3M2RegClass.contains(Reg) ||
                 Capstone::VRN4M1RegClass.contains(Reg) ||
                 Capstone::VRN4M2RegClass.contains(Reg) ||
                 Capstone::VRN5M1RegClass.contains(Reg) ||
                 Capstone::VRN6M1RegClass.contains(Reg) ||
                 Capstone::VRN7M1RegClass.contains(Reg) ||
                 Capstone::VRN8M1RegClass.contains(Reg)) {
        Reg = TRI->getSubReg(Reg, Capstone::sub_vrm1_0);
        assert(Reg && "Subregister does not exist");
      }

      MCOp = MCOperand::createReg(Reg);
      break;
    }
    case MachineOperand::MO_Immediate:
      MCOp = MCOperand::createImm(MO.getImm());
      break;
    }
    OutMI.addOperand(MCOp);
  }

  // Unmasked pseudo instructions need to append dummy mask operand to
  // V instructions. All V instructions are modeled as the masked version.
  const MCInstrDesc &OutMCID = TII->get(OutMI.getOpcode());
  if (OutMI.getNumOperands() < OutMCID.getNumOperands()) {
    assert(OutMCID.operands()[OutMI.getNumOperands()].RegClass ==
               Capstone::VMV0RegClassID &&
           "Expected only mask operand to be missing");
    OutMI.addOperand(MCOperand::createReg(Capstone::NoRegister));
  }

  assert(OutMI.getNumOperands() == OutMCID.getNumOperands());
  return true;
}

void CapstoneAsmPrinter::lowerToMCInst(const MachineInstr *MI, MCInst &OutMI) {
  if (lowerCapstoneVMachineInstrToMCInst(MI, OutMI, STI))
    return;

  OutMI.setOpcode(MI->getOpcode());

  for (const MachineOperand &MO : MI->operands()) {
    MCOperand MCOp;
    if (lowerOperand(MO, MCOp))
      OutMI.addOperand(MCOp);
  }
}

void CapstoneAsmPrinter::emitMachineConstantPoolValue(
    MachineConstantPoolValue *MCPV) {
  auto *RCPV = static_cast<CapstoneConstantPoolValue *>(MCPV);
  MCSymbol *MCSym;

  if (RCPV->isGlobalValue()) {
    auto *GV = RCPV->getGlobalValue();
    MCSym = getSymbol(GV);
  } else {
    assert(RCPV->isExtSymbol() && "unrecognized constant pool type");
    auto Sym = RCPV->getSymbol();
    MCSym = GetExternalSymbolSymbol(Sym);
  }

  const MCExpr *Expr = MCSymbolRefExpr::create(MCSym, OutContext);
  uint64_t Size = getDataLayout().getTypeAllocSize(RCPV->getType());
  OutStreamer->emitValue(Expr, Size);
}

char CapstoneAsmPrinter::ID = 0;

INITIALIZE_PASS(CapstoneAsmPrinter, "capstone-asm-printer", "Capstone Assembly Printer",
                false, false)
