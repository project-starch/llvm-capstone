add_llvm_component_group(Capstone)

set(LLVM_TARGET_DEFINITIONS Capstone.td)

tablegen(LLVM CapstoneGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM CapstoneGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM CapstoneGenCompressInstEmitter.inc -gen-compress-inst-emitter)
tablegen(LLVM CapstoneGenMacroFusion.inc -gen-macro-fusion-pred)
tablegen(LLVM CapstoneGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM CapstoneGenDisassemblerTables.inc -gen-disassembler
              --specialize-decoders-per-bitwidth)
tablegen(LLVM CapstoneGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM CapstoneGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM CapstoneGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM CapstoneGenRegisterBank.inc -gen-register-bank)
tablegen(LLVM CapstoneGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM CapstoneGenSearchableTables.inc -gen-searchable-tables)
tablegen(LLVM CapstoneGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM CapstoneGenExegesis.inc -gen-exegesis)
tablegen(LLVM CapstoneGenSDNodeInfo.inc -gen-sd-node-info)

set(LLVM_TARGET_DEFINITIONS CapstoneGISel.td)
tablegen(LLVM CapstoneGenGlobalISel.inc -gen-global-isel)
tablegen(LLVM CapstoneGenO0PreLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="CapstoneO0PreLegalizerCombiner")
tablegen(LLVM CapstoneGenPreLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="CapstonePreLegalizerCombiner")
tablegen(LLVM CapstoneGenPostLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="CapstonePostLegalizerCombiner")

add_public_tablegen_target(CapstoneCommonTableGen)

add_llvm_target(CapstoneCodeGen
  CapstoneAsmPrinter.cpp
  CapstoneCallingConv.cpp
  CapstoneCodeGenPrepare.cpp
  CapstoneConstantPoolValue.cpp
  CapstoneDeadRegisterDefinitions.cpp
  CapstoneExpandAtomicPseudoInsts.cpp
  CapstoneExpandPseudoInsts.cpp
  CapstoneFoldMemOffset.cpp
  CapstoneFrameLowering.cpp
  CapstoneGatherScatterLowering.cpp
  CapstoneIndirectBranchTracking.cpp
  CapstoneInsertReadWriteCSR.cpp
  CapstoneInsertVSETVLI.cpp
  CapstoneInsertWriteVXRM.cpp
  CapstoneInstrInfo.cpp
  CapstoneInterleavedAccess.cpp
  CapstoneISelDAGToDAG.cpp
  CapstoneISelLowering.cpp
  CapstoneLandingPadSetup.cpp
  CapstoneLateBranchOpt.cpp
  CapstoneLoadStoreOptimizer.cpp
  CapstoneMachineFunctionInfo.cpp
  CapstoneMakeCompressible.cpp
  CapstoneMergeBaseOffset.cpp
  CapstoneMoveMerger.cpp
  CapstoneOptWInstrs.cpp
  CapstonePostRAExpandPseudoInsts.cpp
  CapstonePushPopOptimizer.cpp
  CapstoneRedundantCopyElimination.cpp
  CapstoneRegisterInfo.cpp
  CapstoneSelectionDAGInfo.cpp
  CapstoneSubtarget.cpp
  CapstoneTargetMachine.cpp
  CapstoneTargetObjectFile.cpp
  CapstoneTargetTransformInfo.cpp
  CapstoneVectorMaskDAGMutation.cpp
  CapstoneVectorPeephole.cpp
  CapstoneVLOptimizer.cpp
  CapstoneVMV0Elimination.cpp
  CapstoneZacasABIFix.cpp
  GISel/CapstoneCallLowering.cpp
  GISel/CapstoneInstructionSelector.cpp
  GISel/CapstoneLegalizerInfo.cpp
  GISel/CapstonePostLegalizerCombiner.cpp
  GISel/CapstoneO0PreLegalizerCombiner.cpp
  GISel/CapstonePreLegalizerCombiner.cpp
  GISel/CapstoneRegisterBankInfo.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  IPO
  MC
  Passes
  CapstoneDesc
  CapstoneInfo
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  Vectorize

  ADD_TO_COMPONENT
  Capstone
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(MCA)
add_subdirectory(TargetInfo)
