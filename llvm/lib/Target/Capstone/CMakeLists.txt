add_llvm_component_group(Capstone)

set(LLVM_TARGET_DEFINITIONS Capstone.td)

tablegen(LLVM CapstoneGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(CapstoneCommonTableGen)

add_llvm_target(CapstoneCodeGen
  CapstoneISelLowering.cpp
  CapstoneSimpleConstantPropagation.cpp
  CapstoneSubtarget.cpp
  CapstoneTargetMachine.cpp
  CapstoneTargetObjectFile.cpp
  CapstoneTargetTransformInfo.cpp

  LINK_COMPONENTS
  CapstoneDesc
  CapstoneInfo

  Analysis
  CodeGen
  Core
  MC
  Passes
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  Capstone
)

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
