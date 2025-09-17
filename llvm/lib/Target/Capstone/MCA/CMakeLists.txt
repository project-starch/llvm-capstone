add_llvm_component_library(LLVMCapstoneTargetMCA
  CapstoneCustomBehaviour.cpp

  LINK_COMPONENTS
  CodeGenTypes
  MC
  MCA
  MCParser
  CapstoneDesc
  CapstoneInfo
  Support
  TargetParser

  ADD_TO_COMPONENT
  Capstone
  )
