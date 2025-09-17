add_llvm_component_library(LLVMCapstoneAsmParser
  CapstoneAsmParser.cpp

  LINK_COMPONENTS
  MC
  MCParser
  CapstoneDesc
  CapstoneInfo
  Support
  TargetParser

  ADD_TO_COMPONENT
  Capstone
  )
