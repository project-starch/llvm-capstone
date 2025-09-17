add_llvm_component_library(LLVMCapstoneDisassembler
  CapstoneDisassembler.cpp

  LINK_COMPONENTS
  MC
  MCDisassembler
  CapstoneDesc
  CapstoneInfo
  Support

  ADD_TO_COMPONENT
  Capstone
  )
