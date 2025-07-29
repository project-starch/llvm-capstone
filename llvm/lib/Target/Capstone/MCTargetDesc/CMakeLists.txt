add_llvm_component_library(LLVMCapstoneDesc
  CapstoneMCAsmInfo.cpp
  CapstoneMCTargetDesc.cpp

  LINK_COMPONENTS
  Support
  CapstoneInfo
  MC

  ADD_TO_COMPONENT
  Capstone
  )
