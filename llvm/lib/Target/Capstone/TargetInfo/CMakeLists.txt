add_llvm_component_library(LLVMCapstoneInfo
  CapstoneTargetInfo.cpp

  LINK_COMPONENTS
  MC
  Support

  ADD_TO_COMPONENT
  Capstone
  )
