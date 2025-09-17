add_llvm_component_library(LLVMCapstoneDesc
  CapstoneAsmBackend.cpp
  CapstoneBaseInfo.cpp
  CapstoneELFObjectWriter.cpp
  CapstoneInstPrinter.cpp
  CapstoneMCAsmInfo.cpp
  CapstoneMCCodeEmitter.cpp
  CapstoneMCExpr.cpp
  CapstoneMCObjectFileInfo.cpp
  CapstoneMCTargetDesc.cpp
  CapstoneMatInt.cpp
  CapstoneTargetStreamer.cpp
  CapstoneELFStreamer.cpp

  LINK_COMPONENTS
  MC
  CapstoneInfo
  Support
  TargetParser

  ADD_TO_COMPONENT
  Capstone
)
