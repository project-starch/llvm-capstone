target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-unknown-elf"

define ptr addrspace(200) @dict___contains__(ptr addrspace(200) %self, i1 %tobool.not) addrspace(200) {
entry:
  %_Py_FalseStruct._Py_TrueStruct = select i1 %tobool.not, ptr addrspace(200) %self, ptr addrspace(200) null
  %retval.0 = select i1 %tobool.not, ptr addrspace(200) null, ptr addrspace(200) %_Py_FalseStruct._Py_TrueStruct
  ret ptr addrspace(200) %retval.0
}
