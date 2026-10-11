target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-unknown-elf"
declare ptr addrspace(200) @llvm.ptrmask.p200.i64(ptr addrspace(200), i64)
define ptr addrspace(200) @g(ptr addrspace(200) %p) addrspace(200) {
  %a = call ptr addrspace(200) @llvm.ptrmask.p200.i64(ptr addrspace(200) %p, i64 -4)
  ret ptr addrspace(200) %a
}
