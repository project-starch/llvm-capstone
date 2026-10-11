target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-unknown-elf"
define ptr addrspace(200) @f(ptr addrspace(200) %a, i1 %c) addrspace(200) {
  %x = select i1 %c, ptr addrspace(200) %a, ptr addrspace(200) null
  ret ptr addrspace(200) %x
}
