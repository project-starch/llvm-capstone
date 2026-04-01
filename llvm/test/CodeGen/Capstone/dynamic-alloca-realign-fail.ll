; RUN: not llc -mtriple=capstone64 -o - %s 2>&1 | FileCheck %s
target triple = "capstone64"
; CHECK: LLVM ERROR: Stack realignment is not supported yet in Capstone PureCap
define ptr addrspace(200) @vla_realign(i64 %n) {
entry:
  %p = alloca i8, i64 %n, align 64, addrspace(200)
  ret ptr addrspace(200) %p
}
