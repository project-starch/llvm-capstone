; C-79: library calls that the simplifier emits take and return capabilities.
;
; On Capstone every C pointer is a capability in the data layout's globals
; address space (G200). The simplifier declared every library function it
; emitted with address-space-0 pointers, so a strchr folded to memchr returned a
; plain integer that was compared against a capability null. That IR is
; malformed: release builds run no verifier, and instruction selection crashed
; on it (SPEC CPU2017 500.perlbench_r, toke.c, Perl_validate_proto). Each rewrite
; below must produce a call, and a declaration, in address space 200.
;
; MUTATION: with B.getPtrTy() restored in BuildLibCalls.cpp, opt rejects this
; file's output in its verifier ("Call parameter type does not match function
; signature"), so every CHECK below fails; with only the declarations fixed and
; IRBuilder::CreateGlobalString still defaulting to address space 0, the puts
; call is rejected the same way, on @str (both performed 2026-10-13). @copy on
; its own passes either way: its memcpy is an intrinsic, typed by its operands,
; so it is a control, not a test of the fix.
;
; RUN: opt -passes=instcombine -S < %s | FileCheck %s

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64"

@set = private addrspace(200) constant [12 x i8] c"$@%*;[]&\\_+\00"
@hello = private addrspace(200) constant [7 x i8] c"hello\0A\00"
@src = private addrspace(200) constant [6 x i8] c"hello\00"

declare ptr addrspace(200) @strchr(ptr addrspace(200), i32) addrspace(200)
declare i32 @printf(ptr addrspace(200), ...) addrspace(200)
declare ptr addrspace(200) @strcpy(ptr addrspace(200), ptr addrspace(200)) addrspace(200)

; The puts rewrite below adds a global; globals print before the functions.
; CHECK: @str = private unnamed_addr addrspace(200) constant [6 x i8] c"hello\00"

; strchr of a constant string with a variable character becomes memchr over
; the string, compared against a capability null.
; CHECK-LABEL: @in_set(
; CHECK:       [[P:%.*]] = call addrspace(200) ptr addrspace(200) @memchr(ptr addrspace(200) {{.*}}@set, i32 %c, i64 12)
; CHECK-NEXT:  icmp ne ptr addrspace(200) [[P]], null
define i1 @in_set(i32 %c) addrspace(200) {
  %p = call ptr addrspace(200) @strchr(ptr addrspace(200) @set, i32 %c)
  %r = icmp ne ptr addrspace(200) %p, null
  ret i1 %r
}

; printf of a constant string ending in a newline becomes puts of a NEW global,
; the string without its newline, which must be a capability global too.
; CHECK-LABEL: @hi(
; CHECK:       call addrspace(200) i32 @puts(ptr addrspace(200) @str)
define void @hi() addrspace(200) {
  %r = call i32 (ptr addrspace(200), ...) @printf(ptr addrspace(200) @hello)
  ret void
}

; strcpy whose result is used becomes a memcpy of the known length; the
; returned pointer is the destination capability itself.
; CHECK-LABEL: @copy(
; CHECK:       call addrspace(200) void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) {{.*}}%d, ptr addrspace(200) {{.*}}@src, i64 6, i1 false)
; CHECK-NEXT:  ret ptr addrspace(200) %d
define ptr addrspace(200) @copy(ptr addrspace(200) %d) addrspace(200) {
  %r = call ptr addrspace(200) @strcpy(ptr addrspace(200) %d, ptr addrspace(200) @src)
  ret ptr addrspace(200) %r
}

; CHECK-DAG: declare ptr addrspace(200) @memchr(ptr addrspace(200), i32, i64)
; CHECK-DAG: declare noundef i32 @puts(ptr addrspace(200) noundef readonly captures(none))
