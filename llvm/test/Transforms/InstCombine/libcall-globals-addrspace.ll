; RUN: opt -passes=instcombine -S < %s | FileCheck %s
; RUN: sed -e 's/-G1"/-p7:160:256:256:32-ni:7-G7"/' %s | opt -passes=instcombine -S | FileCheck %s
; RUN: sed -e 's/-G1"/-p8:128:128:128:48-ni:8-G8"/' %s | opt -passes=instcombine -S | FileCheck %s
;
; C-79 controls. Library calls the simplifier emits, and the strings it
; creates, take the globals address space only when the data layout puts
; globals, allocas and functions all in it (a pure-capability layout,
; capstone-libcall-capability-pointers.ll). A layout whose globals alone live
; elsewhere keeps address space 0, as upstream: G1 (AMDGPU's globals, beside A5
; and P0), and globals in AMDGPU's buffer fat pointer address spaces 7 and 8,
; whose pointers are wider than their index but are no C pointer type. Taking G
; here would declare memchr on that address space while its operand is a plain
; ptr.

target datalayout = "e-p:64:64-i64:64-G1"

; CHECK: @str = private unnamed_addr constant [6 x i8] c"hello\00"
@chars = constant [6 x i8] c"abcde\00"
@hello = constant [7 x i8] c"hello\0A\00"

declare ptr @strchr(ptr, i32)
declare i32 @printf(ptr, ...)

; strchr over a constant set becomes memchr, which instcombine then folds to a
; range test; the declaration it made stays.
; CHECK-LABEL: @in_set(
define i1 @in_set(i32 %c) {
  %p = call ptr @strchr(ptr @chars, i32 %c)
  %r = icmp ne ptr %p, null
  ret i1 %r
}

; CHECK-LABEL: @greet(
; CHECK: call i32 @puts(ptr {{.*}}@str)
define void @greet() {
  %n = call i32 (ptr, ...) @printf(ptr @hello)
  ret void
}

; CHECK: declare ptr @memchr(ptr, i32, i64)
