; Every production flag set at every optimisation level in use, verifier only:
; the default ABI, -capstone-gp-captable and -capstone-gp-free, each at -O0,
; -O1 and -O2, plus the two granularity knobs at both values.  The module is a
; kitchen sink of the shapes the corpus builds contain -- globals of every
; kind, a capability initializer, a string, a direct call, an indirect call
; through a function-pointer global, a tail call, aligned and unaligned
; aggregate copies, memset, an address-taken alloca, a dynamic alloca, a switch
; wide enough for a jump table, a varargs call, i128 arithmetic and a select
; between capabilities.  A "Cannot select" or verifier failure at any cell is
; the class of failure every -O1 defect on this target has been.
;
; RUN: %llc_cap -O0 -capstone-gp-free=false -capstone-gp-captable=false < %s -o /dev/null
; RUN: %llc_cap -O1 -capstone-gp-free=false -capstone-gp-captable=false < %s -o /dev/null
; RUN: %llc_cap -O2 -capstone-gp-free=false -capstone-gp-captable=false < %s -o /dev/null
; RUN: %llc_cap -O0 -capstone-gp-captable < %s -o /dev/null
; RUN: %llc_cap -O1 -capstone-gp-captable < %s -o /dev/null
; RUN: %llc_cap -O2 -capstone-gp-captable < %s -o /dev/null
; RUN: %llc_cap -O0 -capstone-gp-free < %s -o /dev/null
; RUN: %llc_cap -O1 -capstone-gp-free < %s -o /dev/null
; RUN: %llc_cap -O2 -capstone-gp-free < %s -o /dev/null
; RUN: %llc_cap -O2 -capstone-shrink-globals=true -capstone-shrink-stack=true < %s -o /dev/null
; RUN: %llc_cap -O2 -capstone-shrink-globals=false -capstone-shrink-stack=false < %s -o /dev/null

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

@zero32 = addrspace(200) global i32 0
@init64 = addrspace(200) global i64 42
@pinit = addrspace(200) global ptr addrspace(200) @init64
@str = private unnamed_addr addrspace(200) constant [6 x i8] c"hello\00"
@fptr = addrspace(200) global ptr addrspace(200) @callee
@buf = addrspace(200) global [48 x i8] zeroinitializer, align 16

declare void @llvm.memcpy.p200.p200.i64(ptr addrspace(200), ptr addrspace(200), i64, i1)
declare void @llvm.memset.p200.i64(ptr addrspace(200), i8, i64, i1)
declare i64 @vararg_callee(i32, ...)

define i64 @callee(i64 %x) {
  %r = add i64 %x, 1
  ret i64 %r
}

define i64 @tail_caller(i64 %x) {
  %r = tail call i64 @callee(i64 %x)
  ret i64 %r
}

define i64 @globals_and_calls(i64 %x) {
  %z = load i32, ptr addrspace(200) @zero32
  %zz = zext i32 %z to i64
  %i = load i64, ptr addrspace(200) @init64
  %p = load ptr addrspace(200), ptr addrspace(200) @pinit
  %pv = load i64, ptr addrspace(200) %p
  %f = load ptr addrspace(200), ptr addrspace(200) @fptr
  %fv = call i64 %f(i64 %x)
  %dv = call i64 @callee(i64 %i)
  %vv = call i64 (i32, ...) @vararg_callee(i32 2, i64 %x, ptr addrspace(200) @str)
  %s1 = add i64 %zz, %pv
  %s2 = add i64 %s1, %fv
  %s3 = add i64 %s2, %dv
  %s4 = add i64 %s3, %vv
  ret i64 %s4
}

define void @copies(ptr addrspace(200) %d, ptr addrspace(200) %s) {
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %d, ptr addrspace(200) align 16 %s, i64 48, i1 false)
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 8 %d, ptr addrspace(200) align 8 %s, i64 44, i1 false)
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 @buf, ptr addrspace(200) align 16 %s, i64 48, i1 false)
  call void @llvm.memset.p200.i64(ptr addrspace(200) align 16 %d, i8 0, i64 48, i1 false)
  ret void
}

define i64 @stack_objects(i64 %n) {
  %a = alloca [4 x i64], addrspace(200)
  %dyn = alloca i8, i64 %n, addrspace(200)
  store i64 %n, ptr addrspace(200) %a
  %r = call i64 @use_ptrs(ptr addrspace(200) %a, ptr addrspace(200) %dyn)
  ret i64 %r
}
declare i64 @use_ptrs(ptr addrspace(200), ptr addrspace(200))

define i64 @wide_switch(i64 %k) {
  switch i64 %k, label %d [
    i64 0, label %c0
    i64 1, label %c1
    i64 2, label %c2
    i64 3, label %c3
    i64 4, label %c4
    i64 5, label %c5
    i64 6, label %c6
    i64 7, label %c7
  ]
c0: ret i64 10
c1: ret i64 11
c2: ret i64 12
c3: ret i64 13
c4: ret i64 14
c5: ret i64 15
c6: ret i64 16
c7: ret i64 17
d:  ret i64 0
}

define i128 @wide_math(i128 %a, i128 %b, i64 %s) {
  %m = mul i128 %a, %b
  %sh = shl i128 %m, 3
  %x = xor i128 %sh, %b
  %sz = zext i64 %s to i128
  %r = lshr i128 %x, %sz
  ret i128 %r
}

define ptr addrspace(200) @cap_select(i1 %c, ptr addrspace(200) %p, ptr addrspace(200) %q) {
  %r = select i1 %c, ptr addrspace(200) %p, ptr addrspace(200) %q
  ret ptr addrspace(200) %r
}
