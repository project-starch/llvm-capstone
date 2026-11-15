; C-69: a load or store of a capability below its 16-byte alignment used to be
; expanded in scalar pieces through a stack slot, which drops the tag even when
; the address is aligned at run time. It now tests the address: aligned, the
; access is an ldc/stc; otherwise the scalar expansion, as before.
;
; @freep is FFmpeg's av_freep after SROA: memcpy(&val, arg, sizeof(val)) is an
; align-1 capability load, so no memcpy is left for C-81 to send to musl.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -stop-after=capstone-misaligned-cap < %s | FileCheck %s --check-prefix=IR
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM0
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-misaligned-cap-dispatch=false -verify-machineinstrs < %s | FileCheck %s --check-prefix=OFF
;
; RUN: rm -f %t.log
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-misaligned-cap-dispatch=false -capstone-misaligned-cap-log=%t.log -o /dev/null < %s
; RUN: FileCheck %s --check-prefix=LOG --input-file=%t.log
;
; The OFF run records the old lowering: no ldc/stc reaches the operand. The
; census is written with the dispatch off, one line per function it would change.
; LOG: capstone-misaligned-cap: {{.*}} freep tested 2 realigned 0 off
; LOG-NEXT: capstone-misaligned-cap: {{.*}} load8 tested 1 realigned 0 off
; LOG-NEXT: capstone-misaligned-cap: {{.*}} store1 tested 1 realigned 0 off
; LOG-NEXT: capstone-misaligned-cap: {{.*}} agg tested 1 realigned 0 off
; LOG-NEXT: capstone-misaligned-cap: {{.*}} known tested 0 realigned 1 off
; LOG-NEXT: capstone-misaligned-cap: {{.*}} at_O0 tested 1 realigned 0 off
; LOG-NOT: controls

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

declare void @av_free(ptr addrspace(200)) addrspace(200)

; IR-LABEL: define void @freep(
; IR: [[WIDE:%[^ ]+]] = ptrtoint ptr addrspace(200) %arg to i128
; IR: [[INT:%[^ ]+]] = trunc i128 [[WIDE]] to i64
; IR: [[LOW:%[^ ]+]] = and i64 [[INT]], 15
; IR: [[OK:%[^ ]+]] = icmp eq i64 [[LOW]], 0
; IR: br i1 [[OK]], label %[[THEN:[^ ,]+]], label %[[ELSE:[^ ,]+]]
; IR: [[THEN]]:
; IR-NEXT: [[CAP:%[^ ]+]] = load ptr addrspace(200), ptr addrspace(200) %arg, align 16
; IR: [[ELSE]]:
; IR-NEXT: [[RAW:%[^ ]+]] = load ptr addrspace(200), ptr addrspace(200) %arg, align 1
; IR: phi ptr addrspace(200) [ [[CAP]], %[[THEN]] ], [ [[RAW]], %[[ELSE]] ]
; IR: ptrtoint ptr addrspace(200) %arg to i128
; IR: and i64 {{%[^ ]+}}, 15
; IR: store ptr addrspace(200) null, ptr addrspace(200) %arg, align 16
; IR: store ptr addrspace(200) null, ptr addrspace(200) %arg, align 1
; IR: call addrspace(200) void @av_free(
;
; ASM-LABEL: freep:
; ASM-DAG: andi {{[a-z0-9]+}}, {{[a-z0-9]+}}, 15
; ASM-DAG: ldc {{[a-z0-9]+}}, 0(a0)
; ASM-DAG: stc zero, 0(a0)
; ASM: .Lfunc_end0:
;
; At -O0 the operands go through stack slots, so ASM0 checks the test and a
; capability access through an argument register (not sp) in each function.
; ASM0-LABEL: freep:
; ASM0: andi {{a[0-9]}}, {{a[0-9]}}, 15
; ASM0: ldc {{[a-z0-9]+}}, 0({{a[0-9]}})
; ASM0: andi {{a[0-9]}}, {{a[0-9]}}, 15
; ASM0: stc {{[a-z0-9]+}}, 0({{a[0-9]}})
; ASM0: .Lfunc_end0:
; ASM0-LABEL: load8:
; ASM0: andi {{a[0-9]}}, {{a[0-9]}}, 15
; ASM0: ldc {{[a-z0-9]+}}, 0({{a[0-9]}})
; ASM0: .Lfunc_end1:
; ASM0-LABEL: store1:
; ASM0: andi {{a[0-9]}}, {{a[0-9]}}, 15
; ASM0: stc {{[a-z0-9]+}}, 0({{a[0-9]}})
; ASM0: .Lfunc_end2:
; ASM0-LABEL: agg:
; ASM0: andi {{a[0-9]}}, {{a[0-9]}}, 15
; ASM0: ldc {{[a-z0-9]+}}, 16({{a[0-9]}})
; ASM0: .Lfunc_end3:
;
; OFF-LABEL: freep:
; OFF-NOT: ldc {{[a-z0-9]+}}, 0(a0)
; OFF-NOT: stc {{[a-z0-9]+}}, 0(a0)
; OFF: .Lfunc_end0:
define void @freep(ptr addrspace(200) %arg) addrspace(200) {
entry:
  %val = load ptr addrspace(200), ptr addrspace(200) %arg, align 1
  store ptr addrspace(200) null, ptr addrspace(200) %arg, align 1
  call addrspace(200) void @av_free(ptr addrspace(200) %val)
  ret void
}

; An 8-aligned load (a 16-byte memcpy between 8-aligned fields, after SROA).
; IR-LABEL: define ptr addrspace(200) @load8(
; IR: load ptr addrspace(200), ptr addrspace(200) %p, align 16
; IR: load ptr addrspace(200), ptr addrspace(200) %p, align 8
;
; ASM-LABEL: load8:
; ASM: ldc {{[a-z0-9]+}}, 0(a0)
; ASM: .Lfunc_end1:
;
; OFF-LABEL: load8:
; OFF-NOT: ldc {{[a-z0-9]+}}, 0(a0)
; OFF: .Lfunc_end1:
define ptr addrspace(200) @load8(ptr addrspace(200) %p) addrspace(200) {
entry:
  %v = load ptr addrspace(200), ptr addrspace(200) %p, align 8
  ret ptr addrspace(200) %v
}

; A store of a capability argument; the aligned arm keeps its tag.
; IR-LABEL: define void @store1(
; IR: store ptr addrspace(200) %v, ptr addrspace(200) %p, align 16
; IR: store ptr addrspace(200) %v, ptr addrspace(200) %p, align 1
;
; ASM-LABEL: store1:
; ASM: stc a1, 0(a0)
; ASM: .Lfunc_end2:
;
; OFF-LABEL: store1:
; OFF-NOT: stc a1, 0(a0)
; OFF: .Lfunc_end2:
define void @store1(ptr addrspace(200) %p, ptr addrspace(200) %v) addrspace(200) {
entry:
  store ptr addrspace(200) %v, ptr addrspace(200) %p, align 1
  ret void
}

; An aggregate holding a capability: the aligned arm loads the whole struct at
; align 16, so the member at offset 16 is an ldc.
; IR-LABEL: define ptr addrspace(200) @agg(
; IR: load { i64, ptr addrspace(200) }, ptr addrspace(200) %p, align 16
; IR: load { i64, ptr addrspace(200) }, ptr addrspace(200) %p, align 8
;
; ASM-LABEL: agg:
; ASM: ldc {{[a-z0-9]+}}, 16(a0)
; ASM: .Lfunc_end3:
;
; OFF-LABEL: agg:
; OFF-NOT: ldc {{[a-z0-9]+}}, 16(a0)
; OFF: .Lfunc_end3:
define ptr addrspace(200) @agg(ptr addrspace(200) %p) addrspace(200) {
entry:
  %s = load { i64, ptr addrspace(200) }, ptr addrspace(200) %p, align 8
  %v = extractvalue { i64, ptr addrspace(200) } %s, 1
  ret ptr addrspace(200) %v
}

; A provably aligned address is re-aligned in place, with no test.
; IR-LABEL: define ptr addrspace(200) @known(
; IR-NOT: ptrtoint
; IR: load ptr addrspace(200), ptr addrspace(200) %p, align 16
; IR-NOT: ptrtoint
; IR: ret ptr addrspace(200)
define ptr addrspace(200) @known(ptr addrspace(200) align 16 %p) addrspace(200) {
entry:
  %v = load ptr addrspace(200), ptr addrspace(200) %p, align 1
  ret ptr addrspace(200) %v
}

; clang -O0 marks every function optnone; the pass must not skip it.
; IR-LABEL: define ptr addrspace(200) @at_O0(
; IR: and i64 {{%[^ ]+}}, 15
; IR: load ptr addrspace(200), ptr addrspace(200) %p, align 16
; IR: load ptr addrspace(200), ptr addrspace(200) %p, align 1
;
; ASM-LABEL: at_O0:
; ASM: andi {{[a-z0-9]+}}, {{[a-z0-9]+}}, 15
; ASM: ldc {{[a-z0-9]+}}, 0({{a[0-9]}})
; ASM: .Lfunc_end5:
define ptr addrspace(200) @at_O0(ptr addrspace(200) %p) addrspace(200) noinline optnone {
entry:
  %v = load ptr addrspace(200), ptr addrspace(200) %p, align 1
  ret ptr addrspace(200) %v
}

; Controls: an under-aligned integer access, and an aligned capability access,
; are left alone.
; IR-LABEL: define i64 @controls(
; IR-NOT: ptrtoint
; IR: load i64, ptr addrspace(200) %p, align 1
; IR: load ptr addrspace(200), ptr addrspace(200) %q, align 16
; IR-NOT: ptrtoint
; IR: ret i64
define i64 @controls(ptr addrspace(200) %p, ptr addrspace(200) %q, ptr addrspace(200) %out) addrspace(200) {
entry:
  %a = load i64, ptr addrspace(200) %p, align 1
  %c = load ptr addrspace(200), ptr addrspace(200) %q, align 16
  store ptr addrspace(200) %c, ptr addrspace(200) %out, align 16
  ret i64 %a
}
