; SHRINKTO for small data globals (-capstone-shrinkto-globals, default off).
; A sized data global whose size plus -capstone-shrinkto-end-adjust fits a
; 12-bit immediate is narrowed with one `shrinkto cap, cap, size+adjust`
; instead of li + add + shrink; a larger one keeps SHRINK. The adjustment is 1
; on bitstreams whose SHRINKTO ends at cursor + imm - 1 (C.5a and earlier).
;
; MUTATION: drop the `Size + CapstoneShrinkToEndAdjust <= 2047` test -> @big
; selects shrinkto with an out-of-range immediate and the BIG checks fire.
; RUN: llc -mtriple=capstone64 -mattr=+m < %s | FileCheck %s --check-prefixes=CHECK,OFF
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-shrinkto-globals < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,ON0
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-shrinkto-globals \
; RUN:   -capstone-shrinkto-end-adjust=1 < %s | FileCheck %s --check-prefixes=CHECK,ON1
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-shrinkto-globals \
; RUN:   -capstone-shrinkto-end-adjust=1 -filetype=obj < %s -o %t.o
; RUN: llvm-objdump -d --triple=capstone64 %t.o | FileCheck %s --check-prefix=OBJ

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

@g = addrspace(200) global [16 x i8] zeroinitializer, align 1
@edge = addrspace(200) global [2046 x i8] zeroinitializer, align 1
@big = addrspace(200) global [2047 x i8] zeroinitializer, align 1

; CHECK-LABEL: load_small:
; OFF: shrink
; OFF-NOT: shrinkto
; ON0: shrinkto {{a[0-9]+}}, {{a[0-9]+}}, 16
; ON1: shrinkto {{a[0-9]+}}, {{a[0-9]+}}, 17
; ON0-NOT: {{^[[:space:]]+shrink[[:space:]]}}
; OBJ-LABEL: <load_small>:
; OBJ: shrinkto {{a[0-9]+}}, {{a[0-9]+}}, 0x11
define i8 @load_small() addrspace(200) {
  %p = getelementptr [16 x i8], ptr addrspace(200) @g, i64 0, i64 3
  %v = load i8, ptr addrspace(200) %p
  ret i8 %v
}

; 2046 + 1 = 2047 is the largest immediate: still SHRINKTO with the adjustment.
; CHECK-LABEL: load_edge:
; ON1: shrinkto {{a[0-9]+}}, {{a[0-9]+}}, 2047
define i8 @load_edge() addrspace(200) {
  %p = getelementptr [2046 x i8], ptr addrspace(200) @edge, i64 0, i64 3
  %v = load i8, ptr addrspace(200) %p
  ret i8 %v
}

; 2047 + 1 does not fit: SHRINK.
; CHECK-LABEL: load_big:
; ON1-NOT: shrinkto
; ON1: shrink {{a[0-9]+}}
define i8 @load_big() addrspace(200) {
  %p = getelementptr [2047 x i8], ptr addrspace(200) @big, i64 0, i64 3
  %v = load i8, ptr addrspace(200) %p
  ret i8 %v
}
