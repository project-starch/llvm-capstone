; -capstone-pad-globals (default off): every DEFINED data global of 4096 bytes or
; more is aligned and tail-padded to exactly representable bounds, and every
; global is narrowed to that length (CapstonePadGlobals.cpp; the representable
; length/alignment helper is MCTargetDesc/CapstoneRepresentable.h, swept against
; QEMU's codec and the RTL's bounds model by tests/capstone-codec-padding).
;
; @big    0x80f00 bytes -> 0x81000, alignment 1 KiB (granule 2^(19-9)).
; @fix    8191 bytes    -> 8192, alignment 16, NOT 8: the rounded length reaches
;                          2^13, whose granule is twice 8191's (the fixpoint).
; @edge   4095 bytes    -> unchanged: below 4096 every length is exact.
; @aligned 0x10000      -> length unchanged, alignment raised to 128.
; @init   20004 bytes, non-zero -> its data, then 28 zero bytes, alignment 32.
; @sect   8191 bytes in an explicit section -> never padded (linker sets), and
;         narrowed to its own 8191 bytes.
; @ext    declared only, 0x80f00 bytes -> narrowed to 0x81000 like its definer.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-pad-globals < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,PAD
; RUN: llc -mtriple=capstone64 -mattr=+m < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NOPAD
; RUN: %llc_cap -O0 -capstone-pad-globals < %s -o /dev/null

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

@big = addrspace(200) global [528128 x i8] zeroinitializer, align 16
@fix = addrspace(200) global [8191 x i8] zeroinitializer, align 1
@edge = addrspace(200) global [4095 x i8] zeroinitializer, align 1
@aligned = addrspace(200) global [65536 x i8] zeroinitializer, align 16
@init = addrspace(200) global { i32, [5000 x i32] } { i32 7, [5000 x i32] zeroinitializer }, align 4
@sect = addrspace(200) global [8191 x i8] zeroinitializer, section "capstone_set", align 1
@tls = thread_local addrspace(200) global [8191 x i8] zeroinitializer, align 1
@ext = external addrspace(200) global [528128 x i8]

; CHECK-LABEL: load_big:
; PAD-NOT: -256
; PAD: lui {{a[0-9]+}}, 129
; PAD-NOT: -256
; NOPAD: addi {{a[0-9]+}}, {{a[0-9]+}}, -256
; CHECK: shrink
define i8 @load_big() addrspace(200) {
  %v = load i8, ptr addrspace(200) @big
  ret i8 %v
}

; The fixpoint: 8191 pads to 8192 (lui 2), not to a multiple of 8191's granule.
; CHECK-LABEL: load_fix:
; PAD: lui {{a[0-9]+}}, 2
; PAD-NOT: , -1{{$}}
; NOPAD: addi {{a[0-9]+}}, {{a[0-9]+}}, -1{{$}}
; CHECK: shrink
define i8 @load_fix() addrspace(200) {
  %v = load i8, ptr addrspace(200) @fix
  ret i8 %v
}

; Below 4096: unchanged either way.
; CHECK-LABEL: load_edge:
; CHECK: addi {{a[0-9]+}}, {{a[0-9]+}}, -1{{$}}
; CHECK: shrink
define i8 @load_edge() addrspace(200) {
  %v = load i8, ptr addrspace(200) @edge
  ret i8 %v
}

; CHECK-LABEL: load_aligned:
; CHECK: lui {{a[0-9]+}}, 16
; CHECK: shrink
define i8 @load_aligned() addrspace(200) {
  %v = load i8, ptr addrspace(200) @aligned
  ret i8 %v
}

; CHECK-LABEL: load_init:
; PAD: addi {{a[0-9]+}}, {{a[0-9]+}}, -448
; NOPAD: addi {{a[0-9]+}}, {{a[0-9]+}}, -476
; CHECK: shrink
define i32 @load_init() addrspace(200) {
  %v = load i32, ptr addrspace(200) @init
  ret i32 %v
}

; An explicit-section global is neither padded nor narrowed past itself.
; CHECK-LABEL: load_sect:
; CHECK: addi {{a[0-9]+}}, {{a[0-9]+}}, -1{{$}}
; CHECK: shrink
define i8 @load_sect() addrspace(200) {
  %v = load i8, ptr addrspace(200) @sect
  ret i8 %v
}

; A module that only declares @ext narrows it to the length its definer pads to.
; CHECK-LABEL: load_ext:
; PAD-NOT: -256
; PAD: lui {{a[0-9]+}}, 129
; PAD-NOT: -256
; NOPAD: addi {{a[0-9]+}}, {{a[0-9]+}}, -256
; CHECK: shrink
define i8 @load_ext() addrspace(200) {
  %v = load i8, ptr addrspace(200) @ext
  ret i8 %v
}

; The definitions.
; CHECK: .globl big
; PAD-NEXT: .p2align 10, 0x0
; NOPAD-NEXT: .p2align 4, 0x0
; CHECK-NEXT: {{^}}big:
; PAD-NEXT: .zero 528384
; NOPAD-NEXT: .zero 528128
; CHECK: .globl fix
; PAD-NEXT: .p2align 4, 0x0
; CHECK-NEXT: {{^}}fix:
; PAD-NEXT: .zero 8192
; NOPAD-NEXT: .zero 8191
; CHECK: .globl edge
; CHECK-NEXT: {{^}}edge:
; CHECK-NEXT: .zero 4095
; CHECK: .globl aligned
; PAD-NEXT: .p2align 7, 0x0
; NOPAD-NEXT: .p2align 4, 0x0
; CHECK-NEXT: {{^}}aligned:
; CHECK-NEXT: .zero 65536
; CHECK: .globl init
; PAD-NEXT: .p2align 5, 0x0
; NOPAD-NEXT: .p2align 2, 0x0
; CHECK-NEXT: {{^}}init:
; CHECK-NEXT: .word 7
; CHECK-NEXT: .zero 20000
; PAD-NEXT: .zero 28
; PAD-NEXT: .size init, 20032
; NOPAD-NEXT: .size init, 20004
; CHECK: .section capstone_set
; CHECK: {{^}}sect:
; CHECK-NEXT: .zero 8191
; CHECK: {{^}}tls:
; CHECK-NEXT: .zero 8191
