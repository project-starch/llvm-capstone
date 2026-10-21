; -capstone-pad-globals' frame rule (CapstonePadGlobals.cpp): a static alloca of
; 4096 bytes or more becomes [length(size) x i8] at its representable alignment,
; so the frame object's SHRINK narrows to exactly representable bounds. Lengths
; and alignments from MCTargetDesc/CapstoneRepresentable.h: 4097 -> 4104 (granule
; 8), 20001 -> 20032 (granule 32, above the stack's 16: the frame realigns).
;
; MUTATION: drop the alloca loop in CapstonePadGlobals.cpp -> PAD's SHRINK
; lengths read 4097 and 20001 and @big has no `andi ..., -32`.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-gp-free -verify-machineinstrs \
; RUN:   -capstone-pad-globals < %s | FileCheck %s --check-prefix=PAD
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-gp-free -verify-machineinstrs \
; RUN:   < %s | FileCheck %s --check-prefix=OFF

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

declare void @use(ptr addrspace(200))

; A PATH_MAX + 1 buffer: 4097 is not a multiple of the granule 8.
; The SHRINK's end is base + length: `lui 1; addi 8` is 4104.
; PAD-LABEL: path:
; PAD:       lui [[H:a[0-9]+]], 1{{$}}
; PAD:       addi [[H]], [[H]], 8{{$}}
; PAD:       add [[E:a[0-9]+]], [[B:a[0-9]+]], [[H]]
; PAD-NEXT:  shrink a0, [[B]], [[E]]
; OFF-LABEL: path:
; OFF:       lui [[H:a[0-9]+]], 1{{$}}
; OFF:       addi [[H]], [[H]], 1{{$}}
; OFF:       add [[E:a[0-9]+]], [[B:a[0-9]+]], [[H]]
; OFF-NEXT:  shrink a0, [[B]], [[E]]
define void @path() {
  %b = alloca [4097 x i8], align 1, addrspace(200)
  call void @use(ptr addrspace(200) %b)
  ret void
}

; 20001 -> 20032 = 5 << 12 - 448, at alignment 32: the frame realigns.
; PAD-LABEL: big:
; PAD:       andi {{a[0-9]+}}, {{a[0-9]+}}, -32
; PAD:       lui [[H:a[0-9]+]], 5{{$}}
; PAD:       addi [[H]], [[H]], -448{{$}}
; PAD:       add [[E:a[0-9]+]], [[B:a[0-9]+]], [[H]]
; PAD-NEXT:  shrink a0, [[B]], [[E]]
; OFF-LABEL: big:
; OFF-NOT:   andi
; OFF:       lui [[H:a[0-9]+]], 5{{$}}
; OFF:       addi [[H]], [[H]], -479{{$}}
; OFF:       add [[E:a[0-9]+]], [[B:a[0-9]+]], [[H]]
; OFF-NEXT:  shrink a0, [[B]], [[E]]
define void @big() {
  %b = alloca [20001 x i8], align 1, addrspace(200)
  call void @use(ptr addrspace(200) %b)
  ret void
}

; CONTROL: below 4096 every length is exact; nothing changes.
; PAD-LABEL: small:
; PAD:       lui [[H:a[0-9]+]], 1{{$}}
; PAD:       addi [[H]], [[H]], -1{{$}}
; PAD:       add [[E:a[0-9]+]], [[B:a[0-9]+]], [[H]]
; PAD-NEXT:  shrink a0, [[B]], [[E]]
define void @small() {
  %b = alloca [4095 x i8], align 1, addrspace(200)
  call void @use(ptr addrspace(200) %b)
  ret void
}
