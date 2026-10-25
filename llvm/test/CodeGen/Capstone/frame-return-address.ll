; C-83: llvm.returnaddress and llvm.frameaddress at every depth, in the three
; profiles. A frame record is two capability-sized slots below the frame
; pointer: the return address at fp - 16 (`sd ra` where calls are integer jumps,
; `stc ra` where they are capability jumps) and the caller's frame pointer, a
; capability, at fp - 32. Depth N therefore walks N saved frame pointers with
; `ldc ..., -32(...)`, and the return address one level up is the integer at
; -16 of that frame (for a spilled capability, its cursor). The walk used to be
; RISC-V's: an integer ADD on the capability frame pointer, which asserted.
;
; Depth 0 reads ra through PseudoTRUNC_CAP, an integer write, so the value is
; never the capability cjalr left in ra. A function that reads it and also
; spills ra must not kill ra at the spill: -verify-machineinstrs on ra0_arg
; (the spill is of C1, the live-in X1).
;
; RUN: llc -mtriple=capstone64 -mattr=+m,+f,+d -target-abi lp64 -verify-machineinstrs \
; RUN:   -capstone-gp-free -capstone-image-gp -O2 < %s | FileCheck %s --check-prefixes=CHECK,INT
; RUN: llc -mtriple=capstone64 -mattr=+m,+f,+d -target-abi lp64 -verify-machineinstrs \
; RUN:   -capstone-gp-free -capstone-image-gp -O0 < %s | FileCheck %s --check-prefixes=CHECK,INT
; RUN: llc -mtriple=capstone64 -mattr=+m,+f,+d -target-abi lp64 -verify-machineinstrs \
; RUN:   -capstone-gp-captable -O2 < %s | FileCheck %s --check-prefixes=CHECK,INT
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs -O2 < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,CAP
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs -O0 < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,CAP
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs -O2 \
; RUN:   -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

declare void @sink(ptr addrspace(200)) addrspace(200)
declare ptr @llvm.returnaddress(i32 immarg) addrspace(200)
declare ptr addrspace(200) @llvm.frameaddress.p200(i32 immarg) addrspace(200)

; CHECK-LABEL: ra0:
; CHECK:       mv a0, ra
define i64 @ra0() addrspace(200) nounwind {
  %a = call ptr @llvm.returnaddress(i32 0)
  %i = ptrtoint ptr %a to i64
  ret i64 %i
}

; The shape clang emits for `sink(__builtin_return_address(0))`.
; CHECK-LABEL: ra0_arg:
; INT:         sd ra, 0(sp)
; CAP:         stc ra, 0(sp)
; CHECK:       mv a0, ra
; INT:         jalr
; CAP:         cjalr ra, 0(
; MIR-LABEL:   name: ra0_arg
; MIR:         [[RA:%[0-9]+]]:gpr = COPY $x1
; MIR:         PseudoTRUNC_CAP [[RA]]
define void @ra0_arg() addrspace(200) nounwind {
  %a = call ptr @llvm.returnaddress(i32 0)
  %i = ptrtoint ptr %a to i64
  %p = inttoptr i64 %i to ptr addrspace(200)
  call addrspace(200) void @sink(ptr addrspace(200) %p)
  ret void
}

; CHECK-LABEL: ra1:
; CHECK:       ldc a0, -32({{s0|a0}})
; CHECK-NEXT:  ld a0, -16(a0)
define i64 @ra1() addrspace(200) nounwind {
  %a = call ptr @llvm.returnaddress(i32 1)
  %i = ptrtoint ptr %a to i64
  ret i64 %i
}

; CHECK-LABEL: ra2:
; CHECK:       ldc a0, -32({{s0|a0}})
; CHECK-NEXT:  ldc a0, -32(a0)
; CHECK-NEXT:  ld a0, -16(a0)
define i64 @ra2() addrspace(200) nounwind {
  %a = call ptr @llvm.returnaddress(i32 2)
  %i = ptrtoint ptr %a to i64
  ret i64 %i
}

; CHECK-LABEL: fa0:
; CHECK:       movc a0, s0
; CHECK-NOT:   ldc a0
; CHECK-LABEL: fa1:
define ptr addrspace(200) @fa0() addrspace(200) nounwind {
  %a = call ptr addrspace(200) @llvm.frameaddress.p200(i32 0)
  ret ptr addrspace(200) %a
}

; CHECK:       ldc a0, -32({{s0|a0}})
; CHECK-NOT:   ldc a0
; CHECK-LABEL: fa2:
define ptr addrspace(200) @fa1() addrspace(200) nounwind {
  %a = call ptr addrspace(200) @llvm.frameaddress.p200(i32 1)
  ret ptr addrspace(200) %a
}

; CHECK:       ldc a0, -32({{s0|a0}})
; CHECK-NEXT:  ldc a0, -32(a0)
define ptr addrspace(200) @fa2() addrspace(200) nounwind {
  %a = call ptr addrspace(200) @llvm.frameaddress.p200(i32 2)
  ret ptr addrspace(200) %a
}
