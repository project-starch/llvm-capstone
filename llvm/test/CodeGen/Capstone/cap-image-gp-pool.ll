; An image-wide readable gp can reach anonymous constants. The cap-table ABI
; still rejects them: its gp is bounded to slots, not to the loaded image.
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-free -capstone-image-gp -capstone-gpfree-constant-pools < %s | FileCheck %s
; RUN: not llc -mtriple=capstone64 -mattr=+m -capstone-gp-free -capstone-gpfree-constant-pools < %s 2>&1 | FileCheck %s --check-prefix=REJECT
; RUN: not llc -mtriple=capstone64 -mattr=+m -capstone-gp-captable -capstone-image-gp -capstone-gpfree-constant-pools < %s 2>&1 | FileCheck %s --check-prefix=REJECT

; CHECK: .LCPI0_0:
; CHECK: .quad -3750763034362895579
; CHECK-LABEL: big:
; CHECK: scc {{.*}}, gp,
; CHECK: ld
; CHECK: ret
; REJECT: constant-pool data has no cap-table slot
define i64 @big() {
  ret i64 -3750763034362895579
}
