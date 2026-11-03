; RUN: llc -mtriple=capstone64-unknown-elf -mattr=+m,+a,+f,+d -target-abi lp64 \
; RUN:   -capstone-gp-free -capstone-image-gp -verify-machineinstrs < %s \
; RUN:   | FileCheck %s
; RUN: llc -mtriple=capstone64-unknown-elf -mattr=+m,+a,+f,+d -target-abi lp64 \
; RUN:   -verify-machineinstrs < %s | FileCheck --check-prefix=DEFAULT %s
;
; Under the image-gp ABI a constant-pool load reached through the gp-derived
; capability does NOT take the %pcrel_lo in the access: auipc; addi; scc gp;
; fld 0, never auipc; scc gp; fld %pcrel_lo. The fold would set the
; capability's cursor to the auipc's partial address, up to 2048 bytes above
; the entry (low part -16 with the entry at 2^38-16 makes it 2^38, refused
; under sv39 and written untagged), and the in-bounds load would fault. The
; capability's cursor is the entry's own address, under both ABIs.

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; CHECK-LABEL: addpi:
; CHECK:       .Lpcrel_hi[[N:[0-9]+]]:
; CHECK-NEXT:  auipc [[HI:a[0-9]+]], %pcrel_hi(.LCPI0_0)
; CHECK-NEXT:  addi [[LO:a[0-9]+]], [[HI]], %pcrel_lo(.Lpcrel_hi[[N]])
; CHECK-NEXT:  scc [[CAP:a[0-9]+]], gp, [[LO]]
; CHECK-NEXT:  fld {{f[a-z0-9]+}}, 0([[CAP]])
; DEFAULT-LABEL: addpi:
; DEFAULT:     addi {{a[0-9]+}}, {{a[0-9]+}}, %pcrel_lo
; DEFAULT:     fld {{f[a-z0-9]+}}, 0(
define double @addpi(double %x) addrspace(200) {
  %r = fadd double %x, 0x400921FB54442D18
  ret double %r
}

; CHECK-LABEL: addpif:
; CHECK:       addi [[LO:a[0-9]+]], {{a[0-9]+}}, %pcrel_lo
; CHECK-NEXT:  scc [[CAP:a[0-9]+]], gp, [[LO]]
; CHECK-NEXT:  flw {{f[a-z0-9]+}}, 0([[CAP]])
define float @addpif(float %x) addrspace(200) {
  %r = fadd float %x, 0x400921FB60000000
  ret float %r
}

; A named global is narrowed to its bounds from the completed address: no fold.
@g = dso_local addrspace(200) global double 0.0, align 8

; CHECK-LABEL: loadg:
; CHECK:       addi {{a[0-9]+}}, {{a[0-9]+}}, %pcrel_lo
; CHECK:       shrink
; CHECK:       ld {{a[0-9]+}}, 0(
define double @loadg() addrspace(200) {
  %v = load double, ptr addrspace(200) @g, align 8
  ret double %v
}
