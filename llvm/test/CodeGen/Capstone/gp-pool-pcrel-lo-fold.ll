; RUN: llc -mtriple=capstone64-unknown-elf -mattr=+m,+a,+f,+d -target-abi lp64 \
; RUN:   -capstone-gp-free -capstone-image-gp -verify-machineinstrs < %s \
; RUN:   | FileCheck %s
; RUN: llc -mtriple=capstone64-unknown-elf -mattr=+m,+a,+f,+d -target-abi lp64 \
; RUN:   -verify-machineinstrs < %s | FileCheck --check-prefix=DEFAULT %s
;
; Under the image-gp ABI a constant-pool load reached through the gp-derived
; capability takes the %pcrel_lo in the access: auipc; scc gp; fld %pcrel_lo,
; not auipc; addi; scc gp; fld 0. The capability's cursor stops at the auipc
; result and only the access lands on the entry; only the access is checked.
; Anything that uses the capability itself (a global narrowed to its bounds,
; here) keeps the completed address, and so does the default ABI.

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; CHECK-LABEL: addpi:
; CHECK:       .Lpcrel_hi[[N:[0-9]+]]:
; CHECK-NEXT:  auipc [[HI:a[0-9]+]], %pcrel_hi(.LCPI0_0)
; CHECK-NOT:   addi
; CHECK:       scc [[CAP:a[0-9]+]], gp, [[HI]]
; CHECK-NEXT:  fld {{f[a-z0-9]+}}, %pcrel_lo(.Lpcrel_hi[[N]])([[CAP]])
; DEFAULT-LABEL: addpi:
; DEFAULT:     addi {{a[0-9]+}}, {{a[0-9]+}}, %pcrel_lo
; DEFAULT:     fld {{f[a-z0-9]+}}, 0(
define double @addpi(double %x) addrspace(200) {
  %r = fadd double %x, 0x400921FB54442D18
  ret double %r
}

; CHECK-LABEL: addpif:
; CHECK:       scc [[CAP:a[0-9]+]], gp,
; CHECK-NEXT:  flw {{f[a-z0-9]+}}, %pcrel_lo(.Lpcrel_hi{{[0-9]+}})([[CAP]])
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
