; -capstone-direct-calls (default off): under the gp-free ABI a direct call to a
; function of the image, a tail call and a libcall are `call`/`tail`
; (auipc + jalr, R_Capstone_CALL_PLT) instead of materialising the callee's
; address for an indirect jalr. An extern-weak callee keeps the indirect form:
; it may be undefined, and only the materialised address reaches it as 0.
;
; MUTATION: drop the hasExternalWeakLinkage test in selectCall -> `call weak`
; and the weak auipc check fires; flip the option's default -> the OFF checks fire.
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-free -capstone-image-gp \
; RUN:   < %s | FileCheck %s --check-prefixes=CHECK,OFF
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-free -capstone-image-gp \
; RUN:   -capstone-direct-calls < %s | FileCheck %s --check-prefixes=CHECK,ON
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-free -capstone-image-gp \
; RUN:   -capstone-direct-calls -filetype=obj < %s -o %t.o
; RUN: llvm-readelf -r %t.o | FileCheck %s --check-prefix=RELOC

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

declare i32 @ext(i32) addrspace(200)
declare extern_weak i32 @weak(i32) addrspace(200)

define internal i32 @local(i32 %x) addrspace(200) noinline {
  %r = add i32 %x, 1
  ret i32 %r
}

; CHECK-LABEL: calls:
; OFF:      auipc [[R1:[a-z0-9]+]], %pcrel_hi(local)
; OFF:      jalr
; OFF-NOT:  {{^[[:space:]]+call[[:space:]]}}
; OFF:      auipc [[R2:[a-z0-9]+]], %pcrel_hi(ext)
; OFF:      jalr
; ON-NOT:   %pcrel_hi(local)
; ON:       {{^[[:space:]]+call[[:space:]]+local$}}
; ON-NOT:   %pcrel_hi(ext)
; ON:       {{^[[:space:]]+call[[:space:]]+ext$}}
; CHECK:    auipc [[W:[a-z0-9]+]], %pcrel_hi(weak)
; CHECK:    jalr
define i32 @calls(i32 %x) addrspace(200) {
  %a = call i32 @local(i32 %x)
  %b = call i32 @ext(i32 %a)
  %c = call i32 @weak(i32 %b)
  ret i32 %c
}

; CHECK-LABEL: tail:
; OFF:     auipc {{[a-z0-9]+}}, %pcrel_hi(ext)
; OFF:     jr
; ON-NOT:  %pcrel_hi(ext)
; ON:      {{^[[:space:]]+tail[[:space:]]+ext$}}
define i32 @tail(i32 %x) addrspace(200) {
  %a = tail call i32 @ext(i32 %x)
  ret i32 %a
}

; CHECK-LABEL: div:
; OFF:     auipc {{[a-z0-9]+}}, %pcrel_hi(__udivti3)
; ON-NOT:  %pcrel_hi(__udivti3)
; ON:      {{^[[:space:]]+call[[:space:]]+__udivti3$}}
define i128 @div(i128 %a, i128 %b) addrspace(200) {
  %r = udiv i128 %a, %b
  ret i128 %r
}

; RELOC-DAG: R_Capstone_CALL_PLT {{.*}} ext + 0
; RELOC-DAG: R_Capstone_CALL_PLT {{.*}} __udivti3 + 0
; RELOC-NOT: R_Capstone_CALL_PLT {{.*}} weak
