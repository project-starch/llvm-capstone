; Thread-local storage in a capability domain (C-47).
;
; A domain is one static image, so every thread-local is local-exec whatever
; model the front end asks for: initial-exec for an external one, general- or
; local-dynamic under PIC, emulated TLS. The address is tp's CAPABILITY advanced
; by the %tprel offset built as an integer. RISC-V's own sequence ADDs tp as an
; integer, which keeps the address and drops the tag; isel used to fail on it
; outright ("Cannot select: c128 = GlobalTLSAddress").
;
; Under -capstone-shrink-globals (the default) the result is then narrowed to
; the variable, as a sized global's is.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,SHRINK
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,SHRINK
; Under PIC a dso_local variable is named through its .L<name>$local alias.
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -relocation-model=pic < %s | FileCheck %s --check-prefixes=CHECK,SHRINK
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -emulated-tls < %s | FileCheck %s --check-prefixes=CHECK,SHRINK
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-shrink-globals=false < %s | FileCheck %s --check-prefixes=CHECK,NOSHRINK
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -filetype=obj < %s | llvm-readobj -r - | FileCheck %s --check-prefix=RELOC

@counter = dso_local thread_local addrspace(200) global i32 41, align 4
@zeroed = dso_local thread_local addrspace(200) global [100 x i8] zeroinitializer, align 64
@other = external thread_local addrspace(200) global i32
@ie = external thread_local(initialexec) addrspace(200) global i32
@gd = thread_local(localdynamic) addrspace(200) global i64 5, align 8

; CHECK-LABEL: get_counter:
; CHECK:       lui [[HI:[a-z0-9]+]], %tprel_hi({{(\.L)?}}counter{{(\$local)?}})
; CHECK-NEXT:  addi [[OFF:[a-z0-9]+]], [[HI]], %tprel_lo({{(\.L)?}}counter{{(\$local)?}})
; CHECK-NEXT:  cincoffset [[P:[a-z0-9]+]], tp, [[OFF]]
; SHRINK:      shrink
; NOSHRINK-NOT: shrink
; CHECK:       lw {{[a-z0-9]+}}, 0(
; CHECK-NOT:   add {{.*}}tp
; CHECK:       cjalr zero, 0(ra)
define i32 @get_counter() nounwind {
  %v = load i32, ptr addrspace(200) @counter
  ret i32 %v
}

; The narrowed capability is [cursor, cursor + 100): the size is the constant.
; CHECK-LABEL: addr_zeroed:
; CHECK:       %tprel_hi({{(\.L)?}}zeroed{{(\$local)?}})
; CHECK:       cincoffset {{[a-z0-9]+}}, tp,
; SHRINK:      addi {{[a-z0-9]+}}, {{[a-z0-9]+}}, 100
; SHRINK:      shrink
; CHECK:       cjalr zero, 0(ra)
define ptr addrspace(200) @addr_zeroed() nounwind {
  ret ptr addrspace(200) @zeroed
}

; An external thread-local: no GOT load (la.tls.ie), no __tls_get_addr, and the
; linker resolves its %tprel like any other in the one static image.
; CHECK-LABEL: get_other:
; CHECK-NOT:   la.tls
; CHECK-NOT:   __tls_get_addr
; CHECK-NOT:   __emutls
; CHECK:       lui {{[a-z0-9]+}}, %tprel_hi(other)
; CHECK:       cincoffset {{[a-z0-9]+}}, tp,
; CHECK:       cjalr zero, 0(ra)
define i32 @get_other() nounwind {
  %v = load i32, ptr addrspace(200) @other
  ret i32 %v
}

; CHECK-LABEL: get_ie:
; CHECK-NOT:   la.tls
; CHECK:       %tprel_hi(ie)
; CHECK:       cincoffset {{[a-z0-9]+}}, tp,
define i32 @get_ie() nounwind {
  %v = load i32, ptr addrspace(200) @ie
  ret i32 %v
}

; CHECK-LABEL: set_gd:
; CHECK-NOT:   __tls_get_addr
; CHECK:       %tprel_hi(gd)
; CHECK:       cincoffset {{[a-z0-9]+}}, tp,
; CHECK:       sd
define void @set_gd(i64 %x) nounwind {
  store i64 %x, ptr addrspace(200) @gd
  ret void
}

; RELOC: R_Capstone_TPREL_HI20 counter
; RELOC: R_Capstone_TPREL_LO12_I counter
; RELOC-NOT: R_Capstone_TLS_
