; C-54: atomics whose VALUE is a capability go to the generic libatomic calls,
; which pass values through memory, never to the sized __atomic_*_16 calls,
; which pass them in integer registers and drop the tag. An i128 atomic keeps
; the sized call: the control at the end.
; RUN: llc -mtriple=capstone64 -mattr=+a -verify-machineinstrs < %s | FileCheck %s
; RUN: %llc_cap -mattr=+a -O0 < %s -o /dev/null
; RUN: %llc_cap -mattr=+a -O1 < %s -o /dev/null

; Each positive pattern ends in ')', so it cannot match the sized __atomic_*_16
; name, and CHECK-LABEL confines it to its function. The generic call takes the
; size (16) in a0, and the OBJECT POINTER as a capability: AtomicExpand used to
; cast it to address space 0, which passed a bare address (`mv a1, a0`) that
; the callee could not dereference. The result comes back through memory (ldc).
; MUTATION: the same output with `movc a1, a0` replaced by `mv a1, a0` fails
; this check (performed 2026-09-23).
; CHECK-LABEL: ld:
; CHECK-DAG: movc a1, a0
; CHECK-DAG: li a0, 16
; CHECK-DAG: %pcrel_hi(__atomic_load)
; CHECK: cjalr ra,
; CHECK-NEXT: ldc a0, 0(
define ptr addrspace(200) @ld(ptr addrspace(200) %p) addrspace(200) {
  %v = load atomic ptr addrspace(200), ptr addrspace(200) %p seq_cst, align 16
  ret ptr addrspace(200) %v
}

; CHECK-LABEL: st:
; CHECK: %pcrel_hi(__atomic_store)
define void @st(ptr addrspace(200) %p, ptr addrspace(200) %v) addrspace(200) {
  store atomic ptr addrspace(200) %v, ptr addrspace(200) %p seq_cst, align 16
  ret void
}

; CHECK-LABEL: xchg:
; CHECK: %pcrel_hi(__atomic_exchange)
define ptr addrspace(200) @xchg(ptr addrspace(200) %p, ptr addrspace(200) %v) addrspace(200) {
  %old = atomicrmw xchg ptr addrspace(200) %p, ptr addrspace(200) %v seq_cst, align 16
  ret ptr addrspace(200) %old
}

; CHECK-LABEL: cas:
; CHECK: %pcrel_hi(__atomic_compare_exchange)
define i1 @cas(ptr addrspace(200) %p, ptr addrspace(200) %e, ptr addrspace(200) %d) addrspace(200) {
  %r = cmpxchg ptr addrspace(200) %p, ptr addrspace(200) %e, ptr addrspace(200) %d seq_cst seq_cst, align 16
  %ok = extractvalue { ptr addrspace(200), i1 } %r, 1
  ret i1 %ok
}

; Control: an i128 value is an integer and keeps the sized call.
; CHECK-LABEL: ld_i128:
; CHECK: %pcrel_hi(__atomic_load_16)
define i128 @ld_i128(ptr addrspace(200) %p) addrspace(200) {
  %v = load atomic i128, ptr addrspace(200) %p seq_cst, align 16
  ret i128 %v
}

