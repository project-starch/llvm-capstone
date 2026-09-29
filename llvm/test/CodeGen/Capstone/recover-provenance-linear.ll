; Unknown pointer origins may hold LINEAR capabilities. Reading an address
; does not consume one; recovering a capability from that address must not
; introduce a second consuming use, even for an identity round trip.
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -stop-after=capstone-provenance < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -stop-after=capstone-provenance < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s -o %t.on
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs -capstone-recover-provenance=false < %s -o %t.off
; RUN: diff %t.on %t.off
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -verify-machineinstrs < %s -o %t.on
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -verify-machineinstrs -capstone-recover-provenance=false < %s -o %t.off
; RUN: diff %t.on %t.off

declare void @log(ptr addrspace(200))
declare ptr addrspace(200) @get_cap()
declare ptr addrspace(200) @llvm.capstone.cap.mrev.p200(ptr addrspace(200))
declare ptr addrspace(200) @llvm.capstone.cap.delin.p200(ptr addrspace(200))
@saved = internal addrspace(200) global ptr addrspace(200) null

; The third argument can be the live LINEAR scratch capability passed by
; start-fpga-nogp.S. Neither nonnull nor an earlier access proves NONLIN.
; CHECK-LABEL: @linear_argument(
; CHECK: %r = inttoptr i64 %m to ptr addrspace(200)
; CHECK: call addrspace(200) void @log(ptr addrspace(200) %r)
; CHECK: store ptr addrspace(200) %cap, ptr addrspace(200) %slot
define void @linear_argument(ptr addrspace(200) %slot, i64 %func,
                            ptr addrspace(200) nonnull %cap) {
  store volatile i8 0, ptr addrspace(200) %cap
  %i = ptrtoint ptr addrspace(200) %cap to i64
  %a = add i64 %i, 15
  %m = and i64 %a, -16
  %r = inttoptr i64 %m to ptr addrspace(200)
  call void @log(ptr addrspace(200) %r)
  store ptr addrspace(200) %cap, ptr addrspace(200) %slot
  ret void
}

; CHECK-LABEL: @linear_argument_identity(
; CHECK: %r = inttoptr i64 %i to ptr addrspace(200)
; CHECK: call addrspace(200) void @log(ptr addrspace(200) %r)
define void @linear_argument_identity(ptr addrspace(200) %cap,
                                     ptr addrspace(200) %slot) {
  %i = ptrtoint ptr addrspace(200) %cap to i64
  %r = inttoptr i64 %i to ptr addrspace(200)
  call void @log(ptr addrspace(200) %r)
  store ptr addrspace(200) %cap, ptr addrspace(200) %slot
  ret void
}

; Revoke-on-free allocators park MREV results in file-scope arrays. A load
; from such storage is not evidence that the stored capability is NONLIN.
; CHECK-LABEL: @linear_global_load(
; CHECK: %cap = load ptr addrspace(200), ptr addrspace(200) @saved
; CHECK: %r = inttoptr i64 %m to ptr addrspace(200)
; CHECK: store ptr addrspace(200) %cap, ptr addrspace(200) @saved
define void @linear_global_load(ptr addrspace(200) %u) {
entry:
  %rev = call ptr addrspace(200) @llvm.capstone.cap.mrev.p200(ptr addrspace(200) %u)
  store ptr addrspace(200) %rev, ptr addrspace(200) @saved
  %cap = load ptr addrspace(200), ptr addrspace(200) @saved
  %nz = icmp ne ptr addrspace(200) %cap, null
  br i1 %nz, label %use, label %out
use:
  %i = ptrtoint ptr addrspace(200) %cap to i64
  %a = add i64 %i, 15
  %m = and i64 %a, -16
  %r = inttoptr i64 %m to ptr addrspace(200)
  call void @log(ptr addrspace(200) %r)
  store ptr addrspace(200) %cap, ptr addrspace(200) @saved
  br label %out
out:
  ret void
}

; CHECK-LABEL: @linear_external_load_identity(
; CHECK: %r = inttoptr i64 %i to ptr addrspace(200)
define void @linear_external_load_identity(ptr addrspace(200) %slot) {
  %cap = load ptr addrspace(200), ptr addrspace(200) %slot
  %i = ptrtoint ptr addrspace(200) %cap to i64
  %r = inttoptr i64 %i to ptr addrspace(200)
  call void @log(ptr addrspace(200) %r)
  store ptr addrspace(200) %cap, ptr addrspace(200) %slot
  ret void
}

; A local spill preserves the uncertainty of its input.
; CHECK-LABEL: @linear_argument_slot(
; CHECK: %r = inttoptr i64 %i to ptr addrspace(200)
define void @linear_argument_slot(ptr addrspace(200) %arg,
                                 ptr addrspace(200) %out) {
  %slot = alloca ptr addrspace(200), addrspace(200)
  store ptr addrspace(200) %arg, ptr addrspace(200) %slot
  %cap = load ptr addrspace(200), ptr addrspace(200) %slot
  %i = ptrtoint ptr addrspace(200) %cap to i64
  %r = inttoptr i64 %i to ptr addrspace(200)
  call void @log(ptr addrspace(200) %r)
  store ptr addrspace(200) %cap, ptr addrspace(200) %out
  ret void
}

; An ordinary callee can return a LINEAR capability too. Even a dominating
; dereference proves validity, not that an additional use cannot consume it.
; CHECK-LABEL: @linear_call(
; CHECK: %r = inttoptr i64 %m to ptr addrspace(200)
define void @linear_call(ptr addrspace(200) %slot) {
  %cap = call ptr addrspace(200) @get_cap()
  store volatile i8 0, ptr addrspace(200) %cap
  %i = ptrtoint ptr addrspace(200) %cap to i64
  %a = add i64 %i, 15
  %m = and i64 %a, -16
  %r = inttoptr i64 %m to ptr addrspace(200)
  call void @log(ptr addrspace(200) %r)
  store ptr addrspace(200) %cap, ptr addrspace(200) %slot
  ret void
}

