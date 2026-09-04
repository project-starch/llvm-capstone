; cap_init and cap_seal CONSUME their source (the spec's MOVC step; QEMU nulls
; rs1), and the RTL's INIT writes the new LINEAR capability to BOTH rs1 and rd
; when they differ (R-25, capstone_flu_unit.anvil:147): two live LINEAR
; capabilities over one region.  Before 2026-09-05 nothing tied the two
; registers, and a source kept live after the builtin produced exactly that
; shape -- measured: `init a1, a0, a1` at -O1 and -O2 for @init_live_source.
; Since then codegen goes through PseudoINIT / PseudoSEAL, whose rd is tied to
; rs1, so the result always overwrites the consumed source; a live source is
; copied first.  This file pins "rd equals rs1" with a FileCheck capture.
;
; MUTATION: the pre-fix output IS the failing case -- `init a1, a0, a1` does not
; satisfy `init [[R]], [[R]],` and the capture check fails on the compiler
; before this change (measured 2026-09-05, before the rebuild).  A later
; regression (dropping the Constraints line) reproduces it.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -O1 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -verify-machineinstrs < %s | FileCheck %s

declare ptr addrspace(200) @llvm.capstone.cap.init.p200(ptr addrspace(200), i64)
declare ptr addrspace(200) @llvm.capstone.cap.seal.p200(ptr addrspace(200))

; The source is still live after the init (it is returned), so the allocator
; used to give the result a different register.
; CHECK-LABEL: init_live_source:
; CHECK: init [[R:a[0-9]+]], [[R]], a{{[0-9]+}}
define ptr addrspace(200) @init_live_source(ptr addrspace(200) %p, i64 %off, ptr addrspace(200) %out) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.init.p200(ptr addrspace(200) %p, i64 %off)
  store ptr addrspace(200) %r, ptr addrspace(200) %out
  ret ptr addrspace(200) %p
}

; CHECK-LABEL: init_dead_source:
; CHECK: init [[R:a[0-9]+]], [[R]], a{{[0-9]+}}
define ptr addrspace(200) @init_dead_source(ptr addrspace(200) %p, i64 %off) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.init.p200(ptr addrspace(200) %p, i64 %off)
  ret ptr addrspace(200) %r
}

; CHECK-LABEL: seal_live_source:
; CHECK: seal [[R:a[0-9]+]], [[R]]
define ptr addrspace(200) @seal_live_source(ptr addrspace(200) %p, ptr addrspace(200) %out) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.seal.p200(ptr addrspace(200) %p)
  store ptr addrspace(200) %r, ptr addrspace(200) %out
  ret ptr addrspace(200) %p
}

; CHECK-LABEL: seal_dead_source:
; CHECK: seal [[R:a[0-9]+]], [[R]]
define ptr addrspace(200) @seal_dead_source(ptr addrspace(200) %p) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.seal.p200(ptr addrspace(200) %p)
  ret ptr addrspace(200) %r
}
