; S-06 codegen workaround: the capability store in a 16-byte-aligned copy must be
; GUARDED by an LCC type query, so plain data never goes through `stc`.
;
; THE DEFECT. A bare `ldc`/`stc` pair copying PLAIN data loses the granule's high
; 8 bytes, by either of two mechanisms. When the cursor is not 0x4000-aligned the
; recompressed metadata is zero, `st_wr_cap` stays 0, and bank 1 is NEVER WRITTEN
; -- the destination keeps whatever it held before. When the cursor IS
; 0x4000-aligned, compress_bounds' cursorless branch manufactures a nonzero
; metadata out of an all-zero input, `st_wr_cap` fires, and bank 1 is written
; with that garbage. Either way the high half is gone. Measured on silicon: the
; minimal repro s06agg returns 5 (both low halves intact, BOTH high halves lost)
; where a correct machine returns 15.
;
; This reached SQLite through the compiler rather than through memcpy: the C
; library copy was fixed first, and the backend independently emitted 283 bare
; granule stores across 41 aggregate-copy runs, including a 112-byte `Mem` copy
; inlined into sqlite3VdbeExec. Repairing a subset of them on the board made a
; wild `Mem*` -- a heap capability whose cursor sat 54336 bytes past its end --
; disappear entirely.
;
; THE TEST HAS BOTH ARMS ON PURPOSE. Checking only the fixed output would pass
; just as happily if the pass silently did nothing, which is the failure mode
; this project keeps hitting: a gate that cannot fire is not a passing gate. The
; OFF arm pins the unfixed sequence, so if the default ever changes or the pass
; starts running unconditionally, this test fails rather than quietly agreeing.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=BARE
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs \
; RUN:     -capstone-guard-cap-granule-copies=true < %s \
; RUN:   | FileCheck %s --check-prefix=GUARD

declare void @llvm.memcpy.p200.p200.i64(ptr addrspace(200), ptr addrspace(200), i64, i1)

; A 32-byte, 16-byte-aligned copy: two capability-grained granules.
;
; BARE-LABEL: copy_32_align16:
; Unfixed: capability-grained stores with nothing asking whether there is a
; capability to store. No type query anywhere.
; BARE:       ldc
; BARE:       stc
; BARE-NOT:   lcc
;
; GUARD-LABEL: copy_32_align16:
; Fixed: both halves written plainly first, then the type query, then a branch
; over the capability store. `lcc rd, rs, 1` is field 1, the TYPE query, which
; is total on enabler silicon and answers 7 for a non-capability.
; GUARD:       sd
; GUARD:       sd
; GUARD:       lcc {{[a-z0-9]+}}, {{[a-z0-9]+}}, 1
; GUARD:       stc
define void @copy_32_align16(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %dst,
                                       ptr addrspace(200) align 16 %src,
                                       i64 32, i1 false)
  ret void
}

; An 8-byte-aligned copy cannot hold an in-place tagged capability, so it never
; used ldc/stc and the pass must leave it alone -- no query, no branch, no cost.
; Without this arm the pass could be rewriting copies that were never at risk and
; the test would not notice.
;
; GUARD-LABEL: copy_16_align8:
; GUARD-NOT:   lcc
; GUARD-NOT:   stc
define void @copy_16_align8(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 8 %dst,
                                       ptr addrspace(200) align 8 %src,
                                       i64 16, i1 false)
  ret void
}

; A volatile copy must not have its stores multiplied or reordered, so the pass
; must decline it even though it is 16-byte aligned.
;
; GUARD-LABEL: copy_32_volatile:
; GUARD-NOT:   lcc
define void @copy_32_volatile(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %dst,
                                       ptr addrspace(200) align 16 %src,
                                       i64 32, i1 true)
  ret void
}
