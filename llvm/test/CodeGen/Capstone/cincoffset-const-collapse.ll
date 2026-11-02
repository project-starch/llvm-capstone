; performCIncOffsetCombine, constant-cursor collapse (lane C, item 2 of the
; 2026-11-02 codegen plan).
;
; When a capability is loaded and its address bits are compared (musl fgetc's
; `rpos != rend`), lowerSETCC truncates both capabilities to i64 and
; DAGCombiner::reduceLoadWidth narrows the loaded c128 to i64, rebuilding the
; pointer with getMemBasePlusOffset(ptr, 0). For c128 that is a PTRADD lowered
; to cincoffset(cincoffset(base, 0x20), 0): a dead +0 cursor move that keeps
; SelectAddrRegImm from landing the 0x20 in the load immediate. Collapsing
;   cincoffset(x, 0) -> x
; lets the surviving constant level fold into the access:
;   cincoffsetimm aX, a0, 32 ; ld 0(aX)   ->   ld 32(a0)
;
; Legal: a zero cursor move is the identity (same cursor, bounds, tag); no new
; cursor is formed, so there is no R-60 representability concern.
;
; MUTATION: delete `if (isNullConstant(Off)) return Base;` from
; performCIncOffsetCombine and @ne keeps a `cincoffsetimm a{{.}}, a0, 32` before
; the compared load.
;
; The gp-free ABI is the application profile; its returns print as `ret`.
; RUN: llc -mtriple=capstone64 -mattr=+m,+f,+d -O2 -capstone-gp-free -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m,+f,+d -O0 -capstone-gp-free -verify-machineinstrs < %s | FileCheck %s --check-prefix=O0
; RUN: %llc_cap -mattr=+f,+d -O0 < %s -o /dev/null
; RUN: %llc_cap -mattr=+f,+d -O1 < %s -o /dev/null
; RUN: %llc_cap -mattr=+f,+d -O2 < %s -o /dev/null

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

declare i64 @llvm.capstone.cap.get.tag.p200(ptr addrspace(200))

; POLICY (lane owner, 2026-11-03, after review): collapsing constant cursor
; moves may KEEP a tag the original sequence lost, never lose one. The
; excursion x-4096 leaves the ~2 KiB representable window and clears the tag,
; and the return to x+1 stays untagged; the collapsed cincoffset(x, 1) keeps
; the tag, which get-tag observes. That is accepted, in the direction of CHERI
; LLVM's rule (folding offsets is allowed, splitting one is not). The backend
; already eliminates this excursion at -O0 through DAG constant folding of the
; address round trip; the nested-constant collapse in performCIncOffsetCombine
; extends it to a nesting of target nodes. Both levels: one `cincoffsetimm 1`,
; no -4096 materialised.
; CHECK-LABEL: cancel:
; CHECK:      cincoffsetimm a0, a0, 1
; CHECK-NEXT: lcc a0, a0, 1
; CHECK-NEXT: sltiu a0, a0, 7
; CHECK-NOT:  4096
; CHECK:      ret
; O0-LABEL: cancel:
; O0:         cincoffsetimm a0, a0, 1
; O0-NOT:     4096
; O0:         ret
define i64 @cancel(ptr addrspace(200) %x) {
  %p = getelementptr i8, ptr addrspace(200) %x, i64 -4096
  %a = ptrtoint ptr addrspace(200) %p to i64
  %v = add i64 %a, 4097
  %q = inttoptr i64 %v to ptr addrspace(200)
  %tag = call i64 @llvm.capstone.cap.get.tag.p200(ptr addrspace(200) %q)
  ret i64 %tag
}

; The field at 0x20 is loaded for the pointer compare with its offset folded
; into the load, not via a separate cursor move.
; CHECK-LABEL: ne:
; CHECK-NOT:  cincoffsetimm a{{[0-9]+}}, a0, 32
; CHECK:      ld {{a[0-9]+}}, 32(a0)
; CHECK:      ret
define i32 @ne(ptr addrspace(200) %s) {
entry:
  %0 = load ptr addrspace(200), ptr addrspace(200) %s, align 16
  %q = getelementptr inbounds nuw i8, ptr addrspace(200) %s, i64 32
  %1 = load ptr addrspace(200), ptr addrspace(200) %q, align 16
  %cmp = icmp ne ptr addrspace(200) %0, %1
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CONTROL: a single narrowed load already folds through SelectAddrRegImm's
; one-level peel (no +0 remnant), so step 1 changes nothing here.
; CHECK-LABEL: load_at:
; CHECK:      ld a0, 32(a0)
; CHECK:      ret
define i64 @load_at(ptr addrspace(200) %s) {
entry:
  %q = getelementptr inbounds nuw i8, ptr addrspace(200) %s, i64 32
  %p = load ptr addrspace(200), ptr addrspace(200) %q, align 16
  %v = ptrtoint ptr addrspace(200) %p to i64
  ret i64 %v
}
