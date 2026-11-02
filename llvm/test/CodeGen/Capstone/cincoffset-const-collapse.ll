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
; LLVM's rule (folding offsets is allowed, splitting one is not).
;
; What this test pins: the WHOLE body, anchored at the label and checked line
; by line, is one `cincoffsetimm a0, a0, 1` and the get-tag lowering; no
; materialised excursion (`lui 1048575; cincoffset; lui 1; cincoffset`, the
; shape a retained -4096/+4097 pair takes, since neither fits simm12) anywhere
; in it, at -O2 and at -O0. The mechanism, read off `-debug-only=dagcombine`
; rather than inferred from the -O0 output (target combines run at -O0 too):
; recoverCapabilityFromAddressArith rebuilds the outer add from the
; ptrtoint/add/inttoptr round trip as `CIncOffset (CIncOffset x, -4096), 4097`,
; and the nested-constant collapse of performCIncOffsetCombine turns that into
; `CIncOffset x, 1` ("Combining: t15: c128 = CIncOffset t16, Constant:i64<4097>
; ... into: t18: c128 = CIncOffset t2, Constant:i64<1>").
; MUTATION: delete the nested-constant collapse -> the two cursor moves are
; materialised and the CHECK-NEXT chain breaks right after `# %bb.0:`. The
; chain is anchored at the label and runs through every line of the function,
; so an excursion kept anywhere in the body (before or after `# %bb.0:`) is
; rejected; the review's `lui 1048575; cincoffset; lui 1; cincoffset` prelude
; fails both prefixes at either placement.
; CHECK-LABEL: cancel:
; CHECK-NEXT:   .cfi_startproc
; CHECK-NEXT: # %bb.0:
; CHECK-NEXT:   cincoffsetimm a0, a0, 1
; CHECK-NEXT:   lcc a0, a0, 1
; CHECK-NEXT:   sltiu a0, a0, 7
; CHECK-NEXT:   ret
; O0-LABEL: cancel:
; O0-NEXT:   .cfi_startproc
; O0-NEXT: # %bb.0:
; O0-NEXT:   cincoffsetimm a0, a0, 1
; O0-NEXT:   lcc a0, a0, 1
; O0-NEXT:   sltiu a0, a0, 7
; O0-NEXT:   ret
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
