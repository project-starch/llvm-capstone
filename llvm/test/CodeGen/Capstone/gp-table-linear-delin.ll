; C-27, and THE DECISION THIS TEST EXISTED TO HAVE FLIPPED HAS BEEN MADE.
;
; Three ABIs, three emissions, each pinned two-sidedly by --implicit-check-not:
;   CAP  default capability ABI  -- cincoffset from gp, then `delin`, once per
;        global and never more than once.  UNCHANGED.
;   CT   -capstone-gp-captable   -- the base is loaded from the capability table;
;        NO delin anywhere.  UNCHANGED.
;   IG   -capstone-image-gp      -- `scc` from gp (an image-bounded gp with cursor
;        0 is unrepresentable, so the cursor is set absolutely) and NO delin,
;        because that ABI asserts a NONLIN gp.  NEW.
;
; Why IG drops it: `delin` of a non-LINEAR capability is UNEXPECTED_CAP_TYPE per
; the spec and raises cause 26 on the RTL, while QEMU's helper returned early on
; NONLIN and hid it -- so the sequence was QEMU-only by construction.  vp2's
; Tier 1/2 boot 2 (2026-10-18) faulted with cause 26 at `delin a0` right after
; `scc a0, gp, a0` on EVERY virtual step; 525 of contract.dom's 530 delins were
; this materialisation.  The removal is sound only because the trusted adapter
; grants every launch slot NONLIN, which is now part of -capstone-image-gp's
; documented contract.  Measured 2026-09-04 on the branch tools; IG arm added
; 2026-10-18.
;
; MUTATION: the three arms are each other's mutation -- the CT and IG arms'
; implicit-check-not on delin fires on the default-ABI output, which carries two
; (performed 2026-09-04 by running both; re-performed for IG 2026-10-18).
;
; RUN: llc -mtriple=capstone64 -mattr=+m < %s | FileCheck %s --check-prefix=CAP --implicit-check-not=delin
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-captable < %s | FileCheck %s --check-prefix=CT --implicit-check-not=delin
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-free -capstone-image-gp < %s | FileCheck %s --check-prefix=IG --implicit-check-not=delin
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

@a = addrspace(200) global i64 1
@b = addrspace(200) global i64 2

; CAP-LABEL: sum:
; CAP: cincoffset [[RA:a[0-9]+]], gp, [[RA]]
; CAP: delin [[RA]]
; CAP: cincoffset [[RB:a[0-9]+]], gp, [[RB]]
; CAP: delin [[RB]]
; CAP: cjalr zero, 0(ra)
; CT-LABEL: sum:
; CT: ldc {{a[0-9]+}}, 0(gp)
; CT: ldc {{a[0-9]+}}, 16(gp)
; CT: ret
; IG-LABEL: sum:
; IG: scc [[SA:a[0-9]+]], gp, [[SA]]
; IG: scc [[SB:a[0-9]+]], gp, [[SB]]
define i64 @sum() {
  %x = load i64, ptr addrspace(200) @a
  %y = load i64, ptr addrspace(200) @b
  %s = add i64 %x, %y
  ret i64 %s
}
