; C-27: under the default global ABI every gp-derived base is `delin`ed once --
; cincoffset from gp, then delin, per global -- and never more than once.  Under
; -capstone-gp-captable the base is loaded from the capability table and there
; is NO delin anywhere.  Both counts are enforced by --implicit-check-not.  This
; pins today's emission so the Tier 4 DELIN decision (drop the delin under the
; NONLIN-gp contract, or make gp-captable the silicon default) has a red test
; to flip.  Measured 2026-09-04 on the branch tools.
;
; MUTATION: the two arms are each other's mutation -- the CT arm's
; implicit-check-not on delin fires on the default-ABI output, which carries two
; (performed 2026-09-04 by running both).
;
; RUN: llc -mtriple=capstone64 -mattr=+m < %s | FileCheck %s --check-prefix=CAP --implicit-check-not=delin
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-captable < %s | FileCheck %s --check-prefix=CT --implicit-check-not=delin
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
define i64 @sum() {
  %x = load i64, ptr addrspace(200) @a
  %y = load i64, ptr addrspace(200) @b
  %s = add i64 %x, %y
  ret i64 %s
}
