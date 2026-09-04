; -capstone-merge-string-constants (gp-captable ABI only) folds private
; unnamed_addr byte-array constants into one container so they take ONE
; cap-table slot; a named constant keeps its own slot.  With the flag on, the
; two literals share slot 1 and the second is reached by an interior offset;
; with it off, or with -capstone-merge-string-max-bytes below what two literals
; need, every literal keeps its own slot and no container exists.  The
; container's private label is doubly .L-prefixed (the pass names it .L__...,
; the printer adds its own).  Measured 2026-09-04 on the branch tools.
;
; MUTATION: the ON and OFF arms are each other's mutation -- the OFF arms'
; implicit-check-not on merged_strs fires on the ON output (performed
; 2026-09-04 by running both).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-captable -capstone-merge-string-constants=true -capstone-merge-string-max-bytes=4096 < %s | FileCheck %s --check-prefix=ON
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-captable -capstone-merge-string-constants=false < %s | FileCheck %s --check-prefix=OFF --implicit-check-not=merged_strs
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-captable < %s | FileCheck %s --check-prefix=OFF --implicit-check-not=merged_strs
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-captable -capstone-merge-string-constants -capstone-merge-string-max-bytes=8 < %s | FileCheck %s --check-prefix=OFF --implicit-check-not=merged_strs
; RUN: %llc_cap -O0 -capstone-gp-captable -capstone-merge-string-constants < %s -o /dev/null
; RUN: %llc_cap -O1 -capstone-gp-captable -capstone-merge-string-constants < %s -o /dev/null

@.str = private unnamed_addr addrspace(200) constant [6 x i8] c"hello\00", align 1
@.str.1 = private unnamed_addr addrspace(200) constant [6 x i8] c"world\00", align 1
@named = internal addrspace(200) constant [4 x i8] c"abc\00", align 1

; ON-LABEL: a:
; ON: ldc a0, 16(gp)
; ON-LABEL: b:
; ON: ldc a0, 16(gp)
; ON-NEXT: cincoffsetimm a0, a0, 6
; ON-LABEL: c:
; ON: ldc a0, 0(gp)
; ON: .L.L__capstone_merged_strs.0:
; ON-NEXT: .asciz "hello"
; ON-NEXT: .asciz "world"
; ON: .section .capstone_gp_table
; ON-NEXT: .p2align 3, 0x0
; ON-NEXT: .quad 2

; OFF-LABEL: a:
; OFF: ldc a0, 0(gp)
; OFF-LABEL: b:
; OFF: ldc a0, 16(gp)
; OFF-LABEL: c:
; OFF: ldc a0, 32(gp)
; OFF: .section .capstone_gp_table
; OFF-NEXT: .p2align 3, 0x0
; OFF-NEXT: .quad 3
define ptr addrspace(200) @a() { ret ptr addrspace(200) @.str }
define ptr addrspace(200) @b() { ret ptr addrspace(200) @.str.1 }
define ptr addrspace(200) @c() { ret ptr addrspace(200) @named }
