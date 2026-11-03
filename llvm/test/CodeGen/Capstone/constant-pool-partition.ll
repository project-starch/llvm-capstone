; RUN: llc -mtriple=capstone64-unknown-elf -mattr=+m,+a,+f,+d -target-abi lp64 \
; RUN:   -capstone-gp-free -capstone-image-gp -verify-machineinstrs \
; RUN:   -partition-static-data-sections -function-sections -data-sections \
; RUN:   -unique-section-names=false < %s | FileCheck %s
; RUN: llc -mtriple=capstone64-unknown-elf -mattr=+m,+a,+f,+d -target-abi lp64 \
; RUN:   -capstone-gp-free -capstone-image-gp -verify-machineinstrs \
; RUN:   -function-sections -data-sections -unique-section-names=false < %s \
; RUN:   | FileCheck %s

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; Static-data partitioning must not change a shared pool's layout. The
; AsmPrinter asks for an entry's section through the five-argument
; getSectionForConstant with the partition's "hot"/"unlikely" suffix; the ELF
; default then emits a hot entry into .rodata.cst8.hot. while an unprofiled
; neighbour stays where rebaseConstantPool predicted it, and the rewritten
; displacements no longer reach the entries. Under the image-gp ABI the
; Capstone override ignores the suffix, so with a profile and partitioning
; (first RUN line) the output is the same as without (second RUN line, same
; CHECKs): every entry in .rodata, one base per function, the loads at 0, 8,
; 16 and 24.
;
; CHECK-NOT:   .rodata.cst
; CHECK:       .section .rodata,"a",@progbits
; CHECK:       .LCPI0_0:
; CHECK:       .LCPI0_3:
; CHECK-NOT:   .rodata.cst
; CHECK-LABEL: hot_poly:
; CHECK:       scc [[BASE:a[0-9]+]], gp,
; CHECK-NOT:   scc {{.*}}, gp,
; CHECK-DAG:   fld {{f[a-z0-9]+}}, 0([[BASE]])
; CHECK-DAG:   fld {{f[a-z0-9]+}}, 8([[BASE]])
; CHECK-DAG:   fld {{f[a-z0-9]+}}, 16([[BASE]])
; CHECK-DAG:   fld {{f[a-z0-9]+}}, 24([[BASE]])
; CHECK:       ret
; CHECK-NOT:   .rodata.cst
; CHECK:       .section .rodata,"a",@progbits
; CHECK:       .LCPI1_0:
; CHECK:       .LCPI1_3:
; CHECK-NOT:   .rodata.cst
; CHECK-LABEL: cold_poly:
; CHECK:       scc [[BASE:a[0-9]+]], gp,
; CHECK-NOT:   scc {{.*}}, gp,
; CHECK-DAG:   fld {{f[a-z0-9]+}}, 0([[BASE]])
; CHECK-DAG:   fld {{f[a-z0-9]+}}, 8([[BASE]])
; CHECK-DAG:   fld {{f[a-z0-9]+}}, 16([[BASE]])
; CHECK-DAG:   fld {{f[a-z0-9]+}}, 24([[BASE]])
; CHECK:       ret
define double @hot_poly(double %x) addrspace(200) nounwind !prof !17 {
entry:
  %m0 = fmul double %x, 0x3FD5555555555555
  %a0 = fadd double %m0, 0x3FE5555555555555
  %m1 = fmul double %a0, 0x3FC999999999999A
  %a1 = fadd double %m1, 0x3FF3333333333333
  ret double %a1
}

define double @cold_poly(double %x) addrspace(200) nounwind !prof !16 {
entry:
  %m0 = fmul double %x, 0x3FD5555555555555
  %a0 = fadd double %m0, 0x3FE5EB851EB851EC
  %m1 = fmul double %a0, 0x3FC999999999999A
  %a1 = fadd double %m1, 0x3FE5C28F5C28F5C3
  ret double %a1
}

!llvm.module.flags = !{!1}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10, !11, !12}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 1460617}
!5 = !{!"MaxCount", i64 849536}
!6 = !{!"MaxInternalCount", i64 32769}
!7 = !{!"MaxFunctionCount", i64 849536}
!8 = !{!"NumCounts", i64 23784}
!9 = !{!"NumFunctions", i64 3301}
!10 = !{!"IsPartialProfile", i64 0}
!11 = !{!"PartialProfileRatio", double 0.000000e+00}
!12 = !{!"DetailedSummary", !13}
!13 = !{!14, !15}
!14 = !{i32 990000, i64 166, i32 73}
!15 = !{i32 999999, i64 1, i32 1463}
!16 = !{!"function_entry_count", i64 1}
!17 = !{!"function_entry_count", i64 100000}
