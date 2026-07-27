; RUN: llc -mtriple=capstone64-unknown-elf -O2 < %s | FileCheck %s
;
; Regression test for issue C-1: "Cannot select: i128 = sign_extend_inreg".
;
; An `int` (not `long`) index feeding capability address arithmetic produces an
; i128 sign_extend_inreg. The Custom lowering for it only runs during Legalize,
; and performSIGN_EXTEND_INREGCombine deliberately handles ONLY the
; any_extend(i64) shape -- expanding the general case in a combine ping-pongs
; against visitSIGN_EXTEND forever. So every other shape reached ISel
; unselectable and crashed the backend at -O1 and above.
;
; It is now selected directly in CapstoneDAGToDAGISel::Select: truncate to XLen,
; sign-extend the source field with a shift pair, widen with
; PseudoSCALAR_COPY_I128.
;
; The test only needs to COMPILE -- the crash was the bug.

; CHECK-LABEL: domain_main:

target datalayout = "e-m:e-p:64:128-p200:128:128:128-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-unknown-elf"
@rh_a = internal unnamed_addr addrspace(200) global [4 x i32] zeroinitializer, align 4

define dso_local void @domain_main(ptr addrspace(200) noundef writeonly captures(none) initializes((0, 64)) %res, i32 noundef %func) local_unnamed_addr addrspace(200) #0 {
entry:
  %0 = tail call addrspace(0) i64 asm sideeffect "csrr $0, mcycle", "=r"() #1, !srcloc !3
  store i32 170, ptr addrspace(200) @rh_a, align 4, !tbaa !4
  tail call addrspace(0) void asm sideeffect "", "~{memory}"() #1, !srcloc !8
  %1 = tail call addrspace(0) i32 asm sideeffect "", "=r,0"(i32 1) #1, !srcloc !9
  %2 = sext i32 %1 to i128
  %3 = getelementptr i32, ptr addrspace(200) @rh_a, i128 %2
  %arrayidx = getelementptr i8, ptr addrspace(200) %3, i128 -4
  store i32 187, ptr addrspace(200) %arrayidx, align 4, !tbaa !4
  tail call addrspace(0) void asm sideeffect "", "~{memory}"() #1, !srcloc !10
  %4 = load i32, ptr addrspace(200) %arrayidx, align 4, !tbaa !4
  %conv = zext i32 %4 to i64
  %arrayidx4 = getelementptr inbounds nuw i8, ptr addrspace(200) %res, i128 24
  store i64 %conv, ptr addrspace(200) %arrayidx4, align 8, !tbaa !11
  store i32 170, ptr addrspace(200) getelementptr inbounds nuw (i8, ptr addrspace(200) @rh_a, i128 4), align 4, !tbaa !4
  tail call addrspace(0) void asm sideeffect "", "~{memory}"() #1, !srcloc !13
  %5 = tail call addrspace(0) i32 asm sideeffect "", "=r,0"(i32 1) #1, !srcloc !9
  %6 = tail call addrspace(0) i32 asm sideeffect "", "=r,0"(i32 1) #1, !srcloc !9
  %idxprom7 = sext i32 %5 to i128
  %arrayidx8 = getelementptr inbounds i32, ptr addrspace(200) @rh_a, i128 %idxprom7
  store i32 187, ptr addrspace(200) %arrayidx8, align 4, !tbaa !4
  tail call addrspace(0) void asm sideeffect "", "~{memory}"() #1, !srcloc !14
  %idxprom9 = sext i32 %6 to i128
  %arrayidx10 = getelementptr inbounds i32, ptr addrspace(200) @rh_a, i128 %idxprom9
  %7 = load i32, ptr addrspace(200) %arrayidx10, align 4, !tbaa !4
  %conv11 = zext i32 %7 to i64
  %arrayidx12 = getelementptr inbounds nuw i8, ptr addrspace(200) %res, i128 32
  store i64 %conv11, ptr addrspace(200) %arrayidx12, align 8, !tbaa !11
  store i32 170, ptr addrspace(200) getelementptr inbounds nuw (i8, ptr addrspace(200) @rh_a, i128 8), align 4, !tbaa !4
  tail call addrspace(0) void asm sideeffect "", "~{memory}"() #1, !srcloc !15
  %8 = tail call addrspace(0) i32 asm sideeffect "", "=r,0"(i32 1) #1, !srcloc !9
  %9 = sext i32 %8 to i128
  %10 = getelementptr i32, ptr addrspace(200) @rh_a, i128 %9
  %arrayidx15 = getelementptr i8, ptr addrspace(200) %10, i128 4
  store i32 187, ptr addrspace(200) %arrayidx15, align 4, !tbaa !4
  tail call addrspace(0) void asm sideeffect "", "~{memory}"() #1, !srcloc !16
  %11 = load i32, ptr addrspace(200) %arrayidx15, align 4, !tbaa !4
  %conv18 = zext i32 %11 to i64
  %arrayidx19 = getelementptr inbounds nuw i8, ptr addrspace(200) %res, i128 40
  store i64 %conv18, ptr addrspace(200) %arrayidx19, align 8, !tbaa !11
  store i32 170, ptr addrspace(200) getelementptr inbounds nuw (i8, ptr addrspace(200) @rh_a, i128 12), align 4, !tbaa !4
  tail call addrspace(0) void asm sideeffect "", "~{memory}"() #1, !srcloc !17
  %12 = tail call addrspace(0) i32 asm sideeffect "", "=r,0"(i32 3) #1, !srcloc !9
  %idxprom21 = sext i32 %12 to i128
  %arrayidx22 = getelementptr inbounds i32, ptr addrspace(200) @rh_a, i128 %idxprom21
  store i32 187, ptr addrspace(200) %arrayidx22, align 4, !tbaa !4
  tail call addrspace(0) void asm sideeffect "", "~{memory}"() #1, !srcloc !18
  %13 = load i32, ptr addrspace(200) %arrayidx22, align 4, !tbaa !4
  %conv25 = zext i32 %13 to i64
  %arrayidx26 = getelementptr inbounds nuw i8, ptr addrspace(200) %res, i128 48
  store i64 %conv25, ptr addrspace(200) %arrayidx26, align 8, !tbaa !11
  store i32 170, ptr addrspace(200) @rh_a, align 4, !tbaa !4
  tail call addrspace(0) void asm sideeffect "", "~{memory}"() #1, !srcloc !19
  store i32 187, ptr addrspace(200) @rh_a, align 4, !tbaa !4
  tail call addrspace(0) void asm sideeffect "", "~{memory}"() #1, !srcloc !20
  %14 = load i32, ptr addrspace(200) @rh_a, align 4, !tbaa !4
  %conv27 = zext i32 %14 to i64
  %arrayidx28 = getelementptr inbounds nuw i8, ptr addrspace(200) %res, i128 56
  store i64 %conv27, ptr addrspace(200) %arrayidx28, align 8, !tbaa !11
  %15 = tail call addrspace(0) i64 asm sideeffect "csrr $0, mcycle", "=r"() #1, !srcloc !3
  store i64 48879, ptr addrspace(200) %res, align 8, !tbaa !11
  %sub31 = sub i64 %15, %0
  %arrayidx32 = getelementptr inbounds nuw i8, ptr addrspace(200) %res, i128 8
  store i64 %sub31, ptr addrspace(200) %arrayidx32, align 8, !tbaa !11
  %arrayidx33 = getelementptr inbounds nuw i8, ptr addrspace(200) %res, i128 16
  store i64 53406, ptr addrspace(200) %arrayidx33, align 8, !tbaa !11
  ret void
}

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+64bit,+i" }
attributes #1 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 22.0.0git (https://github.com/project-starch/llvm-capstone cf4db3b07590d6f82565d45b36d1cd248ce55cb5)"}
!3 = !{i64 1201}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{i64 1440}
!9 = !{i64 1096}
!10 = !{i64 1516}
!11 = !{!12, !12, i64 0}
!12 = !{!"long", !6, i64 0}
!13 = !{i64 1683}
!14 = !{i64 1770}
!15 = !{i64 1921}
!16 = !{i64 1997}
!17 = !{i64 2156}
!18 = !{i64 2228}
!19 = !{i64 2390}
!20 = !{i64 2445}
