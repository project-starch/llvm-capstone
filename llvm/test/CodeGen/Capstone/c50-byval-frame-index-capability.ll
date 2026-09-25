; C-50: a by-value aggregate whose first 16-byte chunk holds a NON-CONSTANT
; int-to-pointer value had its caller-side stack copy addressed with an integer
; `addi` off the capability frame pointer, and the store faulted on silicon with
; cause 24 (untagged operand). Found by the FFmpeg app port; ff_mpv_alloc_pic_pool
; stored to sp+8 as an integer.
;
; CAUSE. LowerCall creates the byval local copy's frame index with
;   DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()))
; and getPointerTy's default argument is AS 0, which on this target is a 64-bit
; INTEGER; a stack object is a capability (datalayout A200). Every other frame
; index LLVM builds already goes through TargetLowering::getFrameIndexTy, which
; is getPointerTy(DL, DL.getAllocaAddrSpace()) -- this site was the outlier.
;
; WHY THE +8 HALF AND NOT THE FIRST. The object is 16-aligned, so `add FI, 8`
; has its low bits known zero and is rewritten to `or disjoint`, which selects
; to the integer ADDI. `or` cannot form on a c128, which is why typing the frame
; index correctly removes the shape rather than merely hiding it. The FIRST half
; is at offset 0 and needs no add, which is why only one of the two stores ever
; faulted.
;
; The check is the ABSENCE of an integer add on the frame pointer. It is a real
; check, not a vacuous one: on the pre-fix compiler this exact IR emits four of
; them, one per triggering function, and none in the control.
;
; RUN: llc -mtriple=capstone64 -O1 < %s | FileCheck %s
;
; CHECK-LABEL: v_union:
; CHECK-NOT: addi {{[a-z0-9]+}}, s0,
; CHECK-LABEL: v_struct:
; CHECK-NOT: addi {{[a-z0-9]+}}, s0,
; CHECK-LABEL: v_union1:
; CHECK-NOT: addi {{[a-z0-9]+}}, s0,
; CHECK-LABEL: v_s32:
; CHECK-NOT: addi {{[a-z0-9]+}}, s0,
; CHECK-LABEL: ctl_ptr:
; CHECK-NOT: addi {{[a-z0-9]+}}, s0,
target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-unknown-elf"

%union.u2 = type { ptr addrspace(200) }
%struct.s1 = type { ptr addrspace(200) }
%union.u1 = type { ptr addrspace(200) }
%struct.s32 = type { ptr addrspace(200), i64, i64, i64 }

; Function Attrs: nounwind
define dso_local ptr addrspace(200) @v_union(i32 noundef %x) local_unnamed_addr addrspace(200) #0 {
entry:
  %.compoundliteral = alloca %union.u2, align 16, addrspace(200)
  %conv = sext i32 %x to i64
  %conv1 = zext i64 %conv to i128
  %0 = inttoptr i128 %conv1 to ptr addrspace(200)
  store ptr addrspace(200) %0, ptr addrspace(200) %.compoundliteral, align 16, !tbaa !3
  %call = tail call addrspace(200) ptr addrspace(200) @sink_u2(i64 noundef 64, i32 noundef 1, ptr addrspace(200) noundef byval(%union.u2) align 16 %.compoundliteral, ptr addrspace(200) noundef null) #2
  ret ptr addrspace(200) %call
}

declare dso_local ptr addrspace(200) @sink_u2(i64 noundef, i32 noundef, ptr addrspace(200) noundef byval(%union.u2) align 16, ptr addrspace(200) noundef) local_unnamed_addr addrspace(200) #1

; Function Attrs: nounwind
define dso_local ptr addrspace(200) @v_struct(i32 noundef %x) local_unnamed_addr addrspace(200) #0 {
entry:
  %.compoundliteral = alloca %struct.s1, align 16, addrspace(200)
  %conv = sext i32 %x to i64
  %conv1 = zext i64 %conv to i128
  %0 = inttoptr i128 %conv1 to ptr addrspace(200)
  store ptr addrspace(200) %0, ptr addrspace(200) %.compoundliteral, align 16, !tbaa !6
  %call = tail call addrspace(200) ptr addrspace(200) @sink_s1(i64 noundef 64, i32 noundef 1, ptr addrspace(200) noundef byval(%struct.s1) align 16 %.compoundliteral, ptr addrspace(200) noundef null) #2
  ret ptr addrspace(200) %call
}

declare dso_local ptr addrspace(200) @sink_s1(i64 noundef, i32 noundef, ptr addrspace(200) noundef byval(%struct.s1) align 16, ptr addrspace(200) noundef) local_unnamed_addr addrspace(200) #1

; Function Attrs: nounwind
define dso_local ptr addrspace(200) @v_union1(i32 noundef %x) local_unnamed_addr addrspace(200) #0 {
entry:
  %.compoundliteral = alloca %union.u1, align 16, addrspace(200)
  %conv = sext i32 %x to i64
  %conv1 = zext i64 %conv to i128
  %0 = inttoptr i128 %conv1 to ptr addrspace(200)
  store ptr addrspace(200) %0, ptr addrspace(200) %.compoundliteral, align 16, !tbaa !3
  %call = tail call addrspace(200) ptr addrspace(200) @sink_u1(i64 noundef 64, i32 noundef 1, ptr addrspace(200) noundef byval(%union.u1) align 16 %.compoundliteral, ptr addrspace(200) noundef null) #2
  ret ptr addrspace(200) %call
}

declare dso_local ptr addrspace(200) @sink_u1(i64 noundef, i32 noundef, ptr addrspace(200) noundef byval(%union.u1) align 16, ptr addrspace(200) noundef) local_unnamed_addr addrspace(200) #1

; Function Attrs: nounwind
define dso_local ptr addrspace(200) @v_s32(i32 noundef %x) local_unnamed_addr addrspace(200) #0 {
entry:
  %.compoundliteral = alloca %struct.s32, align 16, addrspace(200)
  %conv = sext i32 %x to i64
  %conv1 = zext i64 %conv to i128
  %0 = inttoptr i128 %conv1 to ptr addrspace(200)
  store ptr addrspace(200) %0, ptr addrspace(200) %.compoundliteral, align 16, !tbaa !9
  %a = getelementptr inbounds nuw i8, ptr addrspace(200) %.compoundliteral, i64 16
  store i64 0, ptr addrspace(200) %a, align 16, !tbaa !12
  %b = getelementptr inbounds nuw i8, ptr addrspace(200) %.compoundliteral, i64 24
  store i64 0, ptr addrspace(200) %b, align 8, !tbaa !13
  %c = getelementptr inbounds nuw i8, ptr addrspace(200) %.compoundliteral, i64 32
  store i64 0, ptr addrspace(200) %c, align 16, !tbaa !14
  %1 = getelementptr inbounds nuw i8, ptr addrspace(200) %.compoundliteral, i64 40
  store i64 0, ptr addrspace(200) %1, align 8
  %call = tail call addrspace(200) ptr addrspace(200) @sink_s32(i64 noundef 64, i32 noundef 1, ptr addrspace(200) noundef byval(%struct.s32) align 16 %.compoundliteral, ptr addrspace(200) noundef null) #2
  ret ptr addrspace(200) %call
}

declare dso_local ptr addrspace(200) @sink_s32(i64 noundef, i32 noundef, ptr addrspace(200) noundef byval(%struct.s32) align 16, ptr addrspace(200) noundef) local_unnamed_addr addrspace(200) #1

; Function Attrs: nounwind
define dso_local ptr addrspace(200) @ctl_ptr(ptr addrspace(200) noundef %p) local_unnamed_addr addrspace(200) #0 {
entry:
  %.compoundliteral = alloca %union.u2, align 16, addrspace(200)
  store ptr addrspace(200) %p, ptr addrspace(200) %.compoundliteral, align 16, !tbaa !3
  %call = tail call addrspace(200) ptr addrspace(200) @sink_u2(i64 noundef 64, i32 noundef 1, ptr addrspace(200) noundef byval(%union.u2) align 16 %.compoundliteral, ptr addrspace(200) noundef null) #2
  ret ptr addrspace(200) %call
}

attributes #0 = { nounwind "frame-pointer"="all" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+64bit,+i,+m,+zmmul" }
attributes #1 = { "frame-pointer"="all" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+64bit,+i,+m,+zmmul" }
attributes #2 = { nobuiltin nounwind "no-builtins" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 22.0.0git (https://github.com/project-starch/llvm-capstone.git e016204cc7042568cb113f4ad5c773ff102fa56f)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !8, i64 0}
!7 = !{!"", !8, i64 0}
!8 = !{!"any pointer", !4, i64 0}
!9 = !{!10, !8, i64 0}
!10 = !{!"", !8, i64 0, !11, i64 16, !11, i64 24, !11, i64 32}
!11 = !{!"long", !4, i64 0}
!12 = !{!10, !11, i64 16}
!13 = !{!10, !11, i64 24}
!14 = !{!10, !11, i64 32}
