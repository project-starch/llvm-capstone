; C-87: -g must not change the code. CapstoneCursorReadFold drops the integer
; write (PseudoTRUNC_CAP) of an extracted capability address when every reader
; only compares or computes with it. It used to count a DBG_VALUE of the address
; as a reader that is not integer-only, so with debug information the write
; stayed (perlbench's Perl_av_reify, av.c, one instruction longer under -g).
; The same function with and without its debug information must give the same
; instructions.
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 < %s | FileCheck %s
; RUN: opt -strip-debug %s | llc -mtriple=capstone64 -mattr=+m -O2 | FileCheck %s

; CHECK-LABEL: same_address:
; CHECK-NOT: {{addi|mv}}
; CHECK: bne
; CHECK-NOT: {{addi|mv}}
; CHECK: cjalr zero, 0(ra)

define i64 @same_address(ptr addrspace(200) %p, i64 %x) addrspace(200) !dbg !5 {
entry:
  %a = ptrtoint ptr addrspace(200) %p to i64, !dbg !11
    #dbg_value(i64 %a, !10, !DIExpression(), !11)
  %c = icmp eq i64 %a, %x, !dbg !11
  br i1 %c, label %yes, label %no, !dbg !11

yes:
  ret i64 1, !dbg !11

no:
  ret i64 0, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "c87", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "c87.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"Dwarf Version", i32 5}
!5 = distinct !DISubprogram(name: "same_address", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !9)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8}
!8 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DILocalVariable(name: "a", scope: !5, file: !1, line: 2, type: !8)
!11 = !DILocation(line: 2, column: 3, scope: !5)
