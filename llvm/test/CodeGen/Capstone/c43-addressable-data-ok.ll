; C-43 negative control: data that DOES get a cap-table slot, and code that needs no anonymous pool,
; compile clean under -capstone-gp-captable -- the guard must NOT fire here, or it would reject every
; gp-captable build. Two cases:
;   - a NAMED global reached through `ldc gp[i]`, cheap addend materialised inline; and
;   - a PRIVATE unnamed_addr constant array. This is the verified closure of the C-43 private-global
;     residual: isGpCaptableGlobal has no linkage filter, so a private/anonymous GlobalVariable (a
;     SimplifyCFG switch.table, a private constant array) still gets a slot and is reached by `ldc`,
;     NOT by the faulting pcrel+cincoffset-gp form. So it is not part of the C-43 exposure.
; See ISSUES.md C-43.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-captable < %s 2>&1 | FileCheck %s

; CHECK-NOT: C-43
; CHECK-NOT: error:

@g = addrspace(200) global i64 42
@.tab = private unnamed_addr addrspace(200) constant [4 x i32] [i32 11, i32 22, i32 33, i32 44]

; CHECK-LABEL: cheap:
; CHECK: ldc
define i64 @cheap() {
  %v = load i64, ptr addrspace(200) @g
  %r = add i64 %v, 7
  ret i64 %r
}

; The private array gets a slot too: loaded via ldc gp[i], and a .capstone_gp_table
; entry is emitted for it -- no pcrel/scc into .rodata.
; CHECK-LABEL: lut:
; CHECK: ldc a{{[0-9]+}}, {{[0-9]+}}(gp)
define i32 @lut(i64 %i) {
  %p = getelementptr [4 x i32], ptr addrspace(200) @.tab, i64 0, i64 %i
  %v = load i32, ptr addrspace(200) %p
  ret i32 %v
}

; CHECK: .capstone_gp_table
