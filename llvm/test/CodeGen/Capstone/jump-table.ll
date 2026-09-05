; W-17: a dense switch lowers to a jump table in .rodata, and the generic BR_JT
; expansion computed `table + index*4` in the AS0 pointer type -- a 64-bit
; integer -- and loaded the entry through it (`lui/addi %hi/%lo(.LJTI)`, `add`,
; `lw a0, 0(a0)` with a scalar base).  On QEMU and on silicon that is "Cap mem
; access requires capability, rs1 = x10"; every production build carried
; -fno-jump-tables to avoid it.  The fix builds the table address as a
; capability (LGA, as constant pools do) and steps it with cincoffset.
;
; Under -capstone-gp-captable no capability reaches .rodata at all (gp is
; bounded to the cap table), so the backend refuses jump tables there and the
; switch lowers to compares; the table must not exist in that arm.
;
; MUTATION: revert lowerBR_JT to the Expand action -> `%hi(.LJTI0_0)` reappears
; and the CHECK-NOT fires; drop areJTsAllowed -> `.LJTI0_0:` appears in the
; CAPTABLE arm and its CHECK-NOT fires.
;
; RUN: %llc_cap -O0 < %s | FileCheck %s
; RUN: %llc_cap -O1 < %s | FileCheck %s
; RUN: %llc_cap -O2 < %s | FileCheck %s
; RUN: %llc_cap -O2 -capstone-gp-captable < %s | FileCheck %s --check-prefix=CAPTABLE
; RUN: %llc_cap -O2 -capstone-gp-captable -capstone-gp-captable-jump-tables=false < %s | FileCheck %s --check-prefix=CAPTABLE
; The refusal is what removes the table, not something else about the ABI:
; with the experiment knob the table comes back.
; RUN: %llc_cap -O2 -capstone-gp-captable -capstone-gp-captable-jump-tables < %s | FileCheck %s --check-prefix=CAPJT

define i64 @ten_way(i64 %x, ptr addrspace(200) %p) {
; CAPJT-LABEL: ten_way:
; CAPJT: %pcrel_hi(.LJTI0_0)
; CAPJT: jr
; CAPJT: .LJTI0_0:
; CHECK-LABEL: ten_way:
; CHECK-NOT: %hi(.LJTI0_0)
; CHECK: auipc {{[a-z0-9]+}}, %pcrel_hi(.LJTI0_0)
; CHECK: cincoffset {{[a-z0-9]+}}, gp, {{[a-z0-9]+}}
; CHECK-NEXT: delin
; CHECK: cincoffset {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK-NEXT: lw {{[a-z0-9]+}}, 0({{[a-z0-9]+}})
; The entry is a label difference; the dispatch adds the table's runtime
; address (the capability's cursor) before the jump. An absolute entry would be
; the link-time address, which a domain does not execute at.
; CHECK: add
; CHECK: jr
; CHECK: .LJTI0_0:
; CHECK-NEXT: .word .LBB0_{{[0-9]+}}-.LJTI0_0
; An absolute entry (`.word .LBB0_n` with nothing after it) must not appear. The
; end-of-line anchor has to live inside the regex braces: a bare `$` is a
; literal dollar to FileCheck and the guard never fires (found by audit).
; CHECK-NOT: {{\.word \.LBB0_[0-9]+$}}
;
; CAPTABLE-LABEL: ten_way:
; CAPTABLE-NOT: .LJTI0_0
; CAPTABLE-NOT: jr
; CAPTABLE: .Lfunc_end0:
entry:
  switch i64 %x, label %default [
    i64 0, label %c0
    i64 1, label %c1
    i64 2, label %c2
    i64 3, label %c3
    i64 4, label %c4
    i64 5, label %c5
    i64 6, label %c6
    i64 7, label %c7
    i64 8, label %c8
    i64 9, label %c9
  ]
c0:
  store i64 10, ptr addrspace(200) %p
  ret i64 100
c1:
  store i64 11, ptr addrspace(200) %p
  ret i64 101
c2:
  store i64 12, ptr addrspace(200) %p
  ret i64 102
c3:
  store i64 13, ptr addrspace(200) %p
  ret i64 103
c4:
  store i64 14, ptr addrspace(200) %p
  ret i64 104
c5:
  store i64 15, ptr addrspace(200) %p
  ret i64 105
c6:
  store i64 16, ptr addrspace(200) %p
  ret i64 106
c7:
  store i64 17, ptr addrspace(200) %p
  ret i64 107
c8:
  store i64 18, ptr addrspace(200) %p
  ret i64 108
c9:
  store i64 19, ptr addrspace(200) %p
  ret i64 109
default:
  ret i64 -1
}
