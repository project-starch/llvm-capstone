; RUN: not %clang_cc1 -triple capstone64-unknown-elf -target-feature +m \
; RUN:     -x ir -S -o /dev/null %s 2>&1 | FileCheck %s

; THE UNFORGEABLE-CONSTANT LIMIT MUST READ AS A COMPILER ERROR, NOT A CRASH.
;
; Companion to capstone-i128-backend-diagnostic.c, which covers the 128-bit shift.
; Same reasoning, different site: CodeGen/Capstone/cap-constants-invalid.ll pins these
; four messages, but it runs LLC, and llc was always clean -- one line, exit 1. Through
; clang, report_fatal_error still produced "PLEASE submit a bug report" plus a stack
; dump, so no test in the tree could observe what a user actually saw. They are now
; DiagnosticInfoUnsupported.
;
; THE ROUTE IS HAND-WRITTEN IR ON PURPOSE. The C routes to a >64-bit capability constant
; are closed -- __int128 and _BitInt(65..128) are rejected in the front end, and the
; store-merging route that made the SQLite amalgamation fail at -O1 is refused in
; canMergeStoresTo. As with the shift, the diagnostic is the thing under test, NOT the
; reachability: three source routes to the sibling site have been found so far and each
; time "nothing reaches this any more" was recorded it was wrong.

define ptr addrspace(200) @wide_const() {
entry:
  ret ptr addrspace(200) inttoptr (i128 18446744073709551616 to ptr addrspace(200))
}

; It must be a diagnostic...
; CHECK: error:
; ...that names the FUNCTION, so a failure in a large module can be localised at all...
; CHECK-SAME: in function wide_const
; ...says what the limit is...
; CHECK-SAME: Capstone PureCap: Cannot materialize arbitrary >64-bit constants as capabilities
; ...and prints the offending VALUE.
; CHECK-SAME: (value 0x10000000000000000)
; And it must NOT look like a compiler crash.
; CHECK-NOT: PLEASE submit a bug report
; CHECK-NOT: Stack dump
