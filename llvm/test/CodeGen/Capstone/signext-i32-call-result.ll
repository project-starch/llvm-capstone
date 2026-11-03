; An i32 the ABI marks signext arrives sign-extended (clang gives every 32-bit
; integer argument and result on capstone64 the attribute, as on riscv64), so
; the caller uses the call result as a 64-bit value without a sext.w, and the
; callee extends what it returns. The control: without signext the caller
; extends.
; RUN: %llc_cap < %s | FileCheck %s
; RUN: %llc_cap -O0 < %s -o /dev/null

; CHECK-LABEL: count_below:
; CHECK: jalr
; CHECK-NOT: sext.w
; CHECK: {{ret|cjalr zero, 0\(ra\)}}
define i64 @count_below(ptr addrspace(200) %cmp, ptr addrspace(200) %a, ptr addrspace(200) %b) addrspace(200) {
  %r = call signext addrspace(200) i32 %cmp(ptr addrspace(200) %a, ptr addrspace(200) %b)
  %w = sext i32 %r to i64
  %s = add i64 %w, 7
  ret i64 %s
}

; CHECK-LABEL: count_below_any:
; CHECK: jalr
; CHECK: sext.w
; CHECK: {{ret|cjalr zero, 0\(ra\)}}
define i64 @count_below_any(ptr addrspace(200) %cmp, ptr addrspace(200) %a, ptr addrspace(200) %b) addrspace(200) {
  %r = call addrspace(200) i32 %cmp(ptr addrspace(200) %a, ptr addrspace(200) %b)
  %w = sext i32 %r to i64
  %s = add i64 %w, 7
  ret i64 %s
}

; CHECK-LABEL: narrow:
; CHECK: sext.w a0, a0
; CHECK-NEXT: {{ret|cjalr zero, 0\(ra\)}}
define signext i32 @narrow(i64 %x) addrspace(200) {
  %t = trunc i64 %x to i32
  ret i32 %t
}
