target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"
define i128 @mixed_sign_arms(i32 %c) {
  %t = icmp ne i32 %c, 0
  %r = select i1 %t, i128 -1, i128 7
  ret i128 %r
}
