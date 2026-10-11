target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-unknown-elf"
define i64 @f(i64 %a, i1 %c) addrspace(200) {
  %x = select i1 %c, i64 %a, i64 0
  %y = select i1 %c, i64 0, i64 %x
  ret i64 %y
}
