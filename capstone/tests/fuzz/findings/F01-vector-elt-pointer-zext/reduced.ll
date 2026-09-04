target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64"

define i64 @autogen_SD1(i32 %L) addrspace(200) {
BB:
  %E65 = extractelement <8 x i64> zeroinitializer, i32 %L
  ret i64 %E65
}
