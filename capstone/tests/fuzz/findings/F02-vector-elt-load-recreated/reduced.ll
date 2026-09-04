target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64"

define i8 @autogen_SD32(i32 %0) addrspace(200) {
BB:
  %E6 = extractelement <1 x i32> zeroinitializer, i32 %0
  %E13 = extractelement <4 x i8> zeroinitializer, i32 %E6
  ret i8 %E13
}
