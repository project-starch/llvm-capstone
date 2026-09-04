target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64"

define <8 x i8> @autogen_SD155(i32 %0) addrspace(200) {
BB:
  %E35 = extractelement <1 x i32> zeroinitializer, i32 %0
  %I37 = insertelement <8 x i8> zeroinitializer, i8 1, i32 %E35
  ret <8 x i8> %I37
}
