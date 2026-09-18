; C-48: outgoing stack-passed varargs occupy 16-byte slots, because lowerVAARG
; advances by the 16-byte slot stride and the register save area is
; CXLenInBytes per register, written with STC. Before the fix in
; CapstoneCallingConv.cpp the caller packed them at 8 bytes, so a callee reading
; with va_arg skipped every second stack vararg: with three fixed arguments and
; eight int varargs the seventh vanished and the eighth took its place. Found by
; libc-test's inet_pton, whose inet_ntop prints eight %x in one snprintf.
;
; RUN: %llc_cap -O1 < %s | FileCheck %s
;
; Three fixed arguments plus eight int varargs: the first five varargs ride in
; a3-a7, the sixth, seventh and eighth go to the stack and must sit 16 apart.
; CHECK-LABEL: caller:
; CHECK-DAG: sd {{[a-z0-9]+}}, 0(sp)
; CHECK-DAG: sd {{[a-z0-9]+}}, 16(sp)
; CHECK-DAG: sd {{[a-z0-9]+}}, 32(sp)
; CHECK-NOT: sd {{[a-z0-9]+}}, 8(sp)

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-unknown-elf"

@.str = private unnamed_addr addrspace(200) constant [24 x i8] c"%x %x %x %x %x %x %x %x\00", align 1

; Function Attrs: nounwind
define dso_local i32 @caller(ptr addrspace(200) noundef %b) local_unnamed_addr addrspace(200) #0 {
entry:
  %call = tail call addrspace(200) i32 (ptr addrspace(200), i64, ptr addrspace(200), ...) @snp(ptr addrspace(200) noundef %b, i64 noundef 64, ptr addrspace(200) noundef @.str, i32 noundef 17, i32 noundef 34, i32 noundef 51, i32 noundef 68, i32 noundef 85, i32 noundef 102, i32 noundef 119, i32 noundef 136) #2
  ret i32 %call
}

declare dso_local i32 @snp(ptr addrspace(200) noundef, i64 noundef, ptr addrspace(200) noundef, ...) local_unnamed_addr addrspace(200) #1

attributes #0 = { nounwind "frame-pointer"="all" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+64bit,+i,+m,+zmmul" }
attributes #1 = { "frame-pointer"="all" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+64bit,+i,+m,+zmmul" }
attributes #2 = { nobuiltin nounwind "no-builtins" }


