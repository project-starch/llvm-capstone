; ModuleID = '/tmp/claude-1000/-home-diego-llvm-capstone/20f8ed33-3a71-4a17-a640-8127c6f7c29d/scratchpad/one.c'
source_filename = "/tmp/claude-1000/-home-diego-llvm-capstone/20f8ed33-3a71-4a17-a640-8127c6f7c29d/scratchpad/one.c"
target datalayout = "e-m:e-p:64:128-p200:128:128:128-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-unknown-elf"

; Function Attrs: noinline nounwind optnone
define dso_local i32 @g(i32 noundef %x) addrspace(200) #0 {
entry:
  %x.addr = alloca i32, align 4, addrspace(200)
  store i32 %x, ptr addrspace(200) %x.addr, align 4
  %0 = load i32, ptr addrspace(200) %x.addr, align 4
  %1 = call addrspace(200) i32 @llvm.cttz.i32(i32 %0, i1 true)
  ret i32 %1
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.cttz.i32(i32, i1 immarg) addrspace(200) #1

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+64bit,+i" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 22.0.0git (https://github.com/project-starch/llvm-capstone e45575bf24f2d0f6285d634c2e79f5fc1b9a0e65)"}
