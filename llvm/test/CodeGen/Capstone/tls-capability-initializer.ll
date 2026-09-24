; A thread-local whose initializer is the address of a global (C-47).
;
; Its initial value is part of the TLS template, which the runtime copies into
; the thread's block. A capability's tag cannot live in that image, and the
; capability-initializer pass cannot store it either: it runs once, at startup,
; before any thread's block exists. It used to skip thread-locals silently, and
; the variable held an untagged address that trapped on first use. A null
; initializer is fine: it needs no tag.
;
; RUN: not llc -mtriple=capstone64 -mattr=+m < %s -o /dev/null 2>&1 | FileCheck %s
; RUN: sed 's/@p = thread_local addrspace(200) global ptr addrspace(200) @g/@p = thread_local addrspace(200) global ptr addrspace(200) null/' %s \
; RUN:   | llc -mtriple=capstone64 -mattr=+m -o /dev/null

; CHECK: error: thread-local variable 'p' is initialized with the address of a global or function
; CHECK-NOT: error:

@g = addrspace(200) global i32 1
@p = thread_local addrspace(200) global ptr addrspace(200) @g, align 16
@n = thread_local addrspace(200) global ptr addrspace(200) null, align 16

define ptr addrspace(200) @get() {
  %v = load ptr addrspace(200), ptr addrspace(200) @p
  ret ptr addrspace(200) %v
}
