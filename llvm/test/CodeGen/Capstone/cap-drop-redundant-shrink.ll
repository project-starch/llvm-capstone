; -capstone-drop-redundant-shrink (default off): a data global's SHRINKTO, or
; li + add + SHRINK, is dropped when every reader of the narrowed capability is a
; load or store using it as the base at a constant offset, with the whole access
; inside the object. No access through it can leave the object, so the
; narrowing checks nothing. Anything else keeps it: an offset past the end, the
; capability stored as a value or passed to a call, a variable index, or an
; offset too large for an immediate (the access goes through cincoffset).
;
; MUTATION: drop the `Offset + W > Size` test in onlyInBoundsBase -> load_oob
; loses its shrinkto and the ON check there fires; drop the stored-value test
; -> store_cap loses it; flip the option's default -> the OFF checks of
; load_ptr fire.
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-free -capstone-image-gp \
; RUN:   -capstone-shrinkto-globals -capstone-shrinkto-end-adjust=1 < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,OFF
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-free -capstone-image-gp \
; RUN:   -capstone-shrinkto-globals -capstone-shrinkto-end-adjust=1 \
; RUN:   -capstone-drop-redundant-shrink < %s | FileCheck %s --check-prefixes=CHECK,ON,DEF
; The size is the immediate minus the adjustment: with adjustment 0, a 16-byte
; pointer global is `shrinkto 16` and is still dropped.
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-free -capstone-image-gp \
; RUN:   -capstone-shrinkto-globals -capstone-drop-redundant-shrink < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,ON,DEF
; A declaration's size is this unit's belief, not the linked object's: by default
; it keeps its narrowing; -capstone-drop-redundant-shrink-declarations drops it.
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-gp-free -capstone-image-gp \
; RUN:   -capstone-shrinkto-globals -capstone-shrinkto-end-adjust=1 \
; RUN:   -capstone-drop-redundant-shrink -capstone-drop-redundant-shrink-declarations \
; RUN:   < %s | FileCheck %s --check-prefixes=CHECK,ON,DECL

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
@p = addrspace(200) global ptr addrspace(200) null, align 16
@a = addrspace(200) global [4 x i32] zeroinitializer, align 4
@o = addrspace(200) global [4 x i32] zeroinitializer, align 4
@e = addrspace(200) global i64 0, align 8
@c = addrspace(200) global i64 0, align 8
@v = addrspace(200) global [8 x i32] zeroinitializer, align 4
@big = addrspace(200) global [3000 x i8] zeroinitializer, align 1
@x = addrspace(200) global i32 0, align 4
@ext = external addrspace(200) global [64 x i8], align 1
@wk = weak addrspace(200) global [16 x i8] zeroinitializer, align 1
@far = addrspace(200) global [3000 x i8] zeroinitializer, align 1
declare void @use(ptr addrspace(200)) addrspace(200)

; CHECK-LABEL: load_ptr:
; OFF: shrinkto
; ON-NOT: shrink
; ON: ldc {{a[0-9]+}}, 0({{a[0-9]+}})
define ptr addrspace(200) @load_ptr() addrspace(200) {
  %v = load ptr addrspace(200), ptr addrspace(200) @p
  ret ptr addrspace(200) %v
}
; CHECK-LABEL: load_elem:
; OFF: shrinkto
; ON-NOT: shrink
; ON: lw {{a[0-9]+}}, 12({{a[0-9]+}})
define i32 @load_elem() addrspace(200) {
  %q = getelementptr [4 x i32], ptr addrspace(200) @a, i64 0, i64 3
  %v = load i32, ptr addrspace(200) %q
  ret i32 %v
}
; CHECK-LABEL: load_oob:
; CHECK: shrinkto
define i32 @load_oob() addrspace(200) {
  %q = getelementptr i8, ptr addrspace(200) @o, i64 16
  %v = load i32, ptr addrspace(200) %q
  ret i32 %v
}
; CHECK-LABEL: store_cap:
; CHECK: shrinkto
define void @store_cap(ptr addrspace(200) %d) addrspace(200) {
  store ptr addrspace(200) @e, ptr addrspace(200) %d
  ret void
}
; CHECK-LABEL: call_arg:
; CHECK: shrinkto
define void @call_arg() addrspace(200) {
  call void @use(ptr addrspace(200) @c)
  ret void
}
; CHECK-LABEL: var_index:
; CHECK: shrinkto
define i32 @var_index(i64 %i) addrspace(200) {
  %q = getelementptr [8 x i32], ptr addrspace(200) @v, i64 0, i64 %i
  %r = load i32, ptr addrspace(200) %q
  ret i32 %r
}
; CHECK-LABEL: load_big:
; OFF: shrink
; ON-NOT: {{^[[:space:]]+(add|shrink)[[:space:]]}}
; ON: lbu {{a[0-9]+}}, 2000({{a[0-9]+}})
define i8 @load_big() addrspace(200) {
  %q = getelementptr [3000 x i8], ptr addrspace(200) @big, i64 0, i64 2000
  %r = load i8, ptr addrspace(200) %q
  ret i8 %r
}
; CHECK-LABEL: load_far:
; CHECK: {{^[[:space:]]+shrink[[:space:]]}}
define i8 @load_far() addrspace(200) {
  %q = getelementptr [3000 x i8], ptr addrspace(200) @far, i64 0, i64 2999
  %r = load i8, ptr addrspace(200) %q
  ret i8 %r
}
; CHECK-LABEL: two_blocks:
; OFF: shrinkto
; ON-NOT: shrink
define i32 @two_blocks(i1 %b) addrspace(200) {
entry:
  %v0 = load i32, ptr addrspace(200) @x
  br i1 %b, label %t, label %f
t:
  store i32 1, ptr addrspace(200) @x
  ret i32 %v0
f:
  %v1 = load i32, ptr addrspace(200) @x
  ret i32 %v1
}

; CHECK-LABEL: load_extern:
; OFF: shrinkto
; DEF: shrinkto
; DECL-NOT: shrinkto
; DECL: lbu {{a[0-9]+}}, 20({{a[0-9]+}})
; Declarations mode records the assumption right after the function, in a
; section SHF_LINK_ORDER ("o") to the function's own text section, so that
; --gc-sections drops the record with the function: the declared global and the
; bytes from its start the remaining accesses reach (ext[20], one byte: 21), for
; capstone-cc's link-time check. The default mode records nothing.
; DECL: .section .capstone.size_assume,"o",@progbits,.text
; DECL-NEXT: .quad 21
; DECL-NEXT: .ascii "ext"
; DECL-NEXT: .byte 0
; OFF-NOT: .capstone.size_assume
; DEF-NOT: .capstone.size_assume
define i8 @load_extern() addrspace(200) {
  %q = getelementptr [64 x i8], ptr addrspace(200) @ext, i64 0, i64 20
  %r = load i8, ptr addrspace(200) %q
  ret i8 %r
}

; A weak definition may be replaced by a larger or smaller one at link: never
; dropped, in either mode.
; CHECK-LABEL: load_weak:
; CHECK: shrinkto
define i8 @load_weak() addrspace(200) {
  %q = getelementptr [16 x i8], ptr addrspace(200) @wk, i64 0, i64 3
  %r = load i8, ptr addrspace(200) %q
  ret i8 %r
}

; DEF-NOT: .capstone.size_assume
; OFF-NOT: .capstone.size_assume
