; CapstoneRecoverProvenance: an address computed through an integer from ONE
; capability comes back as that capability moved (a GEP), not as an untagged
; inttoptr. Everything else keeps the IR's answer.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -stop-after=capstone-provenance < %s \
; RUN:   | FileCheck %s --check-prefix=IR
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -stop-after=capstone-provenance < %s \
; RUN:   | FileCheck %s --check-prefix=IR
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=capstone64 -mattr=+m -capstone-recover-provenance=false \
; RUN:     -stop-after=capstone-provenance < %s | FileCheck %s --check-prefix=OFF
;
; MUTATION: drop the `Sub` case's carriesAddress() test in the pass -> @difference
; is rewritten into a GEP on %p and its IR-NOT line fails. Make walkSlot() return
; without walking -> @slot_flag and @slot_cursor keep their inttoptr. Walk only
; Instructions again (not constant expressions) -> @global_align and
; @global_constant keep theirs.

; OFF-LABEL: @align_up(
; OFF: inttoptr
; IR-LABEL: @align_up(
; IR-NOT:   inttoptr
; IR:       %[[D:.*]] = sub i64 %{{.*}}, %{{.*}}
; IR:       getelementptr i8, ptr addrspace(200) %p, i64 %[[D]]
; ASM-LABEL: align_up:
; ASM:       cincoffset a0, a0,
define ptr addrspace(200) @align_up(ptr addrspace(200) %p) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %a = add i64 %i, 63
  %m = and i64 %a, -64
  %r = inttoptr i64 %m to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; The clang shape: the i128 carrier, truncated to the address and widened back.
; IR-LABEL: @carrier(
; IR-NOT:   inttoptr
; IR:       getelementptr i8, ptr addrspace(200) %p,
define ptr addrspace(200) @carrier(ptr addrspace(200) %p, i64 %n) {
  %w = ptrtoint ptr addrspace(200) %p to i128
  %t = trunc i128 %w to i64
  %s = add i64 %t, %n
  %z = zext i64 %s to i128
  %r = inttoptr i128 %z to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; An offset built with a multiply is a plain integer: fine.
; IR-LABEL: @scaled_index(
; IR-NOT:   inttoptr
; IR:       getelementptr i8, ptr addrspace(200) %p,
define ptr addrspace(200) @scaled_index(ptr addrspace(200) %p, i64 %i) {
  %a = ptrtoint ptr addrspace(200) %p to i64
  %o = mul i64 %i, 8
  %s = add i64 %a, %o
  %r = inttoptr i64 %s to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; A flag set or cleared on one pointer, chosen by a select.
; IR-LABEL: @flag_select(
; IR-NOT:   inttoptr
; IR:       getelementptr i8, ptr addrspace(200) %p,
define ptr addrspace(200) @flag_select(ptr addrspace(200) %p, i1 %c) {
  %a = ptrtoint ptr addrspace(200) %p to i64
  %set = or i64 %a, 1
  %clr = and i64 %a, -2
  %v = select i1 %c, i64 %set, i64 %clr
  %r = inttoptr i64 %v to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; A uintptr_t cursor stepped in a loop, dereferenced each iteration.
; IR-LABEL: @walk(
; IR-NOT:   inttoptr
; IR:       getelementptr i8, ptr addrspace(200) %p,
; IR:       load i8
define i8 @walk(ptr addrspace(200) %p, i64 %n) {
entry:
  %start = ptrtoint ptr addrspace(200) %p to i64
  br label %loop
loop:
  %cur = phi i64 [ %start, %entry ], [ %next, %loop ]
  %acc = phi i8 [ 0, %entry ], [ %sum, %loop ]
  %q = inttoptr i64 %cur to ptr addrspace(200)
  %b = load i8, ptr addrspace(200) %q
  %sum = add i8 %acc, %b
  %next = add i64 %cur, 1
  %k = sub i64 %next, %start
  %done = icmp uge i64 %k, %n
  br i1 %done, label %exit, label %loop
exit:
  ret i8 %sum
}

; Two pointers: no single owner. Left alone.
; IR-LABEL: @two_sources(
; IR:       inttoptr
define ptr addrspace(200) @two_sources(ptr addrspace(200) %p, ptr addrspace(200) %q) {
  %a = ptrtoint ptr addrspace(200) %p to i64
  %b = ptrtoint ptr addrspace(200) %q to i64
  %x = xor i64 %a, %b
  %r = inttoptr i64 %x to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; p - q + r: the difference is a distance, not a pointer. Left alone.
; IR-LABEL: @difference(
; IR-NOT:   getelementptr
; IR:       inttoptr
define ptr addrspace(200) @difference(ptr addrspace(200) %p, ptr addrspace(200) %q, i64 %r) {
  %a = ptrtoint ptr addrspace(200) %p to i64
  %b = ptrtoint ptr addrspace(200) %q to i64
  %d = sub i64 %b, %a
  %s = add i64 %d, %r
  %v = inttoptr i64 %s to ptr addrspace(200)
  ret ptr addrspace(200) %v
}

; An address shifted is no longer that pointer's address moved. Left alone.
; IR-LABEL: @shifted(
; IR:       inttoptr
define ptr addrspace(200) @shifted(ptr addrspace(200) %p) {
  %a = ptrtoint ptr addrspace(200) %p to i64
  %s = shl i64 %a, 1
  %r = inttoptr i64 %s to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; An integer from memory or from the caller: nothing in the function knows
; where it came from. Left alone (-Wcapstone-pointer-roundtrip's case).
; IR-LABEL: @from_memory(
; IR:       inttoptr
define ptr addrspace(200) @from_memory(ptr addrspace(200) %slot) {
  %v = load i64, ptr addrspace(200) %slot
  %m = and i64 %v, -4
  %r = inttoptr i64 %m to ptr addrspace(200)
  ret ptr addrspace(200) %r
}
; IR-LABEL: @from_argument(
; IR:       inttoptr
define ptr addrspace(200) @from_argument(i64 %v) {
  %r = inttoptr i64 %v to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; The one source does not dominate the cast (it is loaded on one path only).
; IR-LABEL: @not_dominating(
; IR:       inttoptr
define ptr addrspace(200) @not_dominating(ptr addrspace(200) %slot, i1 %c) {
entry:
  br i1 %c, label %have, label %join
have:
  %p = load ptr addrspace(200), ptr addrspace(200) %slot
  %a = ptrtoint ptr addrspace(200) %p to i64
  br label %join
join:
  %v = phi i64 [ %a, %have ], [ 64, %entry ]
  %r = inttoptr i64 %v to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; clang marks every function optnone at -O0. The pass must still run: it is what
; keeps an -O0 round trip from trapping, not an optimization.
; IR-LABEL: @optnone_align(
; IR-NOT:   inttoptr
; IR:       getelementptr i8, ptr addrspace(200) %p,
define ptr addrspace(200) @optnone_align(ptr addrspace(200) %p) #0 {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %m = and i64 %i, -16
  %r = inttoptr i64 %m to ptr addrspace(200)
  ret ptr addrspace(200) %r
}
attributes #0 = { noinline optnone }

; A global's address is a constant expression, and at -O2 the round trip folds
; around it: `(((uintptr_t)(buf + 3) + 63) & -64` keeps only the mask as an
; instruction. The constant ptrtoint inside is the source.
; IR-LABEL: @global_align(
; IR-NOT:   inttoptr
; IR:       getelementptr i8, ptr addrspace(200) getelementptr {{.*}}@g, i64 3)
@g = internal addrspace(200) global [200 x i8] zeroinitializer
define ptr addrspace(200) @global_align() {
  %m = and i64 add (i64 ptrtoint (ptr addrspace(200) getelementptr (i8, ptr addrspace(200) @g, i64 3) to i64), i64 63), -64
  %r = inttoptr i64 %m to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; A round trip of constants only, `(char *)((uintptr_t)buf + 8)`, is a constant
; expression at every -O level and never an instruction: it is expanded so the
; rewrite reaches it.
; IR-LABEL: @global_constant(
; IR-NOT:   inttoptr
; IR:       getelementptr {{.*}}@g, i64 8
define void @global_constant() {
  store i8 1, ptr addrspace(200) inttoptr (i64 add (i64 ptrtoint (ptr addrspace(200) @g to i64), i64 8) to ptr addrspace(200))
  ret void
}

; A constant address with no capability in it is an address and nothing more.
; Left alone, and not expanded.
; IR-LABEL: @absolute_address(
; IR:       store i8 1, ptr addrspace(200) inttoptr (i64 4096
define void @absolute_address() {
  store i8 1, ptr addrspace(200) inttoptr (i64 4096 to ptr addrspace(200))
  ret void
}

; At -O0 every local is a stack slot, so `t = (uintptr_t)n | 1;
; m = (T *)(t & ~1)` reaches the cast through a load. The slot is only ever
; stored to and loaded from, so it holds what was stored: n's address, moved.
; IR-LABEL: @slot_flag(
; IR-NOT:   inttoptr
; IR:       getelementptr i8, ptr addrspace(200) %n,
define ptr addrspace(200) @slot_flag(ptr addrspace(200) %n) #0 {
  %t = alloca i64, align 8, addrspace(200)
  %a = ptrtoint ptr addrspace(200) %n to i64
  %s = or i64 %a, 1
  store i64 %s, ptr addrspace(200) %t
  %l = load i64, ptr addrspace(200) %t
  %c = and i64 %l, -2
  %m = inttoptr i64 %c to ptr addrspace(200)
  ret ptr addrspace(200) %m
}

; A cursor kept in a slot and stepped in a loop: every store is the start or the
; cursor plus 8, so the one source is the array.
; IR-LABEL: @slot_cursor(
; IR-NOT:   inttoptr
; IR:       getelementptr i8, ptr addrspace(200) %p,
define i64 @slot_cursor(ptr addrspace(200) %p, i64 %end) #0 {
entry:
  %c = alloca i64, align 8, addrspace(200)
  %a = ptrtoint ptr addrspace(200) %p to i64
  store i64 %a, ptr addrspace(200) %c
  br label %loop
loop:
  %cur = load i64, ptr addrspace(200) %c
  %q = inttoptr i64 %cur to ptr addrspace(200)
  %v = load i64, ptr addrspace(200) %q
  %cur2 = load i64, ptr addrspace(200) %c
  %next = add i64 %cur2, 8
  store i64 %next, ptr addrspace(200) %c
  %done = icmp uge i64 %next, %end
  br i1 %done, label %exit, label %loop
exit:
  ret i64 %v
}

; A slot whose address is passed on may be written by the callee: it may hold
; anything. Left alone.
; IR-LABEL: @slot_escapes(
; IR:       inttoptr
declare void @take(ptr addrspace(200))
define ptr addrspace(200) @slot_escapes(ptr addrspace(200) %n) #0 {
  %t = alloca i64, align 8, addrspace(200)
  %a = ptrtoint ptr addrspace(200) %n to i64
  store i64 %a, ptr addrspace(200) %t
  call void @take(ptr addrspace(200) %t)
  %l = load i64, ptr addrspace(200) %t
  %m = inttoptr i64 %l to ptr addrspace(200)
  ret ptr addrspace(200) %m
}

; An address taken with the cursor intrinsic is a call's result, not a source:
; this is how code keeps an address that must carry no authority (a zero-byte
; allocation in the whisper port). Left alone.
; IR-LABEL: @explicit_address(
; IR:       inttoptr
declare i64 @llvm.capstone.cap.get.cursor.p200(ptr addrspace(200))
define ptr addrspace(200) @explicit_address(ptr addrspace(200) %p) {
  %a = call i64 @llvm.capstone.cap.get.cursor.p200(ptr addrspace(200) %p)
  %r = inttoptr i64 %a to ptr addrspace(200)
  ret ptr addrspace(200) %r
}
