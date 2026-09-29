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
; The `nonnull` on the parameters below is load-bearing, not decoration: a base
; that MAY BE NULL is left alone, because `cincoffset` raises UNEXPECTED_OPERAND
; on a register holding no capability and null holds none. @maybe_null_align
; pins that, @checked_align pins that a null test is enough, and
; @identity_may_be_null pins the one exemption -- a rewrite with no offset emits
; no cincoffset at all.
;
; MUTATION: drop the `Sub` case's carriesAddress() test in the pass -> @difference
; is rewritten into a GEP on %p and its IR-NOT line fails. Make walkSlot() return
; without walking -> @slot_flag and @slot_cursor keep their inttoptr. Walk only
; Instructions again (not constant expressions) -> @global_align and
; @global_constant keep theirs. Each of the three below was run: let joinInputs()
; take the UNION of its inputs again -> @select_foreign, @phi_foreign and
; @slot_foreign are rewritten and their IR lines fail; drop the
; holdsCapability() gate -> @maybe_null_align, @carrier_mask_align and
; @heap_no_deref are rewritten (@select_fallback needs BOTH mutations, its
; fallback arm being foreign as well); look through a
; narrowing cast in linearOffset() again -> @truncated_address folds to `%p` and
; loses its GEP; make isPlainAddressOf() refuse the carrier mask ->
; @carrier_mask_identity is left to the null gate and keeps its inttoptr, which
; is the shape that broke musl's atexit() in QEMU before it was pinned here; drop
; the dominating-dereference rule in holdsCapability() -> @heap_deref keeps its
; inttoptr; let two different sources agree in joinInputs() -> @select_two_sources
; is rewritten on whichever arm came last; drop the alloca, the extern-weak test
; or the stack-slot look-through in holdsCapability() -> @alloca_align,
; @weak_global_align and @slot_holds_global respectively. All nine were run
; against the build this branch was gated with.

; OFF-LABEL: @align_up(
; OFF: inttoptr
; IR-LABEL: @align_up(
; IR-NOT:   inttoptr
; IR:       %[[D:.*]] = sub i64 %{{.*}}, %{{.*}}
; IR:       getelementptr i8, ptr addrspace(200) %p, i64 %[[D]]
; ASM-LABEL: align_up:
; ASM:       cincoffset a0, a0,
define ptr addrspace(200) @align_up(ptr addrspace(200) nonnull %p) {
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
define ptr addrspace(200) @carrier(ptr addrspace(200) nonnull %p, i64 %n) {
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
define ptr addrspace(200) @scaled_index(ptr addrspace(200) nonnull %p, i64 %i) {
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
define ptr addrspace(200) @flag_select(ptr addrspace(200) nonnull %p, i1 %c) {
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
define i8 @walk(ptr addrspace(200) nonnull %p, i64 %n) {
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
define ptr addrspace(200) @optnone_align(ptr addrspace(200) nonnull %p) #0 {
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
define ptr addrspace(200) @slot_flag(ptr addrspace(200) nonnull %n) #0 {
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
define i64 @slot_cursor(ptr addrspace(200) nonnull %p, i64 %end) #0 {
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

; A source that MAY BE NULL is left alone: `cincoffset` raises UNEXPECTED_OPERAND
; on a base that holds no capability (capstone_flu_unit.anvil), and null holds
; none, so moving it would trap where the untagged answer merely returned an
; address nobody used.
; IR-LABEL: @maybe_null_align(
; IR:       inttoptr
define ptr addrspace(200) @maybe_null_align(ptr addrspace(200) %p) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %a = add i64 %i, 15
  %m = and i64 %a, -16
  %r = inttoptr i64 %m to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; The same pointer, past a null test: known non-null there, and moved.
; IR-LABEL: @checked_align(
; IR-NOT:   inttoptr
; IR:       getelementptr i8, ptr addrspace(200) %p,
define ptr addrspace(200) @checked_align(ptr addrspace(200) %p) {
entry:
  %z = icmp eq ptr addrspace(200) %p, null
  br i1 %z, label %out, label %move
move:
  %i = ptrtoint ptr addrspace(200) %p to i64
  %a = add i64 %i, 15
  %m = and i64 %a, -16
  %r = inttoptr i64 %m to ptr addrspace(200)
  ret ptr addrspace(200) %r
out:
  ret ptr addrspace(200) null
}

; An address that IS the source's asks for no move, so the rewrite is the pointer
; itself and emits no cincoffset: safe even where null is possible. This is
; musl's call(), `((void (*)(void))(uintptr_t)p)()`, the shape #86 had to
; override.
; IR-LABEL: @identity_may_be_null(
; IR-NOT:   inttoptr
; IR:       ret ptr addrspace(200) %p
define ptr addrspace(200) @identity_may_be_null(ptr addrspace(200) %p) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %r = inttoptr i64 %i to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; A select carries the source only when BOTH arms are that one pointer moved.
; Here the false arm is the caller's own integer: giving the result p's tag,
; bounds and permissions would hand out p's authority at an address p never
; held. Left alone.
; IR-LABEL: @select_foreign(
; IR:       inttoptr
define ptr addrspace(200) @select_foreign(i1 %c, ptr addrspace(200) nonnull %p, i64 %x) {
  %a = ptrtoint ptr addrspace(200) %p to i64
  %v = select i1 %c, i64 %a, i64 %x
  %r = inttoptr i64 %v to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; The fallback shape of the same thing: `p ? (uintptr_t)p : fallback`.
; IR-LABEL: @select_fallback(
; IR:       inttoptr
define ptr addrspace(200) @select_fallback(ptr addrspace(200) %p, i64 %fallback) {
  %z = icmp ne ptr addrspace(200) %p, null
  %a = ptrtoint ptr addrspace(200) %p to i64
  %v = select i1 %z, i64 %a, i64 %fallback
  %r = inttoptr i64 %v to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; A phi is the same rule across edges.
; IR-LABEL: @phi_foreign(
; IR:       inttoptr
define ptr addrspace(200) @phi_foreign(ptr addrspace(200) nonnull %p, i64 %x, i1 %c) {
entry:
  br i1 %c, label %fromp, label %join
fromp:
  %a = ptrtoint ptr addrspace(200) %p to i64
  br label %join
join:
  %v = phi i64 [ %a, %fromp ], [ %x, %entry ]
  %r = inttoptr i64 %v to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; And at -O0 the same mixture is two stores into one slot.
; IR-LABEL: @slot_foreign(
; IR:       inttoptr
define ptr addrspace(200) @slot_foreign(ptr addrspace(200) nonnull %p, i64 %x, i1 %c) #0 {
entry:
  %t = alloca i64, align 8, addrspace(200)
  %a = ptrtoint ptr addrspace(200) %p to i64
  store i64 %a, ptr addrspace(200) %t
  br i1 %c, label %other, label %join
other:
  store i64 %x, ptr addrspace(200) %t
  br label %join
join:
  %l = load i64, ptr addrspace(200) %t
  %r = inttoptr i64 %l to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; Two arms, one pointer: still that pointer moved.
; IR-LABEL: @select_same_source(
; IR-NOT:   inttoptr
; IR:       getelementptr i8, ptr addrspace(200) %p,
define ptr addrspace(200) @select_same_source(ptr addrspace(200) nonnull %p, i1 %c) {
  %a = ptrtoint ptr addrspace(200) %p to i64
  %u = add i64 %a, 8
  %d = add i64 %a, 16
  %v = select i1 %c, i64 %u, i64 %d
  %r = inttoptr i64 %v to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; A select over plain integers is an offset like any other and must not
; disqualify anything: the pointer is on the other side of the add.
; IR-LABEL: @select_plain_offset(
; IR-NOT:   inttoptr
; IR:       getelementptr i8, ptr addrspace(200) %p,
define ptr addrspace(200) @select_plain_offset(ptr addrspace(200) nonnull %p, i1 %c) {
  %a = ptrtoint ptr addrspace(200) %p to i64
  %o = select i1 %c, i64 8, i64 16
  %s = add i64 %a, %o
  %r = inttoptr i64 %s to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; `(uint32_t)(uintptr_t)p` is not p's address any more. The offset is built as
; the general `X - addr(p)`, so the result carries the address the program
; computed -- the truncated one. Looking through the narrowing cast instead
; would report offset 0 and rebuild p's own, untruncated address.
; IR-LABEL: @truncated_address(
; IR-NOT:   inttoptr
; IR:       sub i64
; IR:       getelementptr i8, ptr addrspace(200) %p, i64 %
define ptr addrspace(200) @truncated_address(ptr addrspace(200) nonnull %p) {
  %i = ptrtoint ptr addrspace(200) %p to i64
  %t = trunc i64 %i to i32
  %w = zext i32 %t to i64
  %r = inttoptr i64 %w to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; musl's call() as clang ACTUALLY emits it, and the reason the case above is not
; enough: `uintptr_t` is the address width, but the ptrtoint goes to the i128
; carrier and the narrowing is a MASK, not a truncation. The address is still
; p's own, so the answer is p itself -- no offset, no cincoffset, safe on a
; source that may be null. Reading the mask as ordinary arithmetic sends this to
; the null gate instead, a plain argument does not pass it, and musl's own
; atexit() goes back to calling through `mv`, which drops the tag (cause 24 at
; the cjalr; measured against an unmodified musl on 2026-09-28).
; IR-LABEL: @carrier_mask_identity(
; IR-NOT:   inttoptr
; IR:       ret ptr addrspace(200) %p
define ptr addrspace(200) @carrier_mask_identity(ptr addrspace(200) %p) {
  %i = ptrtoint ptr addrspace(200) %p to i128
  %m = and i128 %i, 18446744073709551615
  %r = inttoptr i128 %m to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; A mask that clears a bit OF THE ADDRESS is an alignment, not an identity: it
; moves the pointer, so on a source that may be null it is left alone.
; IR-LABEL: @carrier_mask_align(
; IR:       inttoptr
define ptr addrspace(200) @carrier_mask_align(ptr addrspace(200) %p) {
  %i = ptrtoint ptr addrspace(200) %p to i128
  %m = and i128 %i, -16
  %r = inttoptr i128 %m to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; A heap pointer qualifies through the program's OWN dereference: a load or store
; through it that dominates the cast would have trapped if it held no capability,
; so by the cast it holds one. Without this nothing a function allocates could
; ever be moved -- malloc may return null and nothing else says otherwise -- and
; the pass would stop recovering the flag and cursor shapes that heap-heavy code
; is made of.
declare ptr addrspace(200) @alloc(i64)
; IR-LABEL: @heap_deref(
; IR-NOT:   inttoptr
; IR:       getelementptr i8, ptr addrspace(200) %p,
define i64 @heap_deref() {
  %p = call ptr addrspace(200) @alloc(i64 16)
  store i64 77, ptr addrspace(200) %p
  %a = ptrtoint ptr addrspace(200) %p to i64
  %t = or i64 %a, 1
  %c = and i64 %t, -2
  %m = inttoptr i64 %c to ptr addrspace(200)
  %v = load i64, ptr addrspace(200) %m
  ret i64 %v
}

; The same allocation with nothing dereferenced before the cast proves nothing
; there, and is left alone.
; IR-LABEL: @heap_no_deref(
; IR:       inttoptr
define ptr addrspace(200) @heap_no_deref() {
  %p = call ptr addrspace(200) @alloc(i64 16)
  %a = ptrtoint ptr addrspace(200) %p to i64
  %c = and i64 %a, -16
  %m = inttoptr i64 %c to ptr addrspace(200)
  ret ptr addrspace(200) %m
}

; Two pointers meeting in a select have no single owner either -- @two_sources
; through a different operator.
; IR-LABEL: @select_two_sources(
; IR:       inttoptr
define ptr addrspace(200) @select_two_sources(ptr addrspace(200) nonnull %p, ptr addrspace(200) nonnull %q, i1 %c) {
  %a = ptrtoint ptr addrspace(200) %p to i64
  %b = ptrtoint ptr addrspace(200) %q to i64
  %v = select i1 %c, i64 %a, i64 %b
  %r = inttoptr i64 %v to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; A local object is a capability by construction, so its address is moved with no
; further proof -- the commonest source there is, and the one generic LLVM
; declines to vouch for outside address space 0.
; IR-LABEL: @alloca_align(
; IR-NOT:   inttoptr
; IR:       getelementptr i8, ptr addrspace(200) %b,
define ptr addrspace(200) @alloca_align() {
  %b = alloca [64 x i8], align 8, addrspace(200)
  %i = ptrtoint ptr addrspace(200) %b to i64
  %a = add i64 %i, 15
  %m = and i64 %a, -16
  %r = inttoptr i64 %m to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; At -O0 a local pointer lives in a slot and every use is a fresh load, which no
; query about an SSA value can answer. This slot holds only the global's address,
; so what the load returns is a capability. Without the look-through the whole
; -O0 arm is declined, which is the level this pass exists for.
; IR-LABEL: @slot_holds_global(
; IR-NOT:   inttoptr
; IR:       getelementptr i8, ptr addrspace(200) %l,
define ptr addrspace(200) @slot_holds_global() #0 {
  %s = alloca ptr addrspace(200), align 16, addrspace(200)
  store ptr addrspace(200) getelementptr (i8, ptr addrspace(200) @g, i64 3), ptr addrspace(200) %s
  %l = load ptr addrspace(200), ptr addrspace(200) %s
  %i = ptrtoint ptr addrspace(200) %l to i64
  %a = add i64 %i, 63
  %m = and i64 %a, -64
  %r = inttoptr i64 %m to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; The same slot holding a pointer nothing vouches for is not enough.
; IR-LABEL: @slot_holds_argument(
; IR:       inttoptr
define ptr addrspace(200) @slot_holds_argument(ptr addrspace(200) %p) #0 {
  %s = alloca ptr addrspace(200), align 16, addrspace(200)
  store ptr addrspace(200) %p, ptr addrspace(200) %s
  %l = load ptr addrspace(200), ptr addrspace(200) %s
  %i = ptrtoint ptr addrspace(200) %l to i64
  %a = add i64 %i, 63
  %m = and i64 %a, -64
  %r = inttoptr i64 %m to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; An extern-weak symbol's address may be null: not a capability by construction,
; and not moved.
@w = extern_weak addrspace(200) global i8
; IR-LABEL: @weak_global_align(
; IR:       inttoptr
define ptr addrspace(200) @weak_global_align() {
  %m = and i64 add (i64 ptrtoint (ptr addrspace(200) @w to i64), i64 63), -64
  %r = inttoptr i64 %m to ptr addrspace(200)
  ret ptr addrspace(200) %r
}
