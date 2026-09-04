// The shipping aggregate ABI is DefaultABIInfo: every struct is passed byval
// and returned sret, by reference in a capability register.  What matters on
// this target is that a capability MEMBER keeps its tag along every route --
// the sret store, the byval->sret copy, and a va_arg fetch (which yields a
// REFERENCE to the caller's copy).  Each is pinned to its capability
// load/store below; an integer member goes through ld/sd.  Measured
// 2026-09-04 on the branch tools.
//
// MUTATION: (stc in mk) make both members `long` -> mk stores `sd a1, 0(a0)`
// and the `stc a1, 0(a0)` line fails (performed 2026-09-04).
// MUTATION: (va_arg negative) the implicit-check-not guards a compiler
// regression (the va_arg once produced an addrspace-0 `ptr`, fixed in
// beab9348); demonstrated by rewriting the emitted IR's `, ptr addrspace(200)`
// to `, ptr` and re-running FileCheck, which then fails (performed 2026-09-04).
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=IR --implicit-check-not='{{va_arg .*, ptr$}}'
// RUN: %clang_cc1 -triple capstone64-unknown-elf -O2 -S -o - %s | FileCheck %s --check-prefix=ASM
// RUN: %clang_cc1 -triple capstone64-unknown-elf -O0 -S -o /dev/null %s
// RUN: %clang_cc1 -triple capstone64-unknown-elf -Os -S -o /dev/null %s

struct S { void *p; long n; };

// IR-LABEL: define {{.*}} @mk(ptr addrspace(200) {{.*}}sret(%struct.S) align 16 {{.*}}%agg.result, ptr addrspace(200) {{.*}}%p, i64 {{.*}}%n)
// IR: store ptr addrspace(200) %p, ptr addrspace(200) %agg.result, align 16
// ASM-LABEL: mk:
// ASM: stc a1, 0(a0)
// ASM-NEXT: sd a2, 16(a0)
// ASM-NEXT: sd zero, 24(a0)
// ASM-NEXT: cjalr zero, 0(ra)
struct S mk(void *p, long n) { struct S s = {p, n}; return s; }

// IR-LABEL: define {{.*}} @use(ptr addrspace(200) {{.*}}byval(%struct.S) align 16 {{.*}}%s)
// ASM-LABEL: use:
// ASM: ld a1, 0(a0)
// ASM-NEXT: ld a0, 16(a0)
long use(struct S s) { return s.n + (s.p != 0); }

// ASM-LABEL: first:
// ASM: ldc a0, 0(a0)
// ASM-NEXT: cjalr zero, 0(ra)
void *first(struct S *s) { return s->p; }

// IR-LABEL: define {{.*}} @pass(ptr addrspace(200) {{.*}}sret(%struct.S) {{.*}}, ptr addrspace(200) {{.*}}byval(%struct.S) {{.*}})
// IR: call addrspace(200) void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) {{.*}}%agg.result, ptr addrspace(200) {{.*}}%s, i64 32, i1 false)
// ASM-LABEL: pass:
// ASM: ldc a2, 16(a1)
// ASM-NEXT: ldc a1, 0(a1)
// ASM-NEXT: stc a2, 16(a0)
// ASM-NEXT: stc a1, 0(a0)
// ASM-NEXT: cjalr zero, 0(ra)
struct S pass(struct S s) { return s; }

// IR-LABEL: define {{.*}} @va_second(
// IR: va_arg ptr addrspace(200) %{{.*}}, ptr addrspace(200)
// ASM-LABEL: va_second:
// ASM: ld a0, 16(a1)
long va_second(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  struct S s = __builtin_va_arg(ap, struct S);
  __builtin_va_end(ap);
  return s.n;
}

// IR-LABEL: define {{.*}} @va_ptr(
// IR: va_arg ptr addrspace(200) %{{.*}}, ptr addrspace(200)
// ASM-LABEL: va_ptr:
// ASM: movc a0, a1
void *va_ptr(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  void *p = __builtin_va_arg(ap, void *);
  __builtin_va_end(ap);
  return p;
}
