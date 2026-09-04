// Every __builtin_capstone_cap_* builtin, at -O0 and -O2: each lowers to its
// llvm.capstone.cap.* intrinsic on a ptr addrspace(200) and never through an
// integer (no inttoptr or ptrtoint anywhere in the module), and the two
// noreturn ones end their block with unreachable.  The two constant-argument
// builtins (tighten, ccsrrw) carry the constant as an i64 immediate.
// Measured 2026-09-04 on the branch tools.
//
// MUTATION: in t_get_tag pass `(void *)(unsigned long)c` -> an inttoptr appears
// and --implicit-check-not fires (performed 2026-09-04 at -O0).
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -O0 -emit-llvm -o - %s | FileCheck %s --implicit-check-not=inttoptr --implicit-check-not=ptrtoint
// RUN: %clang_cc1 -triple capstone64-unknown-elf -O2 -emit-llvm -o - %s | FileCheck %s --implicit-check-not=inttoptr --implicit-check-not=ptrtoint

// CHECK-LABEL: @t_get_tag(
// CHECK: {{(tail )?}}call addrspace(200) i64 @llvm.capstone.cap.get.tag.p200(ptr addrspace(200) %
unsigned long t_get_tag(void *c) { return __builtin_capstone_cap_get_tag(c); }
// CHECK-LABEL: @t_get_base(
// CHECK: {{(tail )?}}call addrspace(200) i64 @llvm.capstone.cap.get.base.p200(ptr addrspace(200) %
unsigned long t_get_base(void *c) { return __builtin_capstone_cap_get_base(c); }
// CHECK-LABEL: @t_get_end(
// CHECK: {{(tail )?}}call addrspace(200) i64 @llvm.capstone.cap.get.end.p200(ptr addrspace(200) %
unsigned long t_get_end(void *c) { return __builtin_capstone_cap_get_end(c); }
// CHECK-LABEL: @t_get_perm(
// CHECK: {{(tail )?}}call addrspace(200) i64 @llvm.capstone.cap.get.perm.p200(ptr addrspace(200) %
unsigned long t_get_perm(void *c) { return __builtin_capstone_cap_get_perm(c); }
// CHECK-LABEL: @t_get_cursor(
// CHECK: {{(tail )?}}call addrspace(200) i64 @llvm.capstone.cap.get.cursor.p200(ptr addrspace(200) %
unsigned long t_get_cursor(void *c) { return __builtin_capstone_cap_get_cursor(c); }
// CHECK-LABEL: @t_shrink(
// CHECK: {{(tail )?}}call addrspace(200) ptr addrspace(200) @llvm.capstone.cap.shrink.p200(ptr addrspace(200) %{{.*}}, i64 %{{.*}}, i64 %
void *t_shrink(void *c, unsigned long b, unsigned long e) { return __builtin_capstone_cap_shrink(c, b, e); }
// CHECK-LABEL: @t_tighten(
// CHECK: {{(tail )?}}call addrspace(200) ptr addrspace(200) @llvm.capstone.cap.tighten.p200(ptr addrspace(200) %{{.*}}, i64 7)
void *t_tighten(void *c) { return __builtin_capstone_cap_tighten(c, 7); }
// CHECK-LABEL: @t_scc(
// CHECK: {{(tail )?}}call addrspace(200) ptr addrspace(200) @llvm.capstone.cap.scc.p200(ptr addrspace(200) %{{.*}}, i64 %
void *t_scc(void *c, unsigned long v) { return __builtin_capstone_cap_scc(c, v); }
// CHECK-LABEL: @t_init(
// CHECK: {{(tail )?}}call addrspace(200) ptr addrspace(200) @llvm.capstone.cap.init.p200(ptr addrspace(200) %{{.*}}, i64 %
void *t_init(void *c, unsigned long v) { return __builtin_capstone_cap_init(c, v); }
// CHECK-LABEL: @t_seal(
// CHECK: {{(tail )?}}call addrspace(200) ptr addrspace(200) @llvm.capstone.cap.seal.p200(ptr addrspace(200) %
void *t_seal(void *c) { return __builtin_capstone_cap_seal(c); }
// CHECK-LABEL: @t_delin(
// CHECK: {{(tail )?}}call addrspace(200) ptr addrspace(200) @llvm.capstone.cap.delin.p200(ptr addrspace(200) %
void *t_delin(void *c) { return __builtin_capstone_cap_delin(c); }
// CHECK-LABEL: @t_mrev(
// CHECK: {{(tail )?}}call addrspace(200) ptr addrspace(200) @llvm.capstone.cap.mrev.p200(ptr addrspace(200) %
void *t_mrev(void *c) { return __builtin_capstone_cap_mrev(c); }
// CHECK-LABEL: @t_drop(
// CHECK: {{(tail )?}}call addrspace(200) ptr addrspace(200) @llvm.capstone.cap.drop.p200(ptr addrspace(200) %
void *t_drop(void *c) { return __builtin_capstone_cap_drop(c); }
// CHECK-LABEL: @t_revoke(
// CHECK: {{(tail )?}}call addrspace(200) ptr addrspace(200) @llvm.capstone.cap.revoke.p200(ptr addrspace(200) %
void *t_revoke(void *c) { return __builtin_capstone_cap_revoke(c); }
// CHECK-LABEL: @t_call(
// CHECK: {{(tail )?}}call addrspace(200) ptr addrspace(200) @llvm.capstone.cap.call.p200(ptr addrspace(200) %
void *t_call(void *c) { return __builtin_capstone_cap_call(c); }
// CHECK-LABEL: @t_enter(
// CHECK: {{(tail )?}}call addrspace(200) i64 @llvm.capstone.cap.enter.p200(ptr addrspace(200) %
unsigned long t_enter(void *c) { return __builtin_capstone_cap_enter(c); }
// CHECK-LABEL: @t_ccsrrw(
// CHECK: {{(tail )?}}call addrspace(200) ptr addrspace(200) @llvm.capstone.cap.ccsrrw.p200(ptr addrspace(200) %{{.*}}, i64 1)
void *t_ccsrrw(void *c) { return __builtin_capstone_cap_ccsrrw(c, 1); }
// CHECK-LABEL: @t_return(
// CHECK: {{(tail )?}}call addrspace(200) void @llvm.capstone.cap.return.p200(ptr addrspace(200) %{{.*}}, i64 %
// CHECK-NEXT: unreachable
void t_return(void *c, unsigned long code) { __builtin_capstone_cap_return(c, code); }
// CHECK-LABEL: @t_exit(
// CHECK: {{(tail )?}}call addrspace(200) void @llvm.capstone.cap.exit.p200(ptr addrspace(200) %{{.*}}, i64 %
// CHECK-NEXT: unreachable
void t_exit(void *c, unsigned long code) { __builtin_capstone_cap_exit(c, code); }
