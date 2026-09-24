// C-54: an atomic that only moves or compares a POINTER keeps it a pointer.
// A capability is 16 bytes and more than its bits; cast to an i128 and passed
// to the sized __atomic_*_16 calls in two integer registers it came back
// untagged. Now the IR keeps ptr addrspace(200), and the backend (with
// AtomicExpand's non-integral rule) emits the generic calls, which pass every
// value through memory. Arithmetic on a pointer, and every integer atomic,
// are unchanged: the controls below.
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-feature +a -target-feature +m -O1 -emit-llvm -o - %s 2>/dev/null | FileCheck %s --check-prefix=IR
// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-feature +a -target-feature +m -O0 -S -o - %s 2>/dev/null | FileCheck %s --check-prefix=ASM
// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-feature +a -target-feature +m -O2 -S -o - %s 2>/dev/null | FileCheck %s --check-prefix=ASM

// IR-LABEL: @ld(
// IR: load atomic ptr addrspace(200), ptr addrspace(200) %{{.*}} seq_cst
// ASM-LABEL: ld:
// ASM-NOT: __atomic_load_16
// ASM: %pcrel_hi(__atomic_load)
void *ld(void **p) { return __atomic_load_n(p, __ATOMIC_SEQ_CST); }

// IR-LABEL: @st(
// IR: store atomic ptr addrspace(200) %{{.*}}, ptr addrspace(200) %{{.*}} seq_cst
// ASM-LABEL: st:
// ASM-NOT: __atomic_store_16
// ASM: %pcrel_hi(__atomic_store)
void st(void **p, void *v) { __atomic_store_n(p, v, __ATOMIC_SEQ_CST); }

// IR-LABEL: @xchg(
// IR: atomicrmw xchg ptr addrspace(200) %{{.*}}, ptr addrspace(200) %{{.*}} seq_cst
// ASM-LABEL: xchg:
// ASM-NOT: __atomic_exchange_16
// ASM: %pcrel_hi(__atomic_exchange)
void *xchg(void **p, void *v) { return __atomic_exchange_n(p, v, __ATOMIC_SEQ_CST); }

// IR-LABEL: @cas(
// IR: cmpxchg ptr addrspace(200) %{{.*}}, ptr addrspace(200) %{{.*}}, ptr addrspace(200) %{{.*}} seq_cst seq_cst
// ASM-LABEL: cas:
// ASM-NOT: __atomic_compare_exchange_16
// ASM: %pcrel_hi(__atomic_compare_exchange)
int cas(void **p, void **e, void *d) {
  return __atomic_compare_exchange_n(p, e, d, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

// Control: C11 _Atomic already kept pointers (AtomicInfo); it still does.
// IR-LABEL: @c11_ld(
// IR: load atomic ptr addrspace(200)
void *c11_ld(void *_Atomic *p) { return __c11_atomic_load(p, __ATOMIC_SEQ_CST); }

// Control: arithmetic on a pointer stays on the integer type.
// IR-LABEL: @fetch_add_ptr(
// IR: atomicrmw add ptr addrspace(200) %{{.*}}, i128
char *fetch_add_ptr(char **p) { return __atomic_fetch_add(p, 1, __ATOMIC_SEQ_CST); }

// Control: an integer atomic is untouched and lock-free.
// IR-LABEL: @ld_long(
// IR: load atomic i64
// ASM-LABEL: ld_long:
// ASM-NOT: __atomic
// ASM: .Lfunc_end
long ld_long(long *p) { return __atomic_load_n(p, __ATOMIC_SEQ_CST); }
