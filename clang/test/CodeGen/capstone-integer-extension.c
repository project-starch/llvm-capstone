// capstone64 passes and returns an integer narrower than XLEN extended to 64
// bits, by the RISC-V psABI rule: a 32-bit integer sign-extended whether it is
// signed or not, a narrower one by its own signedness. Aggregates keep the
// default ABI (capstone-abi-cap-struct.c). Without the rule an int result's
// upper half was unspecified and the caller of a qsort comparator
// sign-extended it again after every call (505.mcf_r's spec_qsort).
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=IR
// RUN: %clang_cc1 -triple capstone64-unknown-elf -O2 -S -o - %s | FileCheck %s --check-prefix=ASM

// IR-LABEL: define dso_local noundef signext i32 @ret_int(i32 noundef returned signext %x)
int ret_int(int x) { return x; }

// IR-LABEL: define dso_local noundef signext i32 @ret_uint(i32 noundef returned signext %x)
unsigned ret_uint(unsigned x) { return x; }

// IR-LABEL: define dso_local noundef signext i16 @ret_short(i16 noundef returned signext %x)
short ret_short(short x) { return x; }

// IR-LABEL: define dso_local noundef zeroext i8 @ret_uchar(i8 noundef returned zeroext %x)
unsigned char ret_uchar(unsigned char x) { return x; }

// IR-LABEL: define dso_local noundef zeroext i1 @ret_bool(i1 noundef returned zeroext %x)
_Bool ret_bool(_Bool x) { return x; }

enum sign { NEG = -1, POS = 1 };
// IR-LABEL: define dso_local noundef signext i32 @ret_enum(i32 noundef returned signext %x)
enum sign ret_enum(enum sign x) { return x; }

// A full-width integer and a capability are passed as they are.
// IR-LABEL: define dso_local noundef i64 @ret_long(i64 noundef returned %x)
long ret_long(long x) { return x; }
// IR-LABEL: define dso_local noundef ptr addrspace(200) @ret_ptr(ptr addrspace(200) noundef readnone returned {{.*}}%p)
void *ret_ptr(void *p) { return p; }

// The callee extends what it returns.
// IR-LABEL: define dso_local noundef signext i32 @narrow(i64 noundef %x)
// ASM-LABEL: narrow:
// ASM: sext.w a0, a0
// ASM: {{ret|cjalr zero, 0\(ra\)}}
int narrow(long x) { return (int)x; }

// The caller relies on it: two comparator results summed as 64-bit values
// need no sext.w.
// IR-LABEL: define dso_local {{.*}}i64 @sum(
// IR: call signext addrspace(200) i32 %cmp(
// IR: call signext addrspace(200) i32 %cmp(
// ASM-LABEL: sum:
// ASM-NOT: sext.w
// ASM: {{ret|cjalr zero, 0\(ra\)}}
long sum(int (*cmp)(const void *, const void *), const void *a, const void *b) {
  long s = cmp(a, b);
  s += cmp(b, a);
  return s;
}
