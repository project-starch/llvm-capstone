// A 128-bit integer is an INTEGER on capstone64, not a capability.
//
// It used to be both: MVT::i128 was the machine type for a capability, and the
// backend told a genuine __int128 from a capability by heuristics that defaulted
// to capability. `a + b` came out as `cincoffset a0, a1, a0`, a cursor increment,
// with no diagnostic; `a & b` came out as `lcc a0,a0,2 / lcc a1,a1,2 / and`, so
// the high 64 bits were never in the computation at all; and `v >> 64` was a
// fatal error, "cannot lower a 128-bit right shift by >= XLen". The type was
// disabled in Sema because of it.
//
// A capability is MVT::c128 in its own register class now, so i128 is an ordinary
// illegal type and the generic legalizer expands it. The sequences below are
// instruction-for-instruction what upstream riscv64 emits for the same source.
//
// The implicit-check-nots cover the whole output: cincoffset and lcc are the two
// instructions the wrong lowering used, and no 128-bit libcall may appear either
// -- a freestanding domain has no compiler-rt.
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-feature +m -ffreestanding \
// RUN:   -O2 -mframe-pointer=none -S -o - %s \
// RUN:   | FileCheck %s --implicit-check-not=cincoffset --implicit-check-not=lcc \
// RUN:       --implicit-check-not=__addti3 --implicit-check-not=__divti3 \
// RUN:       --implicit-check-not=__multi3

// A real 128-bit add: two words with the carry made explicit by sltu.
// CHECK-LABEL: add:
// CHECK:      add a1, a3, a1
// CHECK-NEXT: add a0, a2, a0
// CHECK-NEXT: sltu a2, a0, a2
// CHECK-NEXT: add a1, a1, a2
// CHECK-NEXT: cjalr zero, 0(ra)
unsigned __int128 add(unsigned __int128 a, unsigned __int128 b) { return a + b; }

// The high half. This used to be a fatal error; it is a register move, because the
// high word is already in the second argument register.
// CHECK-LABEL: high:
// CHECK:      mv a0, a1
// CHECK-NEXT: cjalr zero, 0(ra)
unsigned long high(unsigned __int128 v) { return (unsigned long)(v >> 64); }

// Bitwise on both halves, not on a cursor.
// CHECK-LABEL: and_:
// CHECK:      and a0, a2, a0
// CHECK-NEXT: and a1, a3, a1
// CHECK-NEXT: cjalr zero, 0(ra)
unsigned __int128 and_(unsigned __int128 a, unsigned __int128 b) { return a & b; }

// _BitInt reached the same machine type and was capped at XLen for the same reason.
// CHECK-LABEL: badd:
// CHECK:      add a1, a3, a1
// CHECK-NEXT: add a0, a2, a0
// CHECK-NEXT: sltu a2, a0, a2
// CHECK-NEXT: add a1, a1, a2
unsigned _BitInt(128) badd(unsigned _BitInt(128) a, unsigned _BitInt(128) b) { return a + b; }

// The control that keeps the implicit-check-nots honest: a uintptr_t round-trip is
// NOT 128-bit integer arithmetic. It reaches the backend as ptrtoint/inttoptr on
// ptr addrspace(200) and costs one `andi`, and it must stay that way -- if this
// grew a cincoffset the checks above would be measuring the wrong thing.
// CHECK-LABEL: align_down:
// CHECK:      andi a0, a0, -32
// CHECK-NEXT: cjalr zero, 0(ra)
void *align_down(void *p) {
  return (void *)((__UINTPTR_TYPE__)p & ~(__UINTPTR_TYPE__)31);
}
