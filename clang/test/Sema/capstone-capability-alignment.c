// -Wcapstone-capability-alignment: a capability occupies one 16-byte tagged
// granule. Stored at an address that is not 16-aligned it faults; copied
// through memory that is not, it loses its tag, and the pointer read back
// faults. Two shapes, both found the hard way porting CPython, one QEMU boot
// each:
//
//  - memcpy/memmove of a value that holds a capability to or from memory whose
//    type promises less (CPython's bytecode inline cache: a PyObject * copied
//    into 16-bit code units). On by default. A byte buffer's type promises
//    nothing, so copying through char * is the same diagnostic's opt-in half.
//  - a cast to a pointer to a type that holds a capability, from a pointer
//    presumed less than 16-aligned (dtoa carving Bigints out of a double array;
//    dict entries placed after a 1-byte index table). Off by default: an
//    allocator that aligns at run time is invisible to the front end, so this
//    is a survey tool, not a gate.
//
// MUTATION: make `cache` in @write_obj a pointer to a typedef of unsigned short
// with __attribute__((aligned(16))) -> its expected diagnostic is no longer
// produced (the presumed alignment is the pointee's) and -verify fails
// (performed 2026-09-24).
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -fsyntax-only -verify=default %s
// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -fsyntax-only -Wcapstone-capability-alignment -verify=default,all %s
// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -fsyntax-only -Wno-capstone-capability-alignment -verify=off %s
// RUN: %clang_cc1 -triple riscv64-unknown-elf -ffreestanding -fsyntax-only -Wcapstone-capability-alignment -verify=off %s
// off-no-diagnostics

typedef __SIZE_TYPE__ size_t;
void *memcpy(void *, const void *, size_t);
void *memmove(void *, const void *, size_t);

typedef struct object { long refcnt; struct object *type; } PyObject;

// The bytecode cache, both directions.
void write_obj(unsigned short *cache, PyObject *val) {
  memcpy(cache, &val, sizeof(val)); // default-warning {{copying 'PyObject *' (aka 'struct object *'), which holds a capability, to memory that is only 2-byte aligned}}
}
PyObject *read_obj(unsigned short *cache) {
  PyObject *val;
  memcpy(&val, cache, sizeof(val)); // default-warning {{copying to 'PyObject *' (aka 'struct object *'), which holds a capability, from memory that is only 2-byte aligned}}
  return val;
}
// A struct holding a pointer, through a narrower integer array.
void save(short *buf, PyObject *o) {
  memmove(buf, o, sizeof(*o)); // default-warning {{copying 'PyObject' (aka 'struct object'), which holds a capability, to memory that is only 2-byte aligned}}
}
// Through a byte buffer: its type says nothing about alignment (opt-in).
void save_bytes(char *buf, PyObject *o) {
  memmove(buf, o, sizeof(*o)); // all-warning {{copying 'PyObject' (aka 'struct object'), which holds a capability, to a byte buffer: the capability's tag survives on Capstone only if the buffer is 16-byte aligned at run time}}
}

// Silent: a buffer declared capability-aligned, two capability-holding types, a
// void * side (says nothing), a copy shorter than one capability, no pointers.
_Alignas(16) char aligned_buf[32];
void fine(PyObject *a, PyObject *b, void *v, short *shorts, long *l) {
  memcpy(aligned_buf, &a, sizeof(a));
  memcpy(a, b, sizeof(*a));
  memcpy(v, &a, sizeof(a));
  memcpy(shorts, &a, 8);
  memcpy(shorts, l, sizeof(*l));
}

// Casts (off by default).
typedef struct Bigint { struct Bigint *next; int k, wds; } Bigint;
Bigint *carve(double *pmem_next) {
  return (Bigint *)pmem_next; // all-warning {{cast from 'double *' to 'Bigint *' (aka 'struct Bigint *') puts a capability where memory is only 8-byte aligned; a capability must be stored 16-byte aligned}}
}
typedef struct { long hash; PyObject *key; PyObject *value; } Entry;
Entry *entries(signed char *indices, long n) {
  return (Entry *)&indices[n]; // all-warning {{cast from 'signed char *' to 'Entry *' puts a capability where memory is only 1-byte aligned}}
}
// Silent: the target holds no capability, or the source is known to be aligned.
typedef struct { long a, b; } Plain;
_Alignas(16) char pool[256];
Plain *plain(char *p) { return (Plain *)p; }
Entry *from_aligned(void) { return (Entry *)pool; }
Entry *from_void(void *p) { return (Entry *)p; }
