/* Minimal reproducer for the crash that blocks MPY-T14/MPY-T15.
 *
 *   clang -target capstone64-unknown-elf -ffreestanding -std=c99 -O0 -c this.c
 *   Assertion `VT.isVector() && "Unable to legalize non-vector shift"' failed
 *   in SelectionDAGLegalize::ExpandNode (llvm/lib/CodeGen/SelectionDAG/LegalizeDAG.cpp)
 *
 * Reproduces on origin/capstone-bootstrap as well, so it predates
 * capstone-codegen-cap-constants. Found while probing whether lib/oofatfs, which
 * MICROPY_VFS needs, compiles for the domain: it does not, dying in @f_mkfs.
 *
 * Root cause, same family as everything in i128-capability-fixes.md: on capstone64
 * i128 is LEGAL because it carries a capability, so a genuine 128-bit shift is
 * never expanded by the generic legaliser and never lowered by the target either.
 * The round-1 fix covered shift-by-XLen-or-more and produced a wrong result; this
 * form has no legalisation at all.
 */
typedef unsigned long long u64;

/* fine: 64-bit shifts legalise normally */
u64 shl64(u64 x, unsigned n) { return x << n; }
u64 shr64(u64 x, unsigned n) { return x >> n; }

/* crash: variable-count shift on the capability-width integer type */
unsigned __int128 shl128(unsigned __int128 x, unsigned n) { return x << n; }
unsigned __int128 shr128(unsigned __int128 x, unsigned n) { return x >> n; }
