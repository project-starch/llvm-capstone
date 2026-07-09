#ifndef CAPSTONE_TESTS_RUNTIME_QEMU_INTRA_DOMAIN_MREV_REVOKE_PROBE_DOMAIN_H
#define CAPSTONE_TESTS_RUNTIME_QEMU_INTRA_DOMAIN_MREV_REVOKE_PROBE_DOMAIN_H

/* Domain-side glue for the held-cap probe. Compiled by the Capstone clang via
 * runtime-qemu/build-domain.sh, so the capability builtins are available.
 *
 * THE RECEIVE PROTOCOL (this is the bit the plan asked us to find):
 *
 *   The monitor's shared_region_annotated() ends with
 *       d = __domcallsaves(d, CAPSTONE_DPI_REGION_SHARE, r);
 *   which ENTERS the domain with the linear region capability `r` in the
 *   capability-argument register. capstone/my_first_domain/start.S already
 *   saves that register (`stc a1, sp, 80`) and reloads it as domain_main's
 *   FIRST argument (`ldc a0, sp, 80`); the DPI function code arrives as the
 *   second, scalar argument (`sd a0, 64(sp)` / `ld a1, 64(sp)`).
 *
 *   So there is NO entry-stub glue to write and no lcc index to guess: the
 *   delivered capability *is* domain_main's `arg` on the entry whose `func` is
 *   PROBE_DPI_REGION_SHARE. Existing runtime probes discard it only because
 *   they are .smode payloads under the sbi.dom scaffold, where the cap lands in
 *   the scaffold's regions[] and S-mode reaches the bytes through ambient cpmp
 *   authority instead. A domain_main-style .dom needs none of that.
 *
 * PARKING THE CAPABILITY ACROSS ENTRIES:
 *
 *   REGION_SHARE and CALL are two separate domain invocations, so the arena has
 *   to survive in memory between them. A .bss capability slot is the right home:
 *   stc/ldc duplicate rather than move (the memory path enforces no linearity),
 *   and cap_compress/cap_uncompress round-trip the type field, so the parked
 *   capability comes back LIN -- which helper_csmrev asserts on.
 */

#include "intra_domain_mrev_revoke_probe.h"

/* The monitor-granted arena, parked between the REGION_SHARE and CALL entries.
 * 16-byte aligned by the i128 pointer ABI, as store_capregval requires. */
static void *probe_arena;

/* Handle the REGION_SHARE entry. Returns 1 when this entry was the delivery and
 * domain_main has nothing else to do. */
static inline int probe_receive(void *arg, unsigned func) {
  if (func == PROBE_DPI_REGION_SHARE) {
    probe_arena = arg; /* stc: the delivered capability, tag intact */
    return 1;
  }
  return 0;
}

/* cssplit rd, rs1, rs2 -- split the capability in rs1 at address `mid`. rs1
 * keeps [base, mid); rd receives [mid, end) with a FRESH revocation-tree node
 * at the same depth, so the two halves are independently revocable. There is no
 * builtin and no assembler mnemonic, hence the raw R-type encoding
 * (Func7=0b0000110, Func3=001, opcode=0b1011011). The emulator makes the op a
 * no-op unless rd != rs1, hence the early-clobber on the result.
 *
 * SHRINK/SHRINKTO are NOT a substitute: they copy rev_node_id unchanged, so a
 * shrunk sub-capability shares the arena's node. See task-005, finding Q3.
 */
static inline void *probe_cap_split(void **lo, unsigned long mid) {
  void *hi;
  void *l = *lo;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x06, %0, %1, %2"
                   : "=&r"(hi), "+r"(l)
                   : "r"(mid));
  *lo = l;
  return hi;
}

#endif
