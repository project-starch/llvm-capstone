#ifndef TAGF_H
#define TAGF_H
/* MINIMAL REPRO for the RTL finding of 2026-08-04: capability metadata is routed into the
 * dcache write-user sideband by EVERY store, not just capability stores.
 *
 *   core/load_store_unit.sv:1003-1020  builds lsu_ctrl.user from cap_data.cap_metadata_b
 *                                      unconditionally for every LOAD/STORE-fu instruction
 *   core/store_unit.sv:344-346         st_user_n = lsu_ctrl.user     (ungated)
 *   core/store_buffer.sv:172-176       req_port_o.data_wuser = ...user  (ungated)
 *   core/cache_subsystem/wt_dcache.sv:70-79  the write buffer tracks `data` per BYTE
 *                                      (dirty/valid bitmasks) but `user` as ONE flat field
 *                                      with no per-byte mask
 *
 * Software-visible consequence to test: store a real capability to a 16-byte slot, then
 * overwrite part of that slot with a PLAIN INTEGER store, then load it back as a capability.
 * A plain integer store over a capability MUST invalidate it. If the reloaded value still
 * reports a live capability type, the integer store did not clear the metadata -- the `user`
 * sideband carried a stale/foreign capability metadata word into that location.
 *
 * ARCHITECTURAL REFERENCE, measured: QEMU aborts in helper_cslcc on `rs1_v->tag` -- i.e. the
 * integer store DOES clear the capability's tag, so the second lcc is an error. Correct
 * silicon behaviour is therefore a TRAP (no result). A RETURNED VALUE means the tag
 * survived an integer overwrite, which is the defect. Run `tagr` first: it is identical up
 * to the overwrite but skips the second lcc, so it separates "lcc trapped" from "the
 * domain wedged earlier".
 *
 * Return: 1000 + (type_after_capability_store * 10) + type_after_integer_overwrite
 * so one number carries both readings and cannot be confused with any other rung. */
#define LCC(rd, rs, f) __asm__ volatile(".insn r 0x5b,0x1,0x4, %0, %1, x" #f : "=r"(rd) : "r"(rs))
/* Image padding. Built WITHOUT the ladder instrumentation (mode 0) this domain's loadable
 * image comes to exactly 0x1000 bytes, which makes the monitor's create_domain SPLIT
 * degenerate -- QEMU asserts in helper_cssplit (`mid > base && mid < end`). Padding the
 * image past 0x1000 keeps that split non-degenerate. It is dead data; nothing reads it. */
static const volatile unsigned long tagf_pad[512] = { 1 };
static char tagf_g[16] = { 1 };
static unsigned tagf_compute(void)
{
  void *cap;
  void *buf[2];                       /* 16-byte, capability-aligned stack slot */
  void *back;
  unsigned long t_cap = 0, t_int = 0;

  __asm__ volatile(".insn i 0x5b, 0x3, %0, 0(gp)" : "=r"(cap));   /* a real capability */

  buf[0] = cap;                       /* stc: the slot now holds a live capability */
  __asm__ volatile("" ::: "memory");
  back = buf[0];                      /* ldc it straight back */
  LCC(t_cap, back, 1);                /* field 1 = cap type, BEFORE the integer overwrite */

  /* Plain 64-bit INTEGER store over the low half of the same 16-byte capability slot.
     This must destroy the capability. */
  *(volatile unsigned long *)(void *)&buf[0] = 0x0123456789abcdefUL;
  __asm__ volatile("" ::: "memory");

  back = buf[0];                      /* load it back as a capability again */
  LCC(t_int, back, 1);                /* field 1 = cap type, AFTER the integer overwrite */

  (void)tagf_g; (void)tagf_pad[0];
  return (unsigned)(1000u + ((unsigned)(t_cap & 7u) * 10u) + (unsigned)(t_int & 7u));
}
#endif
