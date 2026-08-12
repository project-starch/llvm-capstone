#ifndef TRAPCTL_KERNEL_H
#define TRAPCTL_KERNEL_H
/* POSITIVE CONTROL for the in-domain trap handler, and for the LCC selectors that the
 * SQLite OUT_OF_BOUNDS probe is built on.
 *
 * WHY IT EXISTS. The SQLite domain now runs past sqlite3FinishCoding (S-06 fixed) and meets a
 * second fault: mcause 29, OUT_OF_BOUNDS, at vdbeMemClearExternAndSetNull+0x3c on
 * `ldc a1, 0x0(a0)`. To tell the two surviving readings apart -- the address is right and the
 * BOUNDS are wrong, versus the address is WILD -- we need the offending capability's
 * cursor/start/end. A capability fault inside a domain WEDGES (mtvec is 0 unless the glue
 * installs a vector), and a wedged core reports nothing, so the numbers have to be written to
 * the shared region BEFORE the fault and the domain has to survive long enough to return.
 *
 * INTERP_DOMAIN_MTVEC in start-gp-captable-interp.S is what makes that possible: it installs
 * .Ldomain_trap before cap-init, and the handler converts a fault into the ordinary domreturn.
 * On the only occasion it was measured (2026-08-05) the install was sited too late and the
 * handler PROVABLY DID NOT RUN. So it is unproven on this bitstream, and a SQLite run built on
 * it would be uninterpretable: no return would mean either "the trap handler does not work" or
 * "SQLite wedged somewhere else", and those are not the same finding.
 *
 * This rung separates them, in one ~2 KB domain, before the expensive one runs.
 *
 * VERDICT, read from res[0]. Every value is written BEFORE the step it labels, so the last one
 * to survive says exactly how far the domain got:
 *
 *   0x7A05  THE PASS. The deliberate out-of-bounds ldc faulted AND the handler returned.
 *           The instrument works; a SQLite build with the same glue can be trusted.
 *   0x7A06  the deliberate ldc did NOT fault -- the fault construction is wrong, so this rung
 *           says nothing about the handler. Instrument invalid, not a hardware finding.
 *   0x7A0E  LCC's type query called a real capability plain (answered 7). Selector 1 is broken
 *           or the bitstream predates the S-06 enabler.
 *   0x7A0D  the type query worked but cursor/start/end are mutually inconsistent: a freshly
 *           taken capability's own cursor is outside its own bounds. Selectors 2/3/4 are not
 *           returning what this code assumes, which would silently corrupt the SQLite probe.
 *   0x7A00..0x7A04  wedged earlier than the deliberate fault -- see the table in the body.
 *   NO RETURN AT ALL  the handler does not fire on this bitstream. Any SQLite probe built on
 *           it is VOID; do not read one.
 *
 * res[3..7] carry cursor, start, end, type and the offset used, so a pass also proves the four
 * queries the SQLite probe issues return sane values rather than zeros.
 *
 * The distinction between 0x7A05 and 0x7A06 is the whole point: a gate that cannot fire is not
 * a passing gate. 0x7A06 means this control never created the condition it exists to detect.
 */

/* 16-byte aligned static, not a local: a 16-byte-aligned local forces dynamic stack
 * realignment, which this backend cannot legalize. Same reason as s06lcc_kernel.h. */
__attribute__((aligned(16))) static unsigned char trapctl_buf[64];

/* The selector rides the rs2 ENCODING field, so it must be a register NAME whose number is the
 * selector -- x1/x2/x3/x4, never a literal. */
#define TRAPCTL_LCC(out, cap, sel) \
  __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, " sel : "=r"(out) : "r"(cap))
#define TRAPCTL_LDC(out, cap) \
  __asm__ volatile(".insn i 0x5b, 0x3, %0, 0(%1)" : "=r"(out) : "r"(cap))

static void trapctl_run(volatile unsigned long *res)
{
  void *c;
  unsigned long ty = 0, cur = 0, st = 0, en = 0, delta;

  res[1] = 0UL;
  res[2] = 0xD09EUL;
  res[0] = 0x7A00UL;               /* domain entered, glue completed */

  c = (void *)trapctl_buf;

  /* --- selector 1: must NOT call a real capability plain --------------------------- */
  TRAPCTL_LCC(ty, c, "x1");
  res[0] = 0x7A01UL;
  if (ty == 7UL) { res[0] = 0x7A0EUL; return; }

  /* --- selectors 2/3/4: the three the SQLite probe depends on ---------------------- */
  TRAPCTL_LCC(cur, c, "x2");
  TRAPCTL_LCC(st,  c, "x3");
  TRAPCTL_LCC(en,  c, "x4");
  res[3] = cur; res[4] = st; res[5] = en; res[6] = ty;
  res[0] = 0x7A02UL;

  /* A capability just taken from a global must contain its own cursor. If it does not, the
     selectors are not what this code thinks they are, and reading SQLite's numbers through them
     would produce a confident wrong answer. Checked here rather than assumed. */
  if (!(cur >= st && cur <= en - 16UL)) { res[0] = 0x7A0DUL; return; }
  res[0] = 0x7A03UL;

  /* --- a LEGAL ldc through the same capability, so the fault below is attributable to
         the offset and to nothing else ---------------------------------------------- */
  { void *v; TRAPCTL_LDC(v, c); res[9] = (unsigned long)v; }
  res[0] = 0x7A04UL;

  /* --- THE DELIBERATE FAULT -------------------------------------------------------
     Bounds rule for LDC (capstone_dyn_unit.anvil, func LDC): fault iff
     cursor+imm < start || cursor+imm > end-16. Pushing the cursor to end+16 satisfies the
     second disjunct by construction, and the offset is COMPUTED from the capability's own
     end rather than being a guessed constant -- a fixed immediate would silently stay in
     bounds if the global's capability turned out to cover more than the object. */
  delta = (en - cur) + 16UL;
  res[7] = delta;
  res[0] = 0x7A05UL;               /* the last store before the fault */
  {
    void *bad = (void *)((unsigned char *)c + delta);
    void *v;
    TRAPCTL_LDC(v, bad);
    res[8] = (unsigned long)v;
  }
  res[0] = 0x7A06UL;               /* reached only if the ldc did NOT fault */
}
#endif /* TRAPCTL_KERNEL_H */
