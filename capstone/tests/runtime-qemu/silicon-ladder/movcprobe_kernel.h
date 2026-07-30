#ifndef MOVCPROBE_KERNEL_H
#define MOVCPROBE_KERNEL_H
/* DIRECT PROOF of C-14, three instructions, no inference.
 *
 * RTL capstone_flu_unit.anvil:13-21 nulls MOVC's SOURCE register whenever the source's
 * cap_type is not NONLIN -- which includes a plain scalar (NOT_CAP). QEMU
 * (op_helper.c:580-584) guards the same zeroing with `rs1_v->tag &&`, so a scalar
 * survives there. Everything about C-14 so far is inferred from reading those two
 * sources plus one numeric match; nothing has directly observed the source register
 * being clobbered on hardware.
 *
 * This does exactly that and nothing else:
 *     li   src, 42
 *     movc dst, src        ; dst := src, and on silicon src := 0
 *     mv   out, src        ; read the SOURCE back
 *
 * A single scalar global is present only to satisfy the build gate (a gp-captable rung
 * must contain at least one `ldc gp[i]`); it is read once and never written, which is the
 * gpsz-shaped access already proven safe on silicon.
 *
 *   returns 49 (42+7) -> the source survived; C-14's mechanism is WRONG and the numeric match
 *                 was a coincidence I have to explain.
 *   returns  7 ( 0+7) -> the source was destroyed. Direct, one-boot proof, and it also confirms
 *                 the flashed bitstream really does contain this RTL behaviour (the
 *                 .anvil -> SystemVerilog -> Vivado -> bitstream chain has never been
 *                 checked against the running board).
 *
 * QEMU must return 49; that is the gate before this ever reaches the board.
 */
/* external linkage: a `static` const-initialised global gets constant-folded in the
   FPGA variant (ldc-gp=0 fails the build gate), external cannot be. */
unsigned movcprobe_keep = 7;
static unsigned movcprobe_compute(void) {
  unsigned long src, dst, out;
  __asm__ volatile(
      "li   %0, 42\n\t"
      ".insn r 0x5b, 0x1, 0xa, %1, %0, x0\n\t"   /* movc %1, %0 */
      "mv   %2, %0\n\t"
      : "=&r"(src), "=&r"(dst), "=&r"(out)
      :
      : /* no clobbers */);
  (void)dst;
  return (unsigned)out + movcprobe_keep;
}
#endif
