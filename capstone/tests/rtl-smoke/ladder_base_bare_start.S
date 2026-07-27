/* S-mode entry for the bare-metal baseline payload.
 * OpenSBI enters with a0=hartid, a1=fdt pointer. We need only a stack.
 * No trap vector is installed and no interrupt is enabled: nothing may preempt a
 * measurement, and if a trap ever fires we want a hang (a visible failure) rather
 * than a silently perturbed cycle count. */
  .section .text.entry, "ax"
  .globl _start
_start:
  la   sp, __stack_top
  /* Zero .bss before entering C. The rung kernels rely on zero-initialised
     statics (BEEBS arrays, counters); without this they start from whatever DRAM
     held, which would corrupt results silently rather than visibly. */
  la   t0, __bss_start
  la   t1, __bss_end
1:
  bgeu t0, t1, 2f
  sd   zero, 0(t0)
  addi t0, t0, 8
  j    1b
2:
  j    bare_main
