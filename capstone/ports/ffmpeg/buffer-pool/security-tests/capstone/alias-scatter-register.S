/* Leaf fixture: a0 holds the old alias throughout. No call, spill or reload
 * separates revocation from the labelled accesses. Arguments: old alias,
 * parent handle slot, reuse, write, mode, sibling, ready marker, old address.
 * Reuse accesses reclaimed parent authority at the child's old address;
 * memory-holder cases also split and delinearize a fresh child grant. */
.text
.option norvc
.align 2
.globl ff2_alias_scatter_register_span
.type ff2_alias_scatter_register_span, @function
ff2_alias_scatter_register_span:
  lbu t1, 0(a0)
  li t2, 17
  bne t1, t2, .Lbad
  beqz a4, .Lbaseline
  .insn i 0x5b, 0x3, t0, 0(a1)
  .insn r 0x5b, 0x1, 0x00, x0, t0, x0
  beqz a2, .Lsave
  .insn r 0x5b, 0x1, 0x04, t1, t0, x1
  li t2, 3
  bne t1, t2, .Lfresh
  .insn r 0x5b, 0x1, 0x04, t1, t0, x2
  .insn r 0x5b, 0x1, 0x04, t2, t0, x4
.Lzero:
  addi t3, t1, 16
  bgtu t3, t2, .Linit
  .insn s 0x5b, 0x4, x0, 0(t0)
  mv t1, t3
  j .Lzero
.Linit:
  .insn r 0x5b, 0x1, 0x09, t0, t0, x0
.Lfresh:
  .insn r 0x5b, 0x1, 0x04, t1, t0, x2
  bne t1, a7, .Lbad
  li t2, 61
  sb t2, 0(t0)
  lbu t1, 0(t0)
  bne t1, t2, .Lbad
.Lsave:
  .insn s 0x5b, 0x4, t0, 0(a1)
  j .Lsibling
.Lbaseline:
  beqz a2, .Lsibling
  li t2, 61
  sb t2, 0(a0)
.Lsibling:
  lbu t1, 64(a5)
  li t2, 41
  bne t1, t2, .Lbad
  li t2, 42
  sb t2, 65(a5)
  lbu t1, 65(a5)
  bne t1, t2, .Lbad
  .insn r 0x5b, 0x1, 0x43, x0, a6, x0
  bnez a3, .Lwrite
.globl ff2_probe_register_read
ff2_probe_register_read:
  lbu t1, 0(a0)
  li t2, 17
  beqz a2, .Lcheck
  li t2, 61
.Lcheck:
  bne t1, t2, .Lbad
  li a0, 0
  ret
.Lwrite:
  li t2, 93
.globl ff2_probe_register_write
ff2_probe_register_write:
  sb t2, 0(a0)
  lbu t1, 0(a0)
  bne t1, t2, .Lbad
  li a0, 0
  ret
.Lbad:
  li a0, 1
  ret
.size ff2_alias_scatter_register_span, .-ff2_alias_scatter_register_span
