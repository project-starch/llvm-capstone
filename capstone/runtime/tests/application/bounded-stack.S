/* Run an int(void) callback with exactly 32 KiB below its entry SP. Merely
 * setting STACK_BYTES does not enforce that bound: the loader can give the
 * stack a larger backing capability. Save the original capability so normal
 * execution resumes with its original bounds after the callback returns. */
.text
.global capstone_test_with_32k_stack
.type capstone_test_with_32k_stack, @function
capstone_test_with_32k_stack:
  .insn i 0x5b, 0x2, sp, -32(sp)
  .insn s 0x5b, 0x4, ra, 16(sp)
  .insn s 0x5b, 0x4, sp, 0(sp)
  mv t0, sp
  li t1, 32768
  sub t0, t0, t1
  .insn r 0x5b, 0x1, 0x4, t1, sp, x4 /* LCC end */
  .insn r 0x5b, 0x1, 0x1, sp, t0, t1 /* SHRINK */
  .insn i 0x5b, 0x5, ra, 0(a0)       /* capability call */
  .insn i 0x5b, 0x3, t0, 0(sp)
  .insn i 0x5b, 0x3, ra, 16(sp)
  .insn r 0x5b, 0x1, 0xa, sp, t0, x0 /* MOVC */
  .insn i 0x5b, 0x2, sp, 32(sp)
  .insn i 0x5b, 0x5, x0, 0(ra)
.size capstone_test_with_32k_stack, .-capstone_test_with_32k_stack
