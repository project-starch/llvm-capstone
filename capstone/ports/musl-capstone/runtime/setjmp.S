/* setjmp and longjmp for a pure-capability domain.
 *
 * musl's own live in src/setjmp/riscv64 as .S files and are not in libc-capstone.a, which
 * holds surveyed .c files only; they would also be wrong here, because they
 * save ra and sp with sd, which keeps the address and drops the tag, and the
 * restored sp then faults on its first use. This is the mruby port's
 * capstone_setjmp.c with the same register set (ra, s0-s11, sp) written as a
 * .S file rather than naked inline asm: mruby had to keep that C file in its
 * own translation unit because a flatten attribute on the caller inlined the
 * naked body without its argument. An assembly file cannot be inlined.
 *
 * Slot layout matches bits/setjmp.h in the arch overlay: fourteen 16-byte
 * slots, buffer 16-aligned. gp and tp are not saved: both are domain constants
 * that a jump within the domain cannot change.
 */
.text
.global setjmp
.global _setjmp
.global __setjmp
.global sigsetjmp
.global __sigsetjmp
.type setjmp, %function
.type sigsetjmp, %function
/* sigsetjmp saves the same registers with the same instructions, not by a
   call to setjmp: a C wrapper that CALLS setjmp records the wrapper's own
   frame, and the later jump then lands with the wrapper's sp and frame
   pointer instead of the caller's. That is what this file replaced:
   libc-test's setjmp test faulted after its first siglongjmp with a frame
   pointer that belonged to a function that had already returned. musl's own
   riscv64 sigsetjmp tail-calls setjmp for the savemask == 0 case for the same
   reason. With savemask set, the registers are saved first and a C helper
   then records the domain's signal mask into the buffer's __ss field; the
   call clobbers nothing the buffer has not already captured. */
setjmp:
_setjmp:
__setjmp:
	stc ra,   0(a0)
	stc s0,  16(a0)
	stc s1,  32(a0)
	stc s2,  48(a0)
	stc s3,  64(a0)
	stc s4,  80(a0)
	stc s5,  96(a0)
	stc s6, 112(a0)
	stc s7, 128(a0)
	stc s8, 144(a0)
	stc s9, 160(a0)
	stc s10, 176(a0)
	stc s11, 192(a0)
	stc sp, 208(a0)
	li a0, 0
	ret
.size setjmp, .-setjmp

sigsetjmp:
__sigsetjmp:
	stc ra,   0(a0)
	stc s0,  16(a0)
	stc s1,  32(a0)
	stc s2,  48(a0)
	stc s3,  64(a0)
	stc s4,  80(a0)
	stc s5,  96(a0)
	stc s6, 112(a0)
	stc s7, 128(a0)
	stc s8, 144(a0)
	stc s9, 160(a0)
	stc s10, 176(a0)
	stc s11, 192(a0)
	stc sp, 208(a0)
	sd a1, 224(a0)
	beqz a1, 1f
	cincoffsetimm sp, sp, -32
	stc ra, 16(sp)
2:
	auipc t0, %pcrel_hi(__capstone_sigsetjmp_save)
	addi t0, t0, %pcrel_lo(2b)
	cincoffset t0, gp, t0
	cjalr ra, 0(t0)
	ldc ra, 16(sp)
	cincoffsetimm sp, sp, 32
1:
	li a0, 0
	ret
.size sigsetjmp, .-sigsetjmp

.global longjmp
.global _longjmp
.type longjmp, %function
longjmp:
_longjmp:
	/* A jump out of a signal handler must acknowledge the abandoned event
	   using the destination sp, before restoring the saved register image. */
	cincoffsetimm sp, sp, -32
	stc a0, 0(sp)
	sd a1, 16(sp)
2:
	auipc t0, %pcrel_hi(__capstone_longjmp_reap)
	addi t0, t0, %pcrel_lo(2b)
	cincoffset t0, gp, t0
	cjalr ra, 0(t0)
	ldc a0, 0(sp)
	ld a1, 16(sp)
	cincoffsetimm sp, sp, 32
	ldc ra,   0(a0)
	ldc s0,  16(a0)
	ldc s1,  32(a0)
	ldc s2,  48(a0)
	ldc s3,  64(a0)
	ldc s4,  80(a0)
	ldc s5,  96(a0)
	ldc s6, 112(a0)
	ldc s7, 128(a0)
	ldc s8, 144(a0)
	ldc s9, 160(a0)
	ldc s10, 176(a0)
	ldc s11, 192(a0)
	ldc sp, 208(a0)
	/* a0 was the buffer until the last load; the value goes in afterwards.
	   A zero argument comes back as 1, per the contract. */
	mv a0, a1
	bnez a0, 1f
	li a0, 1
1:
	ret
.size longjmp, .-longjmp
