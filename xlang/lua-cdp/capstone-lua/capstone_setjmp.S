/* Capability-aware setjmp/longjmp for a reference-Lua Capstone domain.
 *
 * ABI: the gp-captable / gp-free ABI (-capstone-gp-captable, entry
 * start-gp-captable-interp.S), where `ra` is a SCALAR (sd/ld, return via a plain
 * `ret` = jalr x0,0(x1)) while `sp` and every callee-saved register s0-s11 remain
 * 128-bit CAPABILITIES (stc/ldc at 16-byte offsets). Verified against a compiled
 * frame under -capstone-gp-captable:
 *   sd  ra,0x60(sp); stc s0,0x50(sp); ... ; ld ra,0x60(sp); ldc s0,0x50(sp); ret.
 * (The earlier cjalr-ABI form saved ra with stc and returned via `cjalr zero,0(ra)`;
 * that is wrong here -- ra is not a capability under gp-captable.)
 *
 * jmp_buf layout (224 B, 16-byte aligned -- stc drops the tag on a non-16-aligned
 * address; ra uses only the low 8 B of its 16 B slot):
 *   +0 ra  +16 sp  +32 s0  +48 s1  +64 s2  +80 s3  +96 s4  +112 s5
 *   +128 s6  +144 s7  +160 s8  +176 s9  +192 s10  +208 s11
 *
 * Lua needs this for LUAI_THROW/LUAI_TRY (ldo.c). On the CDP happy path longjmp
 * is never taken (the stale access faults in hardware, not via a Lua error), but
 * setjmp must still set up the protected-call frame correctly.
 */
	.text

	.global setjmp
	.type setjmp, @function
setjmp:                         /* a0 = jmp_buf capability */
	sd    ra,  0(a0)            /* ra is a SCALAR under gp-captable */
	stc   sp,  16(a0)
	stc   s0,  32(a0)
	stc   s1,  48(a0)
	stc   s2,  64(a0)
	stc   s3,  80(a0)
	stc   s4,  96(a0)
	stc   s5,  112(a0)
	stc   s6,  128(a0)
	stc   s7,  144(a0)
	stc   s8,  160(a0)
	stc   s9,  176(a0)
	stc   s10, 192(a0)
	stc   s11, 208(a0)
	li    a0, 0                 /* direct call returns 0 */
	ret                         /* scalar-ra return (jalr x0,0(x1)) */
	.size setjmp, .-setjmp

	.global longjmp
	.type longjmp, @function
longjmp:                        /* a0 = jmp_buf, a1 = val */
	ld    ra,  0(a0)            /* ra is scalar; a0 (jmp_buf) stays the load base */
	ldc   sp,  16(a0)
	ldc   s0,  32(a0)
	ldc   s1,  48(a0)
	ldc   s2,  64(a0)
	ldc   s3,  80(a0)
	ldc   s4,  96(a0)
	ldc   s5,  112(a0)
	ldc   s6,  128(a0)
	ldc   s7,  144(a0)
	ldc   s8,  160(a0)
	ldc   s9,  176(a0)
	ldc   s10, 192(a0)
	ldc   s11, 208(a0)
	mv    a0, a1                /* setjmp appears to return val ... */
	bnez  a1, 1f
	li    a0, 1                 /* ... but longjmp(_,0) must return 1 */
1:
	ret                         /* scalar-ra return: resume just after setjmp */
	.size longjmp, .-longjmp
