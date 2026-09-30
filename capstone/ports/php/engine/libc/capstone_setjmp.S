/* Capability-aware setjmp/longjmp for the PHP engine domain.
 *
 * REQUIRED, not optional: zend_bailout (Zend/zend.c:753) is the engine's ONLY fatal path,
 * reached from zend_hash.c:67, zend_opcode.c:269, zend_execute.c:4083,
 * zend_language_scanner.c:3132,3146. It must link AND be correct, because a bug corpus is
 * exactly the workload that makes it fire.
 *
 * ABI: the DEFAULT capstone64 ABI, NOT -capstone-gp-captable. Determined empirically from
 * a compiled non-leaf frame rather than assumed:
 *
 *     stc  ra, 0x20(sp)          <- ra is a CAPABILITY, saved with stc
 *     stc  s0, 0x10(sp)
 *     ...
 *     ldc  ra, 0x20(sp)
 *     cjalr zero, 0x0(ra)        <- return through the capability
 *
 * xlang/lua-cdp/capstone-lua/capstone_setjmp.S solves the same problem but for the
 * gp-captable ABI, where `ra` is a SCALAR (sd/ld, plain `ret`). Using that file here would
 * store a capability with `sd`, truncating it to 64 bits and losing the tag, and the first
 * longjmp would jump through an untagged value. The two forms are not interchangeable.
 *
 * jmp_buf: 224 bytes, and the buffer MUST be 16-byte aligned -- stc drops the tag on a
 * non-16-aligned address, so a misaligned jmp_buf would silently save an untagged ra and
 * fault only when longjmp is finally taken. stubinc/setjmp.h aligns it.
 *
 *   +0 ra  +16 sp  +32 s0  +48 s1  +64 s2  +80 s3  +96 s4  +112 s5
 *   +128 s6  +144 s7  +160 s8  +176 s9  +192 s10  +208 s11
 */
	.text

	.global setjmp
	.type setjmp, @function
setjmp:                         /* a0 = jmp_buf capability */
	stc   ra,  0(a0)            /* ra is a CAPABILITY under the default ABI */
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
	li    a0, 0                 /* the direct call returns 0 */
	cjalr zero, 0(ra)
	.size setjmp, .-setjmp

	.global longjmp
	.type longjmp, @function
longjmp:                        /* a0 = jmp_buf, a1 = value */
	ldc   ra,  0(a0)
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
	/* longjmp(buf, 0) must appear as 1 at the setjmp site. */
	mv    a0, a1
	bnez  a0, 1f
	li    a0, 1
1:
	cjalr zero, 0(ra)
	.size longjmp, .-longjmp
