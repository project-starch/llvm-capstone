/* B0 (docs/plans/b0-silicon-delegated-runtime.md): link-time extents the runtime's C needs, returned as plain
 * INTEGERS.
 *
 * Under gp-captable a C reference to a linker-script symbol has no cap-table slot, so the compiler derives it from
 * gp (`scc rX, gp, rY; delin rX`), and a delin of a non-linear capability faults on silicon (C-13). The sizes
 * are what tls.c and hostcall.c actually need, and a difference of two pc-relative addresses is an integer: no
 * capability is formed. An extent that is NON-ZERO is refused in C (B0 has no glue support for reading those
 * regions yet); the image's own build gate checks the same thing first.
 *
 * Calls to these are direct `jal`s, which need no derivation. */
	.text
	.global __capstone_silicon_tls_tdata_bytes
__capstone_silicon_tls_tdata_bytes:
	lla	a0, __capstone_tdata_end
	lla	t0, __capstone_tls_image
	sub	a0, a0, t0
	ret

	.global __capstone_silicon_tls_mem_bytes
__capstone_silicon_tls_mem_bytes:
	lla	a0, __capstone_tls_end
	lla	t0, __capstone_tls_image
	sub	a0, a0, t0
	ret

	.global __capstone_silicon_init_array_bytes
__capstone_silicon_init_array_bytes:
	lla	a0, __init_array_end
	lla	t0, __init_array_start
	sub	a0, a0, t0
	ret

	.global __capstone_silicon_fini_array_bytes
__capstone_silicon_fini_array_bytes:
	lla	a0, __fini_array_end
	lla	t0, __fini_array_start
	sub	a0, a0, t0
	ret

/* HELLO's code range (hostcall.c): the launcher uses it only for diagnostics (its fault line and symbolization).
 * Under gp-captable a function pointer is an integer, so its capability bounds cannot be read; the range is the
 * image start to the globals boundary, where the monitor splits the code capability (sbi_capstone.c: dom_code ->
 * [base, base+gpoff)). Runtime addresses, from pc-relative lla. */
	.global __capstone_silicon_code_base
__capstone_silicon_code_base:
	lla	a0, __capstone_silicon_image_start
	ret

	.global __capstone_silicon_code_end
__capstone_silicon_code_end:
	lla	a0, __gpfree_globals_base
	ret

/* B1.6: the .tdata TEMPLATE, as a capability tls.c can read (musl's __copy_tls copies it into every thread's block).
 * The template is image data inside the globals region, so the monitor copied it into dom_data with the rest of the
 * blob, at its offset from __gpfree_globals_base (sbi_capstone.c's blob copy; RUN_CAP_INIT finds its table the same
 * way). The FIRST context's sp still has dom_data's base -- the table and the context arena are carved from the top
 * -- so the template is sp's base plus that offset. The capability keeps sp's bounds; tls.c narrows it to the
 * template. Valid only in the first context, which is where tls.c's describe_tls runs. */
	.global __capstone_silicon_tls_image
__capstone_silicon_tls_image:
	.insn r 0x5b, 0x1, 0x4, t0, sp, x3	/* lcc t0, sp, base */
	lla	t1, __capstone_tls_image
	lla	t2, __gpfree_globals_base
	sub	t1, t1, t2
	add	t0, t0, t1
	.insn r 0x5b, 0x1, 0x5, a0, sp, t0	/* scc a0, sp, t0 */
	li	t0, 0
	ret
