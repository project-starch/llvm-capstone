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
