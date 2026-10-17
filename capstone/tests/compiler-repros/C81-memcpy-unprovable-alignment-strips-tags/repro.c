/* A constant-size memcpy whose 16-alignment the compiler cannot PROVE is expanded into
 * scalar chunks sized to the copy's alignment, which strips every capability tag in the
 * range. The control is the same copy with the alignment provable from the type.
 *
 * Both functions have external linkage and take their operands as parameters, so nothing
 * interprocedural can specialise them -- that matters, see the README's note on IPSCCP.
 *
 * Compile-only: the signature is in the disassembly. The run-time half (the tag reading
 * back 0, and 1 for the control) is runtime/virtual/hosted-contract.c's
 * memcpy_constant_size_unknown_alignment check, which this shape is the reduction of.
 */
/* Declared, not included: this repro runs -nostdinc so it needs no sysroot, and a probe
 * that can fail on a missing header turns every unrelated error into its finding. clang
 * still recognises memcpy by name and inlines it. */
typedef unsigned long size_t;
void *memcpy(void *, const void *, size_t);

/* THE DEFECT. `void *` carries no alignment, so EffAlign is 1 and the 32-byte copy comes
 * out as 32 one-byte loads and stores. A plain `sb` never carries a tag. */
void copy32_unknown(void *d, const void *s) { memcpy(d, s, 32); }

/* THE CONTROL. The type makes both ends 16-aligned and the size a multiple of 16, so the
 * capability path takes it as `ldc`/`stc` units and the tags survive. */
struct cap16 { void *a; void *b; };
void copy32_known(struct cap16 *d, const struct cap16 *s) { memcpy(d, s, 32); }

/* THE DEFECT AT EffAlign 8, which is the shape the fix must also cover. `unsigned long *`
 * tells the compiler 8, so the copy comes out as 4x ld + 4x sd. A region the compiler knows
 * only as 8-aligned can be 16-aligned AT RUN TIME and hold a capability, and a plain `sd`
 * strips its tag exactly as `sb` does. A fix written for the byte-granular shape alone would
 * repeat C-78's mistake: fixing the shape of the reproducer instead of the defect. The
 * criterion is size >= 16 with operands not PROVABLY 16-aligned, whatever the unit. */
void copy32_align8(unsigned long *d, const unsigned long *s) { memcpy(d, s, 32); }

/* A SECOND CONTROL, for the claim that -fno-builtin is only a PARTIAL workaround:
 * __builtin_memcpy is inlined even under -fno-builtin (ISSUES.md C-78, :8574-8577), so
 * this one byte-expands in both arms. If it ever stops doing so, -fno-builtin became a
 * complete workaround and the README's scoping paragraph is stale. */
void copy32_unknown_builtin(void *d, const void *s) { __builtin_memcpy(d, s, 32); }
