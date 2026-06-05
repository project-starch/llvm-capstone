# ee_printf_asm.S - Assembly trampoline for ee_printf on Capstone PureCap
#
# The compiler-generated va_list mechanism stores the variadic-argument
# pointer as a SCALAR integer (sd, not stc).  When reloaded via ld the
# register tag is 0 and any subsequent memory dereference crashes in
# cap_mem mode ("Cap mem access requires capability").
#
# Fix: intercept the call before va_list is involved.  The RISC-V calling
# convention passes all arguments in a0-a7 regardless of whether the
# callee is variadic, so we simply forward a0-a7 unchanged to
# ee_printf_impl(const char *fmt, void *a1..a7).  No va_list is ever
# created.
#
# Capstone capability rules followed here (same as coremark_domain_entry.S):
#   - cincoffsetimm rd==rs1 is an in-place cursor update: does NOT consume.
#   - Memory ops (stc, ldc, sb, lb, ...) do NOT consume their base register.
#   - cincoffset/cincoffsetimm with rd!=rs1 CONSUMES rs1 (LINEAR by default).
#   - gp is auto-initialized from pc_cap(cursor=0) when rs1==gp and tag==0.

	.text
	.globl	ee_printf
	.type	ee_printf, @function
ee_printf:
	# a0=fmt, a1-a7=variadic args — already in the right registers for
	# ee_printf_impl.  Just save ra, call the impl, restore ra, return.
	cincoffsetimm	sp, sp, -16	# in-place (rd==rs1=sp): sp not consumed
	stc	ra, 0(sp)		# save return-address capability
.Lpcrel_hi_impl:
	auipc	t0, %pcrel_hi(ee_printf_impl)
	addi	t0, t0, %pcrel_lo(.Lpcrel_hi_impl)
	cincoffset	t0, gp, t0	# t0 = &ee_printf_impl; gp auto-init+consumed
	cjalr	ra, 0(t0)		# call; return value in a0
	ldc	ra, 0(sp)		# restore ra (sp still valid: only used as base)
	cincoffsetimm	sp, sp, 16	# free frame (in-place: rd==rs1=sp)
	cjalr	zero, 0(ra)		# return to caller
.Lfunc_end_ee_printf:
	.size	ee_printf, .Lfunc_end_ee_printf-ee_printf
