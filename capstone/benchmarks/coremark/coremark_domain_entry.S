# coremark_domain_entry.S - Capstone domain entry point for CoreMark
#
# domain_main(void *arg /*a0*/, unsigned func /*a1*/)
#   func == 0 (CAPSTONE_DPI_CALL):         run CoreMark
#   func == 1 (CAPSTONE_DPI_REGION_SHARE): store shared region capability
#
# Written in assembly because the Capstone LLVM backend cannot generate
# correct code for this function at any -O level:
#
#   -O0  The prologue emits  cincoffsetimm s0, sp, N  (rd!=rs1) which consumes
#        sp (LINEAR type).  The epilogue's subsequent ldc ra, N(sp) then
#        crashes: "Cap mem access requires capability" on sp.tag=0.
#
#   -O1/-O2  The compiler merges the CALL and REGION_SHARE paths into a
#        shared sink block.  Inside the sink the same base register is used
#        first as rs1 in cincoffsetimm (which consumes it because it is
#        LINEAR) and then as the base for a scalar sw, crashing on tag=0.
#
# Key Capstone capability rules followed here:
#   - Memory loads/stores (lw, sw, ldc, stc, ld, sd) do NOT consume their
#     base register (rs1).
#   - Capability arithmetic (cincoffset, cincoffsetimm) with rd!=rs1 CONSUMES
#     rs1 (clears its tag) because capabilities are LINEAR by default.
#   - gp (x3) is auto-initialized from pc_cap(cursor=0) by any cincoffset /
#     cincoffsetimm where rs1==gp and gp.tag==0 (in PRV_C mode).
#   - cjalr resets gp = pc_cap(cursor=0) on every intra-domain call entry.
#   - cincoffsetimm rd, rs1, imm with rd==rs1 is an in-place cursor update
#     and does NOT consume rs1.

	.text
	.globl	domain_main
	.type	domain_main, @function
domain_main:
	sext.w	a1, a1
	li	a2, 1
	bne	a1, a2, .Ldm_call

	# ------------------------------------------------------------------
	# REGION_SHARE path: a0 = shared region capability, func == 1
	# ------------------------------------------------------------------

	# a2 = capability to g_region_count; gp consumed; a1 becomes plain integer
.Lpcrel_hi_grc:
	auipc	a1, %pcrel_hi(g_region_count)
	addi	a1, a1, %pcrel_lo(.Lpcrel_hi_grc)
	cincoffset	a2, gp, a1	# a2 = &g_region_count; gp consumed
	lw	a1, 0(a2)		# a1 = g_region_count; a2 not consumed
	bnez	a1, .Ldm_try_payload

	# g_region_count == 0: hc_metadata = a0
.Lpcrel_hi_hcm0:
	auipc	a3, %pcrel_hi(hc_metadata)
	addi	a3, a3, %pcrel_lo(.Lpcrel_hi_hcm0)
	cincoffset	a3, gp, a3	# a3 = &hc_metadata; gp auto-init+consumed
	stc	a0, 0(a3)		# hc_metadata = region cap
	j	.Ldm_incr

.Ldm_try_payload:
	li	a3, 1
	bne	a1, a3, .Ldm_incr

	# g_region_count == 1: hc_payload = a0
.Lpcrel_hi_hcp0:
	auipc	a3, %pcrel_hi(hc_payload)
	addi	a3, a3, %pcrel_lo(.Lpcrel_hi_hcp0)
	cincoffset	a3, gp, a3	# a3 = &hc_payload; gp auto-init+consumed
	stc	a0, 0(a3)		# hc_payload = region cap

.Ldm_incr:
	# a2 is still a valid capability to g_region_count: it was only used as
	# the base for memory loads/stores, which do NOT consume the base cap.
	lw	a1, 0(a2)		# reload g_region_count
	addiw	a1, a1, 1
	sw	a1, 0(a2)		# g_region_count++
	cjalr	zero, 0(ra)		# return (no stack frame needed)

	# ------------------------------------------------------------------
	# CALL path: func != 1 -- run CoreMark
	# ------------------------------------------------------------------
	# a0 = &res (OpenSBI result buffer capability, unsigned *).
	# OpenSBI's call_domain() reads *res after domreturn to form the call_dom()
	# return value.  We must write *a0 = 0 (HC_V0_RET_DONE) before returning.
	# Save a0 now: it is overwritten below when we load hc_metadata.
	# Frame layout: sp+0 = ra (cap, 16 B), sp+16 = &res (cap, 16 B) = 32 B.
	# cincoffsetimm rd==rs1=sp: in-place cursor update, does NOT consume sp.
.Ldm_call:
	cincoffsetimm	sp, sp, -32
	stc	ra, 0(sp)		# save return address
	stc	a0, 16(sp)		# save &res capability

	# Load hc_metadata; if non-NULL zero its length field (offset 24).
.Lpcrel_hi_hcm1:
	auipc	a0, %pcrel_hi(hc_metadata)
	addi	a0, a0, %pcrel_lo(.Lpcrel_hi_hcm1)
	cincoffset	a0, gp, a0	# a0 = &hc_metadata global; gp consumed
	ldc	a0, 0(a0)		# a0 = hc_metadata value (capability)
	beqz	a0, .Ldm_do_call	# if NULL, skip
	sd	zero, 24(a0)		# hc_metadata->length = 0

.Ldm_do_call:
.Lpcrel_hi_cm:
	auipc	a0, %pcrel_hi(coremark_main)
	addi	a0, a0, %pcrel_lo(.Lpcrel_hi_cm)
	cincoffset	a0, gp, a0	# a0 = &coremark_main; gp auto-init+consumed
	cjalr	ra, 0(a0)		# call; ra = address of next instruction below
	ldc	ra, 0(sp)		# restore ra (sp.tag still 1)
	ldc	a0, 16(sp)		# restore &res capability
	cincoffsetimm	sp, sp, 32	# free frame; rd==rs1, sp NOT consumed
	sw	zero, 0(a0)		# *res = 0 = HC_V0_RET_DONE
	cjalr	zero, 0(ra)		# return to start.S (which calls domreturn)
.Lfunc_end_dm:
	.size	domain_main, .Lfunc_end_dm-domain_main
