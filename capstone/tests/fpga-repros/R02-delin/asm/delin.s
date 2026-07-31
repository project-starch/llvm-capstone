	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.file	"coremark_matrix_fpga_app.c"
	.text
	.globl	domain_main                     # -- Begin function domain_main
	.p2align	2
	.type	domain_main,@function
domain_main:                            # @domain_main
# %bb.0:                                # %entry
	cincoffsetimm	sp, sp, -208
	sd	ra, 192(sp)                     # 16-byte Folded Spill
	stc	s0, 176(sp)                     # 16-byte Folded Spill
	movc	s0, sp
	cincoffsetimm	s0, s0, 208
	movc	a2, a0
                                        # kill: def $x10 killed $x11
	cincoffsetimm	a0, s0, -48
	stc	a0, -112(s0)                    # 16-byte Folded Spill
	stc	a2, 0(a0)
	cincoffsetimm	a2, s0, -52
	sw	a1, 0(a2)
	ldc	a1, 0(a0)
	li	a0, 1
	sd	a0, 520(a1)
.Lpcrel_hi0:
	auipc	a0, %pcrel_hi(ladder_rd_minstret)
	addi	a0, a0, %pcrel_lo(.Lpcrel_hi0)
	stc	a0, -192(s0)                    # 16-byte Folded Spill
	jalr	a0
	movc	a1, a0
	ldc	a0, -112(s0)                    # 16-byte Folded Reload
	cincoffsetimm	a2, s0, -64
	stc	a2, -128(s0)                    # 16-byte Folded Spill
	sd	a1, 0(a2)
	ldc	a1, 0(a0)
	li	a0, 2
	sd	a0, 520(a1)
.Lpcrel_hi1:
	auipc	a0, %pcrel_hi(ladder_rd_mcycle)
	addi	a0, a0, %pcrel_lo(.Lpcrel_hi1)
	stc	a0, -208(s0)                    # 16-byte Folded Spill
	jalr	a0
	cincoffsetimm	a1, s0, -72
	stc	a1, -144(s0)                    # 16-byte Folded Spill
	sd	a0, 0(a1)
.Lpcrel_hi2:
	auipc	a0, %pcrel_hi(coremark_matrix_compute)
	addi	a0, a0, %pcrel_lo(.Lpcrel_hi2)
	jalr	a0
	movc	a1, a0
	ldc	a0, -208(s0)                    # 16-byte Folded Reload
	cincoffsetimm	a2, s0, -76
	stc	a2, -176(s0)                    # 16-byte Folded Spill
	sw	a1, 0(a2)
	jalr	a0
	movc	a1, a0
	ldc	a0, -192(s0)                    # 16-byte Folded Reload
	cincoffsetimm	a2, s0, -88
	stc	a2, -160(s0)                    # 16-byte Folded Spill
	sd	a1, 0(a2)
	jalr	a0
	ldc	a5, -176(s0)                    # 16-byte Folded Reload
	ldc	a3, -160(s0)                    # 16-byte Folded Reload
	ldc	a4, -144(s0)                    # 16-byte Folded Reload
	ldc	a2, -128(s0)                    # 16-byte Folded Reload
	ldc	a1, -112(s0)                    # 16-byte Folded Reload
	movc	a6, a0
	cincoffsetimm	a0, s0, -96
	sd	a6, 0(a0)
	lwu	a5, 0(a5)
	ldc	a6, 0(a1)
	sd	a5, 0(a6)
	ld	a3, 0(a3)
	ld	a4, 0(a4)
	sub	a3, a3, a4
	ldc	a4, 0(a1)
	sd	a3, 8(a4)
	ldc	a4, 0(a1)
	lui	a3, 13
	addi	a3, a3, 158
	sd	a3, 16(a4)
	ld	a0, 0(a0)
	ld	a2, 0(a2)
	sub	a0, a0, a2
	ldc	a1, 0(a1)
	sd	a0, 512(a1)
	ld	ra, 192(sp)                     # 16-byte Folded Reload
	ldc	s0, 176(sp)                     # 16-byte Folded Reload
	cincoffsetimm	sp, sp, 208
	ret
.Lfunc_end0:
	.size	domain_main, .Lfunc_end0-domain_main
                                        # -- End function
	.p2align	2                               # -- Begin function ladder_rd_minstret
	.type	ladder_rd_minstret,@function
ladder_rd_minstret:                     # @ladder_rd_minstret
# %bb.0:                                # %entry
	cincoffsetimm	sp, sp, -48
	sd	ra, 32(sp)                      # 16-byte Folded Spill
	stc	s0, 16(sp)                      # 16-byte Folded Spill
	movc	s0, sp
	cincoffsetimm	s0, s0, 48
	#APP
	csrr	a1, minstret
	#NO_APP
	cincoffsetimm	a0, s0, -40
	sd	a1, 0(a0)
	ld	a0, 0(a0)
	ld	ra, 32(sp)                      # 16-byte Folded Reload
	ldc	s0, 16(sp)                      # 16-byte Folded Reload
	cincoffsetimm	sp, sp, 48
	ret
.Lfunc_end1:
	.size	ladder_rd_minstret, .Lfunc_end1-ladder_rd_minstret
                                        # -- End function
	.p2align	2                               # -- Begin function ladder_rd_mcycle
	.type	ladder_rd_mcycle,@function
ladder_rd_mcycle:                       # @ladder_rd_mcycle
# %bb.0:                                # %entry
	cincoffsetimm	sp, sp, -48
	sd	ra, 32(sp)                      # 16-byte Folded Spill
	stc	s0, 16(sp)                      # 16-byte Folded Spill
	movc	s0, sp
	cincoffsetimm	s0, s0, 48
	#APP
	csrr	a1, mcycle
	#NO_APP
	cincoffsetimm	a0, s0, -40
	sd	a1, 0(a0)
	ld	a0, 0(a0)
	ld	ra, 32(sp)                      # 16-byte Folded Reload
	ldc	s0, 16(sp)                      # 16-byte Folded Reload
	cincoffsetimm	sp, sp, 48
	ret
.Lfunc_end2:
	.size	ladder_rd_mcycle, .Lfunc_end2-ladder_rd_mcycle
                                        # -- End function
	.p2align	2                               # -- Begin function coremark_matrix_compute
	.type	coremark_matrix_compute,@function
coremark_matrix_compute:                # @coremark_matrix_compute
# %bb.0:                                # %entry
	cincoffsetimm	sp, sp, -112
	sd	ra, 96(sp)                      # 16-byte Folded Spill
	stc	s0, 80(sp)                      # 16-byte Folded Spill
	movc	s0, sp
	cincoffsetimm	s0, s0, 112
	ldc	a1, 0(gp)
.Lpcrel_hi3:
	auipc	a0, %pcrel_hi(core_init_matrix)
	addi	a4, a0, %pcrel_lo(.Lpcrel_hi3)
	li	a0, 666
	movc	a2, zero
	cincoffsetimm	a3, s0, -96
	jalr	a4
	movc	a1, a0
	cincoffsetimm	a0, s0, -100
	sw	a1, 0(a0)
	lw	a0, 0(a0)
	ld	ra, 96(sp)                      # 16-byte Folded Reload
	ldc	s0, 80(sp)                      # 16-byte Folded Reload
	cincoffsetimm	sp, sp, 112
	ret
.Lfunc_end3:
	.size	coremark_matrix_compute, .Lfunc_end3-coremark_matrix_compute
                                        # -- End function
	.p2align	2                               # -- Begin function core_init_matrix
	.type	core_init_matrix,@function
core_init_matrix:                       # @core_init_matrix
# %bb.0:                                # %entry
	cincoffsetimm	sp, sp, -160
	sd	ra, 144(sp)                     # 16-byte Folded Spill
	stc	s0, 128(sp)                     # 16-byte Folded Spill
	movc	s0, sp
	cincoffsetimm	s0, s0, 160
                                        # kill: def $x14 killed $x12
                                        # kill: def $x14 killed $x10
	cincoffsetimm	a4, s0, -36
	sw	a0, 0(a4)
	cincoffsetimm	a0, s0, -64
	stc	a1, 0(a0)
	cincoffsetimm	a0, s0, -68
	sw	a2, 0(a0)
	cincoffsetimm	a1, s0, -96
	stc	a3, 0(a1)
	cincoffsetimm	a2, s0, -100
	movc	a1, zero
	sw	a1, 0(a2)
	cincoffsetimm	a3, s0, -148
	li	a2, 1
	sw	a2, 0(a3)
	cincoffsetimm	a2, s0, -156
	sw	a1, 0(a2)
	cincoffsetimm	a2, s0, -160
	sw	a1, 0(a2)
	lw	a0, 0(a0)
	bnez	a0, .LBB4_2
	j	.LBB4_1
.LBB4_1:                                # %if.then
	cincoffsetimm	a1, s0, -68
	li	a0, 1
	sw	a0, 0(a1)
	j	.LBB4_2
.LBB4_2:                                # %if.end
	j	.LBB4_3
.LBB4_3:                                # %while.cond
                                        # =>This Inner Loop Header: Depth=1
	cincoffsetimm	a0, s0, -160
	lw	a0, 0(a0)
	cincoffsetimm	a1, s0, -36
	lw	a1, 0(a1)
	bgeu	a0, a1, .LBB4_5
	j	.LBB4_4
.LBB4_4:                                # %while.body
                                        #   in Loop: Header=BB4_3 Depth=1
	cincoffsetimm	a0, s0, -156
	lw	a1, 0(a0)
	addiw	a1, a1, 1
	sw	a1, 0(a0)
	lw	a0, 0(a0)
	mulw	a0, a0, a0
	slliw	a0, a0, 3
	cincoffsetimm	a1, s0, -160
	sw	a0, 0(a1)
	j	.LBB4_3
.LBB4_5:                                # %while.end
	cincoffsetimm	a0, s0, -156
	lw	a0, 0(a0)
	addiw	a1, a0, -1
	cincoffsetimm	a0, s0, -100
	sw	a1, 0(a0)
	cincoffsetimm	a1, s0, -64
	ldc	a1, 0(a1)
	cincoffsetimm	a2, s0, -128
	stc	a1, 0(a2)
	ldc	a1, 0(a2)
	#APP
	.insn r 91, 1, 3, a1, zero, zero
	#NO_APP
	stc	a1, 0(a2)
	lw	a0, 0(a0)
	ld	ra, 144(sp)                     # 16-byte Folded Reload
	ldc	s0, 128(sp)                     # 16-byte Folded Reload
	cincoffsetimm	sp, sp, 160
	ret
.Lfunc_end4:
	.size	core_init_matrix, .Lfunc_end4-core_init_matrix
                                        # -- End function
	.type	cm_memblk,@object               # @cm_memblk
	.local	cm_memblk
	.comm	cm_memblk,682,16
	.ident	"clang version 22.0.0git (https://github.com/project-starch/llvm-capstone 2ffd621a1ab0f0fa899457530531e80790dc2771)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym ladder_rd_minstret
	.addrsig_sym ladder_rd_mcycle
	.addrsig_sym coremark_matrix_compute
	.addrsig_sym core_init_matrix
	.addrsig_sym cm_memblk
	.section	.capstone_gp_table,"a",@progbits
	.p2align	3, 0x0
	.quad	1
	.quad	682
	.quad	16
	.quad	0
