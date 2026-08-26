# Why an unmodified MicroPython gets no temporal protection from Capstone.
#
# heap-bounds-model.c models what gc_alloc does: hand out a pointer into one
# large static array. Compiled for capstone64 -O2 with our own clang.
#
# Read sub_alloc: the compiler DOES set bounds, and it sets them to the whole
# 384 KiB object (lui a3, 96 -> 96 << 12 -> 393216). cincoffset then moves the
# cursor inside those bounds. So every block MicroPython sub-allocates carries a
# capability spanning the ENTIRE heap, and write_through is a bare sb that the
# hardware has no reason to reject.
#
# Command:
#   clang --target=capstone64 -O2 -S heap-bounds-model.c

	.attribute	4, 16
	.attribute	5, "rv64i2p1"
	.file	"heap-bounds-model.c"
	.text
	.globl	sub_alloc                       # -- Begin function sub_alloc
	.p2align	2
	.type	sub_alloc,@function
sub_alloc:                              # @sub_alloc
# %bb.0:                                # %entry
	cincoffsetimm	sp, sp, -32
	stc	ra, 16(sp)                      # 16-byte Folded Spill
	stc	s0, 0(sp)                       # 16-byte Folded Spill
	movc	s0, sp
	cincoffsetimm	s0, s0, 32
.Lpcrel_hi0:
	auipc	a1, %pcrel_hi(heap)
	addi	a1, a1, %pcrel_lo(.Lpcrel_hi0)
	cincoffset	a1, gp, a1
	delin	a1
	lcc	a2, a1, 2
	lui	a3, 96
	add	a3, a2, a3
	shrink	a1, a2, a3
	cincoffset	a0, a1, a0
	ldc	ra, 16(sp)                      # 16-byte Folded Reload
	ldc	s0, 0(sp)                       # 16-byte Folded Reload
	cincoffsetimm	sp, sp, 32
	cjalr	zero, 0(ra)
.Lfunc_end0:
	.size	sub_alloc, .Lfunc_end0-sub_alloc
                                        # -- End function
	.globl	write_through                   # -- Begin function write_through
	.p2align	2
	.type	write_through,@function
write_through:                          # @write_through
# %bb.0:                                # %entry
	cincoffsetimm	sp, sp, -32
	stc	ra, 16(sp)                      # 16-byte Folded Spill
	stc	s0, 0(sp)                       # 16-byte Folded Spill
	movc	s0, sp
	cincoffsetimm	s0, s0, 32
	sb	a1, 0(a0)
	ldc	ra, 16(sp)                      # 16-byte Folded Reload
	ldc	s0, 0(sp)                       # 16-byte Folded Reload
	cincoffsetimm	sp, sp, 32
	cjalr	zero, 0(ra)
.Lfunc_end1:
	.size	write_through, .Lfunc_end1-write_through
                                        # -- End function
	.globl	whole                           # -- Begin function whole
	.p2align	2
	.type	whole,@function
whole:                                  # @whole
# %bb.0:                                # %entry
	cincoffsetimm	sp, sp, -32
	stc	ra, 16(sp)                      # 16-byte Folded Spill
	stc	s0, 0(sp)                       # 16-byte Folded Spill
	movc	s0, sp
	cincoffsetimm	s0, s0, 32
.Lpcrel_hi1:
	auipc	a0, %pcrel_hi(heap)
	addi	a0, a0, %pcrel_lo(.Lpcrel_hi1)
	cincoffset	a0, gp, a0
	delin	a0
	lcc	a1, a0, 2
	lui	a2, 96
	add	a2, a1, a2
	shrink	a0, a1, a2
	ldc	ra, 16(sp)                      # 16-byte Folded Reload
	ldc	s0, 0(sp)                       # 16-byte Folded Reload
	cincoffsetimm	sp, sp, 32
	cjalr	zero, 0(ra)
.Lfunc_end2:
	.size	whole, .Lfunc_end2-whole
                                        # -- End function
	.type	heap,@object                    # @heap
	.local	heap
	.comm	heap,393216,32
	.ident	"clang version 22.0.0git (https://github.com/project-starch/llvm-capstone 97cf978388d6f3671e788c660881f9a732003595)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym heap
