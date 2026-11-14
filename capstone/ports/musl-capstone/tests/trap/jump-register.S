/* a0 is the buffer, a1 the reference, a2 selects sigsetjmp. */
.text
.global jump_register
.type jump_register, @function
jump_register:
    cincoffsetimm sp, sp, -80
    stc ra, 0(sp)
    stc s2, 16(sp)
    stc a0, 32(sp)
    sd a2, 48(sp)
    movc s2, a1
    beqz a2, 1f
    li a1, 1
    call sigsetjmp
    j 2f
1:
    call setjmp
2:
    bnez a0, 3f
    ldc a0, 32(sp)
    ld a1, 48(sp)
    call jump_frame_one
    unimp
3:
    /* LCC selector 1 is 1 for a reference and 7 for an untagged value. */
    .insn r 0x5b, 1, 4, t0, s2, x1
    li t1, 1
    bne t0, t1, 4f
    lw t0, 0(s2)
    li t1, 42
    bne t0, t1, 4f
    /* longjmp(env, 0) must return 1. */
    addi a0, a0, -1
    j 5f
4:
    li a0, 9
5:
    ldc ra, 0(sp)
    ldc s2, 16(sp)
    cincoffsetimm sp, sp, 80
    ret
.size jump_register, .-jump_register

.type jump_frame_one, @function
jump_frame_one:
    cincoffsetimm sp, sp, -32
    stc ra, 0(sp)
    stc s2, 16(sp)
    li s2, 0
    call jump_frame_two
    unimp
.size jump_frame_one, .-jump_frame_one

.type jump_frame_two, @function
jump_frame_two:
    cincoffsetimm sp, sp, -32
    stc ra, 0(sp)
    stc s2, 16(sp)
    mv t0, a1
    li a1, 0
    beqz t0, 1f
    call siglongjmp
1:
    call longjmp
    unimp
.size jump_frame_two, .-jump_frame_two
