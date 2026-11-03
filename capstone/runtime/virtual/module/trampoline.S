/* Documentation source for the RV64I words in binfmt-dom.h. */
.option norvc
.text
.global trampoline
trampoline:
loop:
 mv a0,s0
 mv a1,s1
 li a2,0
 li a7,29
 ecall
 li t0,257
 beq a0,t0,protect
 blez a0,loop
 addi a0,a0,-1
 li a7,94
 ecall
 j .
protect:
 mv a0,a1
 mv a1,a2
 mv a2,a3
 li a7,226
 ecall
 mv a2,a0
 mv a0,s0
 mv a1,s2
 li a7,29
 ecall
 j loop
