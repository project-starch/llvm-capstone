/* Controls for sup-static-audit.py, generated as raw words (llvm-mc rejects some CSR aliases).
 * Every word between pos_start and pos_end must be flagged with the rule named; none between neg_start and neg_end. */
  .text
  .globl pos_start
pos_start:
  .word 0x34229073   # CSR csrw mcause,t0
  .word 0x343022f3   # CSR csrr t0,mtval (a read)
  .word 0xb0002373   # CSR csrr t1,mcycle
  .word 0x30002073   # CSR csrrs x0,mstatus,x0
  .word 0x3040d073   # CSR csrrwi mie (funct3 5)
  .word 0x800022f3   # CSR 0x800 (explicit list)
  .word 0x811022f3   # CSR 0x811 (explicit list)
  .word 0x30200073   # PRIV mret
  .word 0x10200073   # PRIV sret
  .word 0x10500073   # PRIV wfi
  .word 0x00000073   # TRAP ecall
  .word 0x00100073   # TRAP ebreak
  .word 0x400110db   # CALL
  .word 0x447312db   # NEST cssupervise
  .word 0x1a94105b   # ENTER capenter
  .word 0x4604105b   # VMOP 0x23
  .word 0x0012f05b   # CCSR cih
  .word 0x0102f05b   # CCSR cpmp0
  .word 0x01f0735b   # CCSR cpmp15
  .word 0x083100fb   # MINT capcreate
  .word 0x103100fb   # MINT capbound
  .globl neg_start
neg_start:
  .word 0xc00022f3   # csrr t0,cycle 0xc00
  .word 0xc02022f3   # csrr t0,instret 0xc02
  .word 0x003022f3   # frcsr 0x003
  .word 0x0002f05b   # ccsrrw ctvec 0x000
  .word 0x0022f05b   # ccsrrw cepc 0x002
  .word 0x0040715b   # ccsrrw 0x004
  .word 0x0202f05b   # ccsrrw 0x020 (past cpmp15)
  .word 0x4203935b   # RETURN 0x21
  .word 0x0c6111db   # split 0x06
  .word 0x183100fb   # custom-3 0x0c
  .word 0x123100fb   # custom-3 0x09 (outside mints)
  .word 0x063100fb   # custom-3 0x03 (outside mints)
  .word 0x083110fb   # custom-3 funct3 1 funct7 0x04
  .word 0x00c58533   # add
  .word 0x00813283   # ld t0,8(sp)
  .globl neg_end
neg_end:
