/* Positive control for BREAK severity, and a NEGATIVE control that must stay silent. */
#define cincoffset(rd, rs1, rs2) .insn r 0x5b, 0x1, 0xc, rd, rs1, rs2
#define stc(rs1, rs2, offset) .insn s 0x5b, 0x4, rs1, offset(rs2)
#define split(rd, rs1, rs2) .insn r 0x5b, 0x1, 0x6, rd, rs1, rs2
#define delin(rd) .insn r 0x5b, 0x1, 0x3, rd, x0, x0

.section .text
.global must_report_break_B
must_report_break_B:
  split(a2, sp, t3)          /* a2 LINEAR, never delin'd            */
  stc(a2, sp, 32)            /* rule B site                          */
  sd a2, 64(sp)              /* READ of a2 before redefinition       */
  ret

.global must_report_break_A
must_report_break_A:
  split(a2, sp, t3)          /* a2 LINEAR                            */
  cincoffset(a4, a2, x0)     /* rule A: rd != rs1                    */
  sd a2, 72(sp)              /* READ of a2 before redefinition       */
  ret

.global must_stay_silent
must_stay_silent:
  split(a2, sp, t3)
  delin(a2)                  /* now provably NONLIN                  */
  cincoffset(a4, a2, x0)     /* rd != rs1 but source is NONLIN       */
  stc(a2, sp, 32)            /* NONLIN source, no clear required     */
  sd a2, 80(sp)
  ret
