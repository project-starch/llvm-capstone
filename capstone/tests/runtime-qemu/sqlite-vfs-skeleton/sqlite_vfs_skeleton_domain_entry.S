#ifndef __CAPSTONE_SQLITE_VFS_DOMAIN_ENTRY_BUILTIN__
#define __CAPSTONE_SQLITE_VFS_DOMAIN_ENTRY_BUILTIN__
#define cincoffset(rd, rs1, rs2) .insn r 0x5b, 0x1, 0xc, rd, rs1, rs2
#define cincoffsetimm(rd, rs, offset) .insn i 0x5b, 0x2, rd, offset(rs)
#define ldc(rd, rs, offset) .insn i 0x5b, 0x3, rd, offset(rs)
#define stc(rs1, rs2, offset) .insn s 0x5b, 0x4, rs1, offset(rs2)
#define cjalr(rd, rs) .insn i 0x5b, 0x5, rd, 0(rs)
#endif

.text
  .global domain_main
  .type domain_main, @function
domain_main:
  cincoffsetimm(sp, sp, -0x310)
  stc(ra, sp, 0x300)
  cincoffsetimm(a2, sp, 0)

.Lpcrel_hi_sqlite_vfs_skeleton_domain_impl:
  auipc t0, %pcrel_hi(sqlite_vfs_skeleton_domain_impl)
  addi t0, t0, %pcrel_lo(.Lpcrel_hi_sqlite_vfs_skeleton_domain_impl)
  cincoffset t0, gp, t0
  cjalr(ra, t0)

  ldc(ra, sp, 0x300)
  cincoffsetimm(sp, sp, 0x310)
  cjalr(zero, ra)

