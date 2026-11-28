## Capstone stage 2: EF_CAPSTONE_CRVC (0x0100). An image with it is decoded in
## the CRVC mode, so an object with compressed instructions but without the flag
## (it may hold C.FLD and the other displaced forms) cannot join it. An object
## without EF_CAPSTONE_RVC has no 16-bit instruction and can.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=capstone64 -mattr=+c,+cap-crvc crvc.s -o crvc.o
# RUN: llvm-mc -filetype=obj -triple=capstone64 -mattr=+c rvc.s -o rvc.o
# RUN: llvm-mc -filetype=obj -triple=capstone64 norvc.s -o norvc.o

# RUN: ld.lld crvc.o norvc.o -o ok
# RUN: llvm-readelf -h ok | FileCheck %s --check-prefix=OK
# OK: Flags: 0x101

# RUN: ld.lld rvc.o norvc.o -o plain
# RUN: llvm-readelf -h plain | FileCheck %s --check-prefix=PLAIN
# PLAIN: Flags: 0x1{{$}}

# RUN: not ld.lld crvc.o rvc.o -o bad 2>&1 | FileCheck %s --check-prefix=ERR
# RUN: not ld.lld rvc.o crvc.o -o bad 2>&1 | FileCheck %s --check-prefix=ERR
# ERR: error: rvc.o: cannot link an object with compressed instructions but without EF_CAPSTONE_CRVC into an image with it, from crvc.o

#--- crvc.s
.globl _start
_start:
  c.movc c8, c9
  ret

#--- rvc.s
.globl f
f:
  c.nop
  ret

#--- norvc.s
.globl g
g:
  ret
