# RUN: llvm-mc -filetype=obj -triple=capstone64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj --file-headers %t | FileCheck --check-prefix=HDR %s
# RUN: ld.lld -m elf64lcapstone %t.o -o %t
# RUN: llvm-readobj --file-headers %t | FileCheck --check-prefix=HDR %s
# RUN: echo 'OUTPUT_FORMAT(elf64-littlecapstone)' > %t.script
# RUN: ld.lld %t.script %t.o -o %t
# RUN: llvm-readobj --file-headers %t | FileCheck --check-prefix=HDR %s

# HDR:      ElfHeader {
# HDR-NEXT:   Ident {
# HDR-NEXT:     Magic: (7F 45 4C 46)
# HDR-NEXT:     Class: 64-bit (0x2)
# HDR-NEXT:     DataEncoding: LittleEndian (0x1)
# HDR-NEXT:     FileVersion: 1
# HDR-NEXT:     OS/ABI: SystemV (0x0)
# HDR-NEXT:     ABIVersion: 0
# HDR-NEXT:     Unused: (00 00 00 00 00 00 00)
# HDR-NEXT:   }
# HDR-NEXT:   Type: Executable (0x2)
# HDR-NEXT:   Machine: 0x103
# HDR-NEXT:   Version: 1
# HDR-NEXT:   Entry:

.globl _start
_start:
  nop


