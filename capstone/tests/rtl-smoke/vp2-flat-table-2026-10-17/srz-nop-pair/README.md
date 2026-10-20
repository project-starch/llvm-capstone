# srz / srz-nop: a one-variable pair for pb10's FRTY 2 -> 0 under srevroot-zero (the runtime lane's request)

**Control:** 607164b9c1407c7c, the srevroot-zero module pb10-srz ran (source ce25d6de7280, capstone_vm.c
ac120a4b98ef72a5). It is rebuilt from that source by the recipe here, byte-identical.

**Variant:** 2a8f85f70d8efcdd7cfd30d97cc1c2f7dbcdedbb3f30a8940a8ac0ef1d11c379. The same source with the two fused
`csrw 0x5c1, zero` that follow a CSRUNV replaced by `nop`, and the banner tag changed to `srevroot-nopf-1`, of the
same length, so .rodata keeps its layout. restore_root's write is unchanged.
- Both: 4,577 instructions, 0 RVC, .text 18,080 bytes.
- .text differs in exactly 3 words: offsets 5520, 15836 and 16804, each 0x5c101073 (csrw 0x5c1, x0) to 0x00000013
  (nop). The 3 inlined sites of the 2 asm statements.
- srz-check: control CSRUNV 6 (both parameter branches compiled), 3 followed by the zeroing write; variant 0.

**The limit, stated before any run:** a CSR write also flushes the pipeline at commit, and a nop does not. So this
pair separates "the post-CSRUNV write" from "no write", but not "the zeroed srevroot (a namespace edge)" from "the
flush". A third arm with a flushing write that leaves srevroot unchanged would separate those two.

**Predictions** (run on pb10 on the same kit, one module each):
- control: FRTY 0, as pb10-srz read;
- variant: FRTY 2, as pb10 without srz read, if the post-CSRUNV write is what removed it; FRTY 0 if something else in
  the srz module did (for example the restore_root write, which both keep).
