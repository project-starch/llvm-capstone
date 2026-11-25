# CREVOKE run lengths in the B5b-2 SPEC images: pre-registration (2026-11-25)

**Why.** The rtl lane reports that on silicon a CREVOKE walk longer than about 1,048,574 cycles makes a verdict-miss
LDC behind it time out and take a spurious cause 25. At about 46 cycles per node that is about 22,700 nodes. A fix is
coming. This count says which stage and arm rows are exposed until it lands.

**What is counted.** The run length of every CREVOKE and CRENEW execution: the records after the operand that are
deeper than it. The model's walk (cap_rev_tree.c walk_run, its read-only first pass) visits those records. The
operand's own record and the record that ends the run are read too, and are not in the count.
- Per phase: executions, max, p99, the number over 1,000 and over 20,000 (strictly above), by op.
- The max at 300 s and 590 s into each stage, so a stage cut at 600 s shows whether its max was still growing.

**What it cannot distinguish.** It counts the MODEL's run. It predicts the silicon walk only if the CREVOKE_S walk
visits exactly the run. If the silicon walk visits other records too (a free-list or table scan, or a search for the
stop record beyond the run), this count is a lower bound on the silicon walk, not its length. rtl's word on what
CREVOKE_S visits decides how to read it.

## Instrument

capstone-qemu sublet/crevoke-scan-count fa6ce7554d, on 741be49bb0 (the untagged-arith counter), on 3a9fc39167 (the
trap-route twin's source). The binary is pinned at c4/qbin/qemu-scan-fa6ce7554d, sha256 1da6e8b57877ce56.
- With CAPSTONE_CREVOKE_SCAN_COUNT=<file> set, the helpers record cap_rev_last_scan after the tree operation, before
  the status can raise.
- <file>.hist holds the exact histogram of run lengths (below 2^21), rewritten every 100 ms when it changed.
- <file> gets a BIG line for every run of 1,000 or more, with pc, privilege and host ms.

Unset, the helpers do nothing more.

## Positive control (first two phases of the boot)

crevoke-ctl.dom (sha256 4f223793ff8dfcd3, built with the B5T SDK's capstone-cc -O1).
- Both arguments malloc and free one 256 KiB buffer.
- With 1 it also derives four parents inside it, gives them 0, 10, 1500 and 25000 one-byte children, and ends each
  parent with one CREVOKE.

The run's scripts (init, make.sh, drive.py, hist.py) stay with the run's local files, because they carry host paths.
hist.py's check (host-tested on synthetic snapshots: the pass, a mutant missing the 25000 run that must fail, and an
unreadable snapshot that must fail):
- ctl1 minus ctl0 must be exactly one CREVOKE each with runs 0, 10, 1500 and 25000;
- nothing may be missing, and CRENEW must be equal.

If the check fails, the driver ends the boot at once, and no stage is read.

## Platform and procedure

The trap-route twin: the trapk linux-6.1 kernel 749fed2606e3ebca (pinned copy), sbi-keep-run fw-y, -m 963M,
sublet.dom_binfmt=1, the board buildroot rootfs, repacked root-owned.
- The stages are b5b2spec/board-stages/<ARM> with the boards' run.sh, B5TL1 then B5T, each in the order lbm, mcf,
  nab, xz, perlbench, gcc, imagick, x264.
- x264's ldecod line is replaced by a cp of the host-decoded BuckBunny.yuv (c3a3d3e2..., the board's ldecod output).
- One boot, under the host-wide QEMU lock.
- Each phase is bracketed by @@SC-BEGIN/@@SC-END with 2 s of guest quiet on each side. The driver copies .hist at each
  marker and 300 s and 590 s into each stage.
- Each stage is killed 600 s after it starts. A killed stage's numbers cover its first 600 s only.

## Predictions

1. **Control:** as above, or the boot is void.
2. **Every stage, both arms: every CREVOKE and every CRENEW has a run of 0.** So max 0, p99 0, none over 1,000, none
   over 20,000. No row is exposed. The issuers, found by encoding (funct7 0x52/0x54 under opcode 0x5b, funct3 1) and
   named from the source:
   - **User, all arms:** mallocng's free_slot (1 site) and __libc_realloc (2 on B5T, 4 on B5TL1) end the object that
     malloc derived (heap_issue). The object has no children, since no SPEC program derives inside its own heap
     objects.
   - **User, B5TL1 only:** __heap_unmapping (runtime/virtual/malloc-mapping.c, the round-evict mapping cache) CRENEWs a
     group mapping's owner when mallocng gives the mapping back. mallocng gives a group back only when every slot is
     free, and each free has already ended its object. So the owner's run is empty.
   - **Supervisor:** the trap kernel's sublet_ref_revoke (munmap of a block's last pages), sublet_block_rollback, and
     root_derive's grant unwind. A block is unmapped after its objects are freed, for the same reason. The kernel has
     no CRENEW: its two raw 0x54 matches are inside trim_bitmaps and ZSTD_resetCCtx_internal, i.e. other bytes, as are the 0x52 matches in four network and MMC driver functions.
   - The per-binary site counts: 3 CREVOKE on B5T; 5 CREVOKE and 1 CRENEW on B5TL1; 2 in crevoke-ctl.dom (the control
     and free). They were checked by a raw-word scan of every stage binary.
3. **CRENEW:** 0 executions on every B5T stage. On B5TL1 the count is not predicted (one per mapping given back to
   the cache), and the run of each is 0.
4. **Privilege:** U and S only, no M.

A version of this pre-registration that predicted no kernel CREVOKE and no CRENEW on any arm was never pushed. A
mnemonic grep had missed the kernel's CV_KILL and the lean runtime's heap_renew call. The encoding scan above
replaced it.

A stage whose max is above 0 refutes prediction 2 for that stage. Its BIG lines (pc, privilege) then name the issuer.
