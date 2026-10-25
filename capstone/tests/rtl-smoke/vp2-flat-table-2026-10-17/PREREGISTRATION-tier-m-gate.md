# The tier-m gate: pre-registered 2026-10-20 before any run

The tree is integration/pr-tier-m 5fae9ed2848e:
- dev caa4d987;
- collaborator PR #242: 2 GiB per mapping, 4 GiB per process, 8192 mappings, with the runtime lane's two conflict
  resolutions;
- PR #257: CV_ADD checks pinned frames for duplicates in O(n);
- PR #254: the retire granule sweep;
- the cursor-window fix, dc1ad0a8a6fb.

Dev has since moved by ad598d2c, which touches only capstone/bug-corpora. None of those paths is in this tree's
diff, so the gated runtime and module bytes are the rebased tree's.

## Builds

- **Modules** (build-adapter.sh, buildroot cross-gcc, -march=rv64g, flat table): QEMU tm-qemu-flat 7a2283ac195bfd2f;
  board tm-board-flat deb6dcf834e335ff; test hooks tmtest-qemu-flat c5ccd49b5a6bc0ae. Launcher (QEMU kit)
  ab7880387be9eb6a.
- **Board module census:** vermagic `6.4.14 SMP riscv`; 0 compressed of 4,663 instructions; 0 QEMU-only CSR
  accesses; vp2-silicon-gate (trusted_S) 0 refused / 0 candidates / 0 attributed / 4 conditional / 0 unimp. That is
  the v6 module 39d8492d's profile (4,535 instructions).
- **SDK:** built with CAPSTONE_SDK_CORPUS=1; check-sdk-builtins reads effective -fno-builtin.
- **Images:** rebuilt from this tree (kit tm-kit): contract 9c41c82ae623d470, malloc f4cacf6c87457249, pin
  a4269b5c90a1b619, frame-contract b74d93c0ebd459a2, sqlite3 5d4fe13aa06ec41e, mruby 704bffc8bd5f8621.
- **Image censuses:** delin 5/5/4 with 0 gp-derived, and residual writers ldc 5/5/4, as on the v6 kit; 0
  QEMU-only CSR accesses in all four images. Negated own cursor 0 and negated base 0 in contract and malloc,
  where the v6 images read 2 + 2 before the cursor-window fix.
- **Kits:** qkit7-tm (one slot) and qkit7-tms2 (two slots) are qkit7-v6 and qkit7-v6s2 with the adapter swapped.
  QEMU 7a4f3a64776f0c88, b290f5656600, svadu=false.

## Predictions: equal to the v6 gate everywhere except where #242 changes something on purpose

1. **Counters.** Baseline and no-context read (0,0,0). Contract `ok`: resume_sfences = that step's faults,
   satp_refreshes = steps - 1, satp_changed 0.
2. **Frame tests (two slots).** multipage -22; twice: first > 0, second -16, child_ran=1; copy cause 25; overlap
   PASS; 0 halts.
3. **Step hooks.** Every check PASS, including malloc16_ok_exit, malloc64_ok_exit and no_contract_failure.
   malloc-contract's expected-failure growth now asks for CV_MAX_REGION_BYTES + 1 (2 GiB + 1). That is refused
   early in `__capstone_malloc_map` (`bytes > CV_MAX_REGION_BYTES`), as 512 MiB was refused before. launcher_first is
   reported.
4. **Tier 1: 45 checks.** #242 adds the vm-large and vm-mappings steps, and the checks vm-large, vm-mappings,
   vm_large_contract and vm_mappings_contract.
   - The 41 existing checks read as in gate6-t1-v6: the same 15 FALSE.
   - The `vm` step still exits 1 at "execute through mapping PCC", before its realloc check (which #242 moved to
     2 GiB + 1), so that change cannot reach a reading.
   - The 4 new checks PASS: vm-large exits 0 with VIRTUAL_VM_LARGE_OK, and vm-mappings exits 0 with
     VIRTUAL_VM_MAPPINGS_OK 1200. That gives 30/45.
   - The exits line is the v6 gate's 22 in order, with vm-large:0 and vm-mappings:0 after the contract loop's
     modes. Console banner 2.
   - **Flip:** a new check reading FALSE is a #242 finding, reported before anything lands. These are the cases
     the 2 GiB cap and the 8192 mappings exist to make pass: growth above 256 MiB, VA above 1 GiB, and more than
     512 mappings.
5. **Chain, 15 arms (the prebuilt nb images).**
   - Every arm's collected images equal the v6 chain's.
   - Every case is identical to gate6-t3-v6 (compare-gate.py), with banner-gated refusals 0.
   - No flip: these images carry their own libc caps (256 MiB, from t3-freeze's runtime), so the module's larger
     caps are never reached, and (predicted, not counted) no corpus case maps more than 512 regions.
   - #257 and #254 are predicted to change no reading. #254's premise (one 8-byte store clears a granule's tag) is
     confirmed on 8522d72fe.
6. **mpt, two-sided.** The v6 gate's mpt was built from t3-freeze's builtins SDK, which dev's runner now refuses.
   So mpt runs on qkit7-v6 and on qkit7-tm with t3-freeze-nb's SDK, through dev's gated runner. Images are equal
   between the two, and so is every case.

Any Tier 1 or chain difference is reported before anything lands.
