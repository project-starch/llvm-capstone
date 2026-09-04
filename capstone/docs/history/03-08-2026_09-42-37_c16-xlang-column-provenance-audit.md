# C-16 vs the xlang Capstone column: provenance audited, column unaffected

**Question (raised by the team, 2026-08-03).** 9 of the xlang Capstone column's
12 temporal catches surface as QEMU's `assert(rs1_v->tag)` — the same assert a
C-16 miscompile trips. Was the column measured with a pre-fix compiler, and if
so, are any of those catches actually C-16 artifacts?

**Answer: the column WAS measured with a pre-fix compiler, and it is provably
unaffected.** The timing argument offered with the question ("fix landed 15:30,
column committed 17:57") is refuted; the conclusion survives on stronger
evidence.

## Timeline (all 2026-08-02, from git and file mtimes)

- 13:37–13:38 — column build; runs follow (obj dirs and logs in the repro SHARE)
- 15:30 — C-16 fix lands (`1faf145ef1c7`)
- 17:57 — column committed (`f0cacdd3c194`)
- The clang that built the column: built 2026-07-28 21:39; DWARF producer of the
  measured binaries names `f7132369bdb5`, which does NOT contain the fix.
  `.ninja_log` has zero entries on 2026-08-02 — the compiler was not rebuilt
  between fix and measurement on this machine.

## Evidence the column is unaffected (three legs + adversarial audit)

1. **Binary identity (load-bearing).** Compiler rebuilt 2026-08-03 ~09:19 from
   HEAD (`036bdf31b614`, contains the fix; rebuilt `SelectionDAG.cpp.o` and
   `libLLVMSelectionDAG.so` carry the fix's new `DstPtrTy` symbol in DWARF).
   All 30 domain variants rebuilt with it are **byte-identical** to the measured
   pre-fix binaries after `llvm-objcopy --strip-debug --remove-section=.comment`
   (code AND data; raw files differ only in debug info, decisively the DWARF
   producer string). The fix is a no-op for these translation units.
2. **No C-16 signature in the binaries.** C-16's shape is `PseudoTRUNC_CAP`
   (expands to `mv`) feeding an offset `ADDI` into a memset destination. The
   memset libcall path IS live (6 of 30 doms call `memset`, all 30 call
   `mock_memset0`), but every `mv` in all 30 binaries is a self-move feeding no
   memset, and no `addi rd, rs, N` with `rd != rs` exists. The defect, not just
   the fix, is absent.
3. **Control-lane discrimination.** Every temporal row's control (same binary
   path minus the runtime revoke; `.text` differs by 16 bytes, one resolved
   call) ran to completion (MISS). A C-16 miscompile is revocation-independent
   and would assert in both lanes. (Limitation: says nothing about rows 6/11,
   which FAULT in both lanes by design — but those assert nowhere; they fault
   `cause=7`/`cause=5`, not C-16's signature.)
4. **End-to-end re-run on the fixed compiler.** Rows 1 (`rlua_userdata_uaf`)
   and 5 (`mruby_kwargs_uaf`): `REPRODUCED 2/2 rows identical to
   expected-results.tsv` — revoke FAULT via the same assert, control MISS.
   (With leg 1 in hand this only re-confirms run-to-run reproducibility.)

A claim-auditor pass tried to refute the conclusion and CONFIRMED it; its named
weakest link is that leg 2 is a static-disassembly argument rather than an
execution trace. The optional nail (not run): boot one revoke domain under
`qemu -d in_asm` and read the last translated block before the abort.

## Corrections found on the way

- **The assert group is 9 rows, not 8.** Row 7 (`secp256k1_preallocated_uaf`,
  added last when the original row 7 was replaced) is assert-flavor per its log,
  but the manifestation table in `xlang/capstone/RESULTS.md` was written before
  row 7 landed and never updated ("all 11 temporal rows" likewise). Corrected
  to 9 (rows 1, 3, 4, 5, 7, 8, 10, 13, 15) in `xlang/capstone/RESULTS.md`,
  `xlang/RESULTS.md`, and the paper-facing measurements doc.
- **Helper precision:** 8 of the 9 assert in `helper_cscincoffsetimm`
  (`op_helper.c:626`); row 1 asserts in `helper_cscincoffset` (`op_helper.c:598`).
  Same C-16-adjacent helper family, so the helper name discriminates nothing.

## Open question relayed to the board lane

This box has no post-fix compiler build before 2026-08-03 and no non-xlang
artifacts dated 2026-08-02, so the C-16 "Verified" evidence in ISSUES.md must
have been produced on the board lane's own machine. Worth a 10-second check
there: `DW_AT_producer` of an Aug-2 verification artifact should name a commit
containing `1faf145ef1c7`. If it names `f7132369bdb5`, that verification ran on
a pre-fix compiler and needs re-running. (Not evaluable from this machine;
nothing here contradicts the board lane's result.)

## Evidence artifacts

- Pre-fix originals preserved (bit-exact copy of the measured SHARE dir):
  `/tmp/capstone/xlang-capstone-repro-PREFIX-COMPILER-20260802/` — /tmp, not
  durable; hashes below are the durable record.
- Post-fix rebuild + re-run: `/tmp/capstone/xlang-capstone-repro-postfix/`

    # sha256 of debug-stripped (.debug_*, .comment removed) domain binaries
    # pre-fix = xlang-capstone-repro-PREFIX-COMPILER-20260802 (clang built 2026-07-28)
    # post-fix = xlang-capstone-repro-postfix (clang built 2026-08-03, contains 1faf145ef1c7)
    bbc27ea95f34c7382f78c7fd23d5ce70d416997b58c88f28437f65f93db450da  IDENTICAL  xlang_libpulse_iterator_uaf.dom
    719efd9043e0f1ae641695dfafcc74b46b32db6791217bca4dd4137c01196ab8  IDENTICAL  xlang_libpulse_iterator_uaf_norevoke.dom
    d178ba43922544a50dcdad57c4c5b2c2e48d9168de9e1ce2e58ccf5194e1df71  IDENTICAL  xlang_mruby_bytecode_overflow.dom
    8dee109a23e75aa4323831c6506944b6fc98d68eaaaa7630432388915e09e7fa  IDENTICAL  xlang_mruby_bytecode_overflow_norevoke.dom
    6f44b1a9b701cc4d3b4108d685199cb5d8f715c74651e5f23b3e889326e3d034  IDENTICAL  xlang_mruby_gc_stackroot_uaf.dom
    7dfd96c4a1a33079a0a951d813c5dae169ef7df7c0bae3d2216301ebf1544140  IDENTICAL  xlang_mruby_gc_stackroot_uaf_norevoke.dom
    aaada37ca3e793453404d40d2ea207d1f6b5d64fe1e9b14fa415cbc59ad3a029  IDENTICAL  xlang_mruby_hash_slice_uaf.dom
    f563b037f6ad5dca7b9907787f8d048f31b8d290f89f83d988a0da0cb997d8bd  IDENTICAL  xlang_mruby_hash_slice_uaf_norevoke.dom
    5119f77cfac9fa5dcd3cd8c76f70f446a201a50a2fb7257a1e0095ea0b4ee8e3  IDENTICAL  xlang_mruby_io_dataptr_uaf.dom
    d76a18ea5bad8d31098237c9d72c689e721d667015340400226d0b4989120999  IDENTICAL  xlang_mruby_io_dataptr_uaf_norevoke.dom
    39102733c757ffa96403bf7d2aaa2b89169231acbcb31f2e765ac3ebee9da253  IDENTICAL  xlang_mruby_irep_pool_uaf.dom
    ce5dd10e55647bdb5781758f40334eeeffce614268b2bf2e704f4cc22006fbd0  IDENTICAL  xlang_mruby_irep_pool_uaf_norevoke.dom
    a95c93fffd6207ee7ab989e28a4ec4665501ed5b365513ef73c69c810cc630d5  IDENTICAL  xlang_mruby_kwargs_uaf.dom
    aacb8cc47e51629260945b47232af5bbccbbeb41608e3f0868fa2a616c61ceac  IDENTICAL  xlang_mruby_kwargs_uaf_norevoke.dom
    2a5e26a6ad38b614fc03fe929af7cfc3b3ee8eddad861c2d461ac0ecd2eb9b29  IDENTICAL  xlang_mruby_range_uaf.dom
    f112f8f6460753d85b9fe9a8b5e450261b91360f2aa6510b20f7a6c0d220429f  IDENTICAL  xlang_mruby_range_uaf_norevoke.dom
    5a03b843806fb2b6ed1ecd94afca0fbd4b68838015ac722cb476fd373a20714f  IDENTICAL  xlang_mruby_sprintf_argv_uaf.dom
    3b70106591430042cb26fc8d8594013c043e7da1cd07d8f7d07b2d878a46bd89  IDENTICAL  xlang_mruby_sprintf_argv_uaf_norevoke.dom
    364a9280d0eaa7344629d14b49eec394806d4cf252b22a2e389754b676d35f74  IDENTICAL  xlang_mruby_upvar_overflow.dom
    b5bc086cfb807210edbec1cc802a6dfea82adce885bb9e6f54a6b8ebdec0e407  IDENTICAL  xlang_mruby_upvar_overflow_norevoke.dom
    6a79caf7afcd32429751bc04d2fae1ae54232b1133868e63616e2eb36e33cbfc  IDENTICAL  xlang_mruby_values_at_uaf.dom
    fdc1417e30b7b07278d5897f5a42408bd5777440136b8ff842478b93fd1d70d0  IDENTICAL  xlang_mruby_values_at_uaf_norevoke.dom
    7c5fe4ffc777cd9f441500c6432a6925209f74b6d8c53209e363443534a4f2fa  IDENTICAL  xlang_mruby_vmstack_uaf.dom
    f04c6db705edcc4beeb59464840f6b3af565dd3e25170b23b6c3ae7bb420a3a0  IDENTICAL  xlang_mruby_vmstack_uaf_norevoke.dom
    5d5dfc06bf17e94ac2ec9273bcd79f0185b9cb0049e1b8b834118f18839a9869  IDENTICAL  xlang_rlua_escaped_handle_uar.dom
    797543d40c9c9f3e248a52b71aa9a6d9c9c94265f7dd1e8c1478321de30ad567  IDENTICAL  xlang_rlua_escaped_handle_uar_norevoke.dom
    4eeb300f35bffb2741b659ed15edc4ee748b5a8c15eab1074424f08f57d5a683  IDENTICAL  xlang_rlua_userdata_uaf.dom
    e4658837a3e6c7cb9ff10e34a294a8adf5e0361a14c66e54a48f15c9782040cb  IDENTICAL  xlang_rlua_userdata_uaf_norevoke.dom
    337710a733c8d51dedaf951af780f953e4381dbd48bc46d68c7f5a97320cdbdb  IDENTICAL  xlang_secp256k1_preallocated_uaf.dom
    76865dbfda4ff2b32688b575dff259e9cfe57c07954622421bc516b3600c2724  IDENTICAL  xlang_secp256k1_preallocated_uaf_norevoke.dom
