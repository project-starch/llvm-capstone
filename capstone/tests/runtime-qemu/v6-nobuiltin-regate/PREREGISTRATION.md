# v6 landing re-gate on a -fno-builtin SDK: pre-registration, 2026-10-20, before any run

The v6 gate's kit SDK (v6-kit) lacked the campaign's `-fno-builtin`. Its malloc contract then read
`MALLOC_FAIL:failed VM growth preserves original`: allocation elision folded the growth check to false. That was
retracted as a runtime regression. The same unmodified contract with `-fno-builtin` passes.

**Kit `v6nb-kit`:**
- the SDK is v6-kit's with `-fno-builtin` appended to sdk.json codegen (effective setting verified);
- every image is rebuilt from the v6 landing tree at commit A aa31432c025f, which includes the runtime lane's `kept()`
  barrier in malloc-contract.c;
- the adapters are unchanged (module d073889f, vexec 2b4c2042).
- Recipe controls: contract, pin and frame-contract rebuilt with the default-flags SDK by the same command reproduce
  v6-kit's a5da753d, 15f3f59e and 06cd3e87 byte for byte, so `-fno-builtin` is their only difference.
- New images: contract c634fb95e447fa46, malloc b2d73ddb7e2008b3, pin 05068367b0bf6104, frame-contract
  8039490c8bb6c169, sqlite 224742db2fc19bae, mruby 92d674800b2f662c, and C's dense-churn contract d683db2acb7f2ff8.

**Predictions:** every reading equal to the v6 gate's, except step hooks.
- Tier 1 (A+B, and C's dense contract): PASS 26/41, the same 15 FALSE checks and 22 exits as gate6-t1-v6.
- Counters: (resume_sfences, satp_refreshes, satp_changed) with sfences = faults, refreshes = steps - 1, changed 0.
- Frame tests: multipage -22; twice -16 with child_ran=1; copy cause 25; overlap PASS.
- Step hooks: malloc16_ok_exit, malloc64_ok_exit and no_contract_failure now PASS. Point 4 is variance: it failed 1
  of 2 earlier and is reported either way.
- dense-churn at 4096: vm-contract ok first; "collect skipped" exactly once; progress >= 1000; ends 139 with cause 30
  at a CDERIVE.
