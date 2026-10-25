# The v6 landing gate: the module on the v6 launcher and SDK, and the -fno-builtin re-gate

Sibling records in this folder: `RESULT-five-item-revision-gate.md` (the module revision this gate carries over) and
`RESULT-sixth-change-gate-and-collections-census.md`. Readings are result lines; logs stay outside the repo.

**What was gated.** The v6 userspace runtime (the landing's commit A, the launcher, mallocng and the representability
predicate that ran on silicon, with collaborator PR #261 in the launcher) paired with the module of
`RESULT-five-item-revision-gate.md` plus collaborator PR #256's test counters. Module source 7e8478edac9bbf88.
QEMU-kit module d073889f18f89cd1, whose code equals the gated a5dec6a32c49b3d3 except two `__LINE__` immediates moved by
#256; board module 39d8492d5c18216b, byte-identical to the five-item revision's. Launcher capstone-vexec 2b4c20424874985b.
QEMU b290f5656600 (binary 7a4f3a64776f0c88), svadu=false. Pre-registered before step 1 ran.

## The gate as first run (SDK v6-kit: lane musl 0009 plus collaborator patch 0010)

| step | reading | prediction |
|---|---|---|
| 1 counters (resume_sfences, satp_refreshes, satp_changed) | baseline (0,0,0); no context (0,0,0); contract `ok` (426,537,0), 426 = the step's faults, 537 = steps − 1 | holds |
| 2 frame tests (two supervisor slots) | multipage −22; twice first=1, second −16, child_ran=1; copy cause 25; overlap PASS; 0 halts | holds |
| 3 step hooks (test-hooks module 57abc0aae18638d2) | 204 PASS, 6 FAIL: no_contract_failure, malloc16_ok_exit, malloc64_ok_exit, and point 4's three walk_accounting checks | **differs** |
| 4 Tier 1 | 26/41, the same 15 FALSE and the same 22 exits in order as the five-item revision | holds |
| 5 chain + mpt (the prebuilt images of the earlier gate; varies only the launcher and the two immediates) | 16 arms, 0 differ, 0 no-data (`compare-gate.py`, banner-gated refusals 0/0) | holds |

The contract step count fell from 547 to 538 against the five-item revision because patch 0010 opens a meta area's pages in
one call instead of one request per page; that is direct evidence 0010 is in the rebuilt images.

**Step 3's malloc failures were the SDK, not the runtime.** The contract read `MALLOC_FAIL:failed VM growth preserves
original`. v6-kit's SDK lacked the campaign's `-fno-builtin` (C-81), so `malloc`/`realloc` were library builtins and
LLVM folded the contract's expected-failure check away. That is upstream allocation-elision semantics, not a v6
regression; a regression reading was raised and retracted. Commit A wraps the two expected-failure reallocs in an
opaque barrier (`kept()`) so the contract reads the same under either setting. Point 4's three walk_accounting checks
had failed 1 of 2 on an earlier build and were pre-registered as variance.

## The re-gate on a -fno-builtin SDK (kit v6nb-kit)

Pre-registered on the v6-cursor-window lane at e2577424e045 (`tests/runtime-qemu/v6-nobuiltin-regate/`) before any
run. The SDK is v6-kit's with `-fno-builtin` appended (effective setting verified), and every image is rebuilt from
commit A. The recipe control: contract, pin and frame-contract rebuilt by the same command with the default-flags SDK
reproduce v6-kit's images byte for byte, so `-fno-builtin` is their only difference. Images: contract c634fb95e447fa46,
malloc b2d73ddb7e2008b3, pin 05068367b0bf6104, frame-contract 8039490c8bb6c169, the dense-churn contract d683db2acb7f2ff8.

| step | reading | prediction |
|---|---|---|
| Tier 1, A+B | 26/41; FALSE set, exits line and console counts (banner 2, `collect skipped` 2) equal to the first run's | holds |
| Tier 1, C's dense contract in place of the contract | the same | holds |
| counters | (426,536,0): 426 = faults, 536 = steps − 1 (537) | holds |
| frame tests | multipage −22; twice −16 with child_ran=1; copy cause 25; overlap PASS; 0 halts | holds |
| step hooks | 210 PASS, 0 FAIL: no_contract_failure, malloc16_ok_exit and malloc64_ok_exit pass; point 4 passes; `launcher_first` is reported (#256) | holds |
| dense-churn, node_flat_records=4096 (read back) | vm-contract `ok` exit 0 first; `collect skipped` exactly once; progress derived=3000; ends 139 with cause 30 at a CDERIVE (`.insn r 0x5b,1,0x51`, read at the faulting pc) | holds |

Every pre-registered reading holds.
