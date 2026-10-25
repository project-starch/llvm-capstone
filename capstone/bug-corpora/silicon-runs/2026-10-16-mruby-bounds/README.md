# mruby's release-differential corpus on silicon, bounds only: pre-registration (2026-10-16, before the boot)

The board campaign's wave 4 (`docs/plans/2026-10-16-board-campaign-all.md`, mruby/release-differential). This is the
bounds-only arm, `silicon-mruby-bounds`. The revoking arm waits for a revoking allocator in the silicon build.

**What it runs.** Each case is upstream's own regression test (`trigger.rb`), extracted onto a self-contained
harness, run by one interpreter image:
- B5's `mruby.dom` a7ac283d7050a5d8, mruby 4.0.0-rc2 as a silicon delegated application. It is the image of boot P
  and the B5 pair, unchanged.
- Its allocator is `ports/mruby/app/silicon-alloc.c`: size classes, every object narrowed to exactly its request, no
  revocation.
- That is the corpus's `sysalloc-bounds` configuration (per-object bounds, nothing revoked) on a different allocator.

One arm, one boot on `caplifive_sublet_d93a8d66b`: private image af665ea2487e980a, firmware 6bf139245abaabc1, on the
exact-regions supervised-call line of sets 2-9. The image is staged once and every case's script beside it.

**The control** is the port's `scripts/smoke.rb`, the corpus's own control for its domain arm. It must print
SMOKE_DONE.
- It writes its file into `/tmp` on both platforms, so its M12 line reads `false` on both: smoke.rb is not in
  `/tmp`.
- The tool's four controls run too, with level0's default bounds.

**The outcomes** (kind `mruby-rb`, `tools/run-silicon-cases.py`):
- CAUGHT: any capability fault. The corpus's bounds-arm oracle is the cause, not a probe.
- COMPLETED: the harness printed `["PASS"]`, so the defect ran silently.
- WRONG-ANSWER: the harness printed its failures and exited 0.

**Three of the 23 are not run:**
- 05 needs mruby-task, which the silicon build does not carry (no timer HAL, no SIGALRM);
- 11 is a C-API case, which needs its own build;
- 23 does not terminate: its recorded reading is the watchdog.

## The silicon-mimic QEMU reading (capstone-qemu silicon/cderive-round ce1be0c642, `CAPSTONE_CDERIVE_ROUND=1`)

The smoke control completes (SMOKE_DONE), and the tool controls are as declared.

| case | class | QEMU | the corpus's recorded `sysalloc-bounds` reading |
|---|---|---|---|
| 01, 07 | temporal | CAUGHT 24 in `mrb_vformat` (`lbu` at address 0x10) | cause 24 |
| 02 | temporal | CAUGHT 24 in `mrb_vm_exec` (`lbu`, 0x1604) | cause 24 |
| 03 | temporal | **WRONG-ANSWER** (`sprintf with to_s mutating format string`) | **PASS** |
| 04, 08, 10 | temporal | WRONG-ANSWER | wrong answer |
| 09 | temporal | COMPLETED, `["PASS"]` | PASS |
| 06 | temporal | CAUGHT 5 in `kh_get_set_val` (a declared site) | cause 5 |
| 12, 14 | spatial | CAUGHT 5 in `ar_get`, `mrb_obj_ceqq` | cause 5 |
| 13, 15, 16 | spatial | CAUGHT 7 in `mrb_pack_pack`, `str_prepend`, `ary_fill_exec` | cause 7 |
| 17 | spatial + temporal | CAUGHT 24 in `mrb_func_basic_p` (`ldc` through a non-capability) | cause 24 |
| 18-21 | null-deref | CAUGHT 24 in `mt_put` (`lw` at address 0) | cause 24 |
| 22 | type confusion | CAUGHT 24 in `mrb_proc_parameters` (`cincoffsetimm` on a null base) | cause 24 |

19 of 20 rows read as the corpus recorded them. Case 03 is the one difference. Its stale format string reads back
changed under B5's size classes, where the recorded level0 arm found it intact. That is the allocator's reuse
pattern, not the platform's: both arms leave the stale read unchecked.

## The board prediction

Every row as QEMU read it, by outcome and site. Causes map as follows:
- **5 / 7 read 28**, the RTL's OUT_OF_BOUNDS, as in every earlier set.
- **24 stays 24.** This arm revokes nothing, so ISSUES Q-11's 24 -> 25 cannot apply. Each 24 here is a value that was
  never a capability (a null, or an integer used as a pointer), and the RTL raises UNEXPECTED_OPERAND on it.
- **The sites carry over.** Four of the five 24 sites are loads. The fifth is case 22's `cincoffsetimm` on a null
  base, which the RTL also refuses: `CINCOFFSETIMM` raises UNEXPECTED_OPERAND on a NOT_CAP base
  (`core/anvil_build/capstone_flu_unit.anvil:57-60` at d93a8d66b; the register form does not, `SILICON-BLOCKER.md`).

**What it does and does not test:**
- A **temporal** case caught here is not caught by revocation. The 24s are untagged values the defect reached, and
  the arm has no lifetimes.
- So for the temporal rows this arm is pre-registered **MISSED as a detection**, whatever it reads. What it records
  is how the defect manifests on silicon under per-object bounds.
- The spatial rows (12-16) are the arm's own detections: per-object bounds.

**New on silicon, and what a failure would say:**
- A smoke control that does not finish inside the corpus rung's 120 s watchdog. It took about 65 s in B5's own rung.
- A cause other than 28 on a 5 / 7 row: the RTL's bounds check is not the one QEMU models.
- A 24 row that completes: the RTL accepts an operand QEMU refuses.
