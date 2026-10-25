# mruby's release-differential corpus on silicon, bounds only: results (2026-10-16, caplifive_sublet_d93a8d66b)

Pre-registered in README.md (lane 13f97a853606, sums amended at 3bff031359fd, before the boot). One boot,
18:55-19:36:
- firmware 6bf139245abaabc1, Image af665ea2487e980a;
- 25 runs: 20 cases, the smoke control and the tool's four controls.

**Every row and every control is as pre-registered,** by outcome, cause and site: 21 of 21. smoke.rb printed
SMOKE_DONE and exited 0, and the tool's four controls are as declared.

| case | class | silicon | as predicted |
|---|---|---|---|
| 01, 07 | temporal | CAUGHT 24 in `mrb_vformat` (`lbu` at address 0x10) | yes |
| 02 | temporal | CAUGHT 24 in `mrb_vm_exec` (address 0x1604) | yes |
| 03 | temporal | WRONG-ANSWER (`sprintf with to_s mutating format string`) | yes; the corpus's recorded level0 arm read PASS |
| 04, 08, 10 | temporal | WRONG-ANSWER | yes |
| 09 | temporal | COMPLETED, `["PASS"]` | yes |
| 06 | temporal | CAUGHT 28 in `kh_get_set_val` (a declared site) | yes (QEMU 5) |
| 12, 14 | spatial | CAUGHT 28 in `ar_get`, `mrb_obj_ceqq` | yes (QEMU 5) |
| 13, 15, 16 | spatial | CAUGHT 28 in `mrb_pack_pack`, `str_prepend`, `ary_fill_exec` | yes (QEMU 7) |
| 17 | spatial + temporal | CAUGHT 24 in `mrb_func_basic_p` (`ldc` at 0x30274e0b65a71c20) | yes |
| 18-21 | null dereference | CAUGHT 24 in `mt_put` (address 0) | yes |
| 22 | type confusion | CAUGHT 24 in `mrb_proc_parameters` (`cincoffsetimm`) | yes |

**What it shows:**
- **mruby 4.0.0-rc2's own regression tests run on silicon** in the delegated interpreter. Under per-object bounds
  the defects manifest as on the emulator, run for run:
  - every spatial defect (12-16) is refused out of bounds (28) at the overflowing access;
  - every temporal defect either reaches a value that was never a capability (24), returns a wrong answer, or runs
    silently.
- **The arm revokes nothing,** so as pre-registered the temporal rows are MISSED as detections, whatever they read.
  The arm's own detections are the five spatial rows. Rows 18-22 are outside the temporal/spatial scope.
- **QEMU's 5 / 7 read 28, and its 24 reads 24.** This is the first arm with no revocation, and its reading is consistent
  with the 24 -> 25 of the earlier sets being Q-11's (revoked capabilities) only.

**Two observations recorded as read, not explained here:**
- **Every faulting case outlived the corpus rung's 120 s watchdog**: all 15 of them.
  - The other ten runs did not: the nine that returned, and the tool's `oob` control, which faults in its own small
    image.
  - Each faulting run printed capstone-exec's fault line, with its cause, pc and address, and the watchdog line came
    after it. The rows are read from that fault line.
  - The rung's retval, 15, counts those runs, and that is why the runner reported rc=1.
  - Runs took about 90-135 s each, judged from the watchdog's quiet stretches.
  - No earlier set showed this: 168 faulting runs in sets 2-9, with domains up to 128 MiB, all ended inside 120 s.
  - So it is specific to faults inside the interpreter image, in their fault path or the run's length, and it is not
    established which.
- **Case 22's fault address** reads 0x7d8000 on silicon where QEMU reported 0x0, at the same instruction and cause.
  What the RTL reports as the address of an UNEXPECTED_OPERAND on `cincoffsetimm` is not established here.

**Not run:** 05 (needs mruby-task), 11 (a C-API case), 23 (does not terminate). The reasons are in each arm's
manifest.

**Not established:** N = 1 per case. Nothing here is a revocation result: the revoking arm waits for a revoking
allocator in the silicon build.
