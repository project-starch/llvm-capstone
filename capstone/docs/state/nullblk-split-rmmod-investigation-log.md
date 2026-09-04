# Split null_blk rmmod blocker - resolved investigation log

## 1. Current State & Symptoms

- **Status:** resolved on 2026-06-10. `capstone/tests/runtime-qemu/run-nullblk-split-rmmod.sh` now passes after QEMU interrupt-delivery fixes and cleanup of investigation diagnostics.
- **The Blocker:** historically, `capstone/tests/runtime-qemu/run-nullblk-split-rmmod.sh` did not complete. The visible failing component was split `null_blk` unload interacting with Linux block teardown in `del_gendisk()`, but the root cause was below Linux in Capstone/QEMU timer interrupt delivery.
- **Observed Symptoms:** focused kernel instrumentation placed the unload hang at `del_gendisk(nullb->disk)`, specifically after `device_del()` and before return from `blk_mq_freeze_queue_wait(q)`. The diagnostic line was:
  - `capstone freeze_wait disk=nullb0 q=... init_done=1 flags=4042 is_zero=0 depth=1`
- **Observed Symptoms:** with normal partition scanning restored, split module registration performs one partition-scan BIO and balances it:
  - `capstone qref get ... caller=submit_bio_noacct_nocheck+0x92/0x19a`
  - `null_blk: capstone split null_submit_bio: entry op=0 sector=0 sectors=8`
  - `null_blk: capstone split null_submit_bio: exit op=0 sector=0 sectors=8`
  - `capstone qref put ... caller=__submit_bio+0x8c/0x112`
- **Observed Symptoms:** despite the balanced BIO, immediate `rmmod null_blk` still reaches `blk_mq_freeze_queue_wait()` with `percpu_ref_is_zero(&q->q_usage_counter) == 0` and times out in the QEMU smoke harness.
- **Observed Symptoms:** a targeted `del_gendisk()` diagnostic now proves the split path stalls before the original freeze wait. After `blk_queue_start_drain(q)`, a diagnostic `synchronize_rcu()` prints `09a before diagnostic synchronize_rcu` and never reaches `09b after diagnostic synchronize_rcu` in the split run.
- **Observed Symptoms:** the same diagnostic in the baseline null_blk unload does complete:
  - `capstone del_gendisk nullb0: 09a before diagnostic synchronize_rcu`
  - `capstone del_gendisk nullb0: 09b after diagnostic synchronize_rcu`
  - `capstone freeze_wait ... is_zero=1 depth=1`
  This means the visible `blk_mq_freeze_queue_wait()` hang is downstream of missing RCU/percpu-ref progress, not proof of an ordinary leaked request reference.
- **Observed Symptoms:** existing QEMU trace events show a split-vs-baseline timer difference. In the baseline trace, after `__BEFORE_RMMOD__`, there are 44 `capstone_h_int H-int 7` events and unload reaches `__AFTER_RMMOD__`. In the split trace, after `__BEFORE_RMMOD__`, there are zero `H-int 7` timer events, one `H-int 9` external event, and no `09b` / `__AFTER_RMMOD__`.
- **Observed Symptoms:** OpenSBI `print_regions()` exposed through a temporary `SBI_EXT_CAPSTONE_REGION_PRINT_ALL` ecall shows the region dump after split module init and before `del_gendisk()` has the same shape. This weakens the earlier theory that a fresh partition-scan BIO directly leaks a new CPMP entry before unload.
- **Observed Symptoms:** a separate post-`insmod` progress issue was isolated. The permanent module-load share of `metadata_region` with `CAPSTONE_ANNOTATION_REV_SHARED` can stall guest timer/process progress. `sleep 1` hangs immediately after `insmod /nullb/capstone_split/null_blk.ko` in affected builds.
- **Historical reproduction:** source the test environment and run:
  ```bash
  source capstone/tests/capstone-test-env.sh
  bash capstone/tests/runtime-qemu/run-nullblk-split-rmmod.sh
  ```
- **Historical guest command:** equivalent guest command used by the wrapper while
  debugging:
  ```sh
  dmesg -n 8 && modprobe configfs && /null_blk.user && insmod /nullb/capstone_split/null_blk.ko && echo __BEFORE_RMMOD__ && rmmod null_blk && echo __AFTER_RMMOD__
  ```
  The current wrapper uses `dmesg -n 1` to avoid unrelated remote-fence log floods
  and still checks `__BEFORE_RMMOD__` / `__AFTER_RMMOD__`.
- **Reproduction:** `/null_blk.user` must run first. Running the split module without that setup crashes QEMU during `insmod` in `helper_csrevoke` with `Assertion rs1_v->val.cap.type == CAP_TYPE_REV`.

## 2. Tested Hypotheses & Eliminated Tracks

- **Partition scan suppression with `GENHD_FL_NO_PART`:** added `disk->flags |= GENHD_FL_NO_PART` in split `null_gendisk_register()`. *Result & why it failed:* the rmmod hang still reproduced. It also removed the registration-time partition-scan BIO that baseline normally balances, so this should not be kept as a fix.
- **Full `struct nullb_device` copyback corruption:** replaced `memcpy(dev, nullb_dev_region_base, sizeof(struct nullb_device))` with scalar copyback of validated configuration fields. *Result & why it failed:* the rmmod hang still reproduced. The scalar copyback is safer than copying kernel-owned fields back from the domain, but it did not resolve the blocker.
- **Uncompleted request path:** instrumented split `null_submit_bio()` and `null_queue_rq()` with entry/exit logs. *Result & why it failed:* early rmmod failures originally showed no request-path logs, ruling out a simple outstanding split request. After restoring partition scanning, the one observed BIO completed and its queue reference was put.
- **Generic block registration failure:** instrumented `blk_register_queue()` and `blk_mq_freeze_queue_wait()`. *Result & why it failed:* registration is normal:
  - `blk_register_queue disk=nullb0 ... ret=0 init_done=0 flags=400040`
  - `blk_register_queue switch_to_percpu disk=nullb0 ...`
  The freeze later still sees `is_zero=0`.
- **Baseline null_blk regression:** ran `run-nullblk-baseline.sh` under the same instrumented kernel. *Result & why it failed:* baseline passes. It reaches freeze with `is_zero=1`, so the issue is split-specific.
- **`/null_blk.user` alone poisoning guest state:** ran `/null_blk.user && sleep 1`. *Result & why it failed:* QEMU smoke passed. Domain creation alone is not sufficient.
- **Module region setup without disk creation:** ran split module with `nr_devices=0` followed by `sleep 1`. *Result & why it failed:* with the original permanent metadata share, `sleep` hung; after removing that permanent share and borrowing `metadata_region` per domain call, the `nr_devices=0` probe passed. This identifies one real issue but not the full rmmod blocker.
- **`null_validate_conf` domain call as remaining full-device trigger:** temporarily bypassed the split-domain `null_validate_conf` call while keeping the metadata lifetime change. *Result & why it failed:* full-device `insmod ... && sleep 1` still hung, so that call is not the remaining trigger.
- **Post-`add_disk()` RCU grace-period wait:** added `synchronize_rcu()` after successful `add_disk()`. *Result & why it failed:* the actual rmmod wrapper still timed out; this is not just a missing local grace period in split `null_gendisk_register()`.
- **Forcing an RCU grace period at the actual drain point:** added a diagnostic-only `synchronize_rcu()` immediately after `blk_queue_start_drain(q)` in `del_gendisk()`. *Result & why it failed as a fix:* baseline passes this point and split hangs inside the diagnostic `synchronize_rcu()`. This confirms the problem is not fixed by waiting; the split environment is failing to make the timer/RCU progress that the wait needs.
- **Timer callback discriminator in `del_gendisk()`:** temporarily armed a Linux timer before the diagnostic `synchronize_rcu()`. *Result & why it failed:* this changed behavior enough to make baseline hang too, so it was removed and should not be used as evidence.
- **Stale region/CPMP entry after partition-scan BIO:** exposed OpenSBI `print_regions()` to S-mode and dumped regions after module init and before `del_gendisk()`. *Result & why it failed:* the two split dumps did not show a new shape appearing between init and unload. Region/CPMP state may still matter, but the current evidence does not support a simple "partition scan leaked a new CPMP entry" explanation.
- **QEMU existing trace events:** ran split and baseline with `-trace enable=capstone_h_int`, `capstone_dom_switch_async`, `capstone_dom_switch_sync`, and `riscv_write_timecmp`. *Result:* baseline continues to receive timer H-interrupts through rmmod; split stops seeing `H-int 7` before `/null_blk.user` domain creation and never sees another timer event through the unload hang. `riscv_write_timecmp` only appeared in the QEMU command line, not as runtime trace events, likely because the Capstone SBI timer path writes `mtimecmp` through the capability directly.

## 3. Resolved Cause & Fix

- The active lead was confirmed: split `null_blk` domain activity changed Capstone
  timer interrupt progress. After split setup, external H-interrupts still arrived
  but timer H-interrupt progress stopped, preventing RCU/percpu-ref completion
  during `del_gendisk()`.
- ACLINT/QEMU trace events showed that, after split domain creation, `mtimecmp`
  writes and ACLINT timer callbacks stopped, while the baseline continued to
  program timers and receive timer H-interrupts.
- A return-path trace showed the final relevant timer H-interrupt was taken with
  `mie=0` in M-mode. The Capstone async handler posted `MIP_MTIP`, but because
  `mie.MTIP` was disabled, OpenSBI did not handle/reprogram the timer. No later
  `mtimecmp` write occurred before the unload hang.
- The QEMU fix is in:
  - `capstone/capstone-qemu/target/riscv/cpu_helper.c`
  - `capstone/capstone-qemu/target/riscv/csr.c`
- Fix details:
  - `riscv_cpu_local_irq_pending()` only selects Capstone H-interrupts that are
    enabled in `env->mie`.
  - `rmw_mie64()` reruns `riscv_cpu_check_interrupts()` after `env->mie` changes,
    so already-pending H-interrupts become deliverable when software reenables them.
- The split `null_blk` package also keeps two safer changes found during the
  investigation:
  - `metadata_region` is borrowed and revoked per domain call instead of being
    permanently shared at module load.
  - `null_validate_conf()` copies back only validated scalar configuration fields.
- Temporary Linux printk probes, forced `synchronize_rcu()`, QEMU trace events,
  and OpenSBI region-dump ecall exposure were removed before final verification.

## 4. Verification & Next Actions

- Focused checks passed after rebuilding QEMU and the Buildroot kernel/nullblk
  image from cleaned sources:
  ```bash
  bash capstone/tests/runtime-qemu/run-nullblk-split-rmmod.sh
  bash capstone/tests/runtime-qemu/run-nullblk-split-io.sh
  bash capstone/tests/runtime-qemu/run-nullblk-baseline.sh
  ```
- Broader regression also passed:
  ```bash
  "$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone
  "$CAPSTONE_LLVM_LIT" -sv clang/test/CodeGen/capstone-builtins.c clang/test/CodeGen/builtins-capstone.c
  "$CAPSTONE_LLVM_LIT" -sv lld/test/ELF/emulation-capstone.s
  "$CAPSTONE_LLVM_LIT" -sv clang/test/Driver/capstone-linux-toolchain.c
  bash capstone/tests/runtime-qemu/run-coremark.sh
  bash capstone/tests/runtime-qemu/run-shared-region-probe.sh
  bash capstone/tests/runtime-qemu/run-hostcall-stdout-probe.sh
  bash capstone/tests/runtime-qemu/run-hostcall-filewrite-probe.sh
  bash capstone/tests/runtime-qemu/run-hostcall-fileread-probe.sh
  bash capstone/tests/runtime-qemu/run-hostcall-file-open-close-probe.sh
  bash capstone/tests/runtime-qemu/run-hostcall-file-handle-write-probe.sh
  bash capstone/tests/runtime-qemu/run-hostcall-file-handle-read-probe.sh
  bash capstone/tests/runtime-qemu/run-hostcall-file-handle-sync-probe.sh
  bash capstone/tests/runtime-qemu/run-hostcall-file-handle-stat-probe.sh
  bash capstone/tests/runtime-qemu/run-hostcall-file-handle-truncate-probe.sh
  bash capstone/tests/runtime-qemu/run-hostcall-path-access-probe.sh
  bash capstone/tests/runtime-qemu/run-hostcall-path-delete-probe.sh
  bash capstone/tests/runtime-qemu/run-hostcall-combined-file-object-probe.sh
  bash capstone/benchmarks/beebs/run-beebs-fac.sh
  bash capstone/benchmarks/beebs/run-beebs-insertsort.sh
  bash capstone/benchmarks/beebs/run-beebs-fibcall.sh
  bash capstone/benchmarks/beebs/run-beebs-cnt.sh
  bash capstone/benchmarks/beebs/run-beebs-bubblesort.sh
  bash capstone/benchmarks/beebs/run-beebs-prime.sh
  bash capstone/benchmarks/beebs/run-beebs-recursion.sh
  e2fsck -fn "$CAPSTONE_BUILDROOT_DIR/build/images/rootfs.ext2"
  ```
- `run-static-cap-globals-probe.sh` was attempted as an extra diagnostic, but its
  expected-failure case was stopped after taking much longer than the positive
  cases. Do not treat that optional diagnostic as part of this blocker's verified
  baseline.
- Next work should return to the benchmark plan: add one more small BEEBS wrapper,
  with `janne_complex` as the recommended first candidate.
