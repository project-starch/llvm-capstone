# Native checked access, steps 0 and 1: the moved epoch words and the firmware-emulated check (QEMU)

Branch `native/checked-emu-measure`, based on `native/linux-task-integration` 04350015dcad. QEMU is
capstone-qemu `native/checked-emu-measure`, based on dd041bac5ad9 (the native branch's pin). The plan is
`docs/plans/2026-11-08-native-checked-access-rtl-order.md` on `land/rtl-native-plan` 99f164d097cc, its
"Addendum (8 Nov)". Steps 2-4 (RTL) are gated on the number this note records.

## Question

How much does the direct register syscall ABI save over the exchange route, in exits per run and in
instructions per delegated call, on SPEC 500.perlbench_r test and 502.gcc_r train01, with CSCHECKR/W
emulated by firmware (the boards' C.5a bitstream lacks them)?

## Platform actually built (deviations from the collaborator's platform are listed)

| Part | What | sha256 (16) |
|---|---|---|
| QEMU | capstone-qemu native/checked-emu-measure (step 0 below), `--enable-plugins` | `8f3f9118b1f57791` (step 0) |
| Kernel | upstream Linux 6.1 `830b3c68c1fb` + `linux-6.1-unpinned.patch`, `kernel-native-mprotect.patch`, `kernel-native-gcc.patch`, then `install-native-task.sh` | `fff43af57f429016` (step 0 Image) |
| Kernel config | **deviation:** the collaborator's platform-dev4 config (`9c724799...`) is not on this host (746 local Images scanned with `extract-ikconfig`); the base is the QEMU kit's embedded config (`3849714d...`) with KSM and THP off, then the install script's own changes (SMP off, CONFIG_RISCV_CAPSTONE_TASK) | `.config` `2d6cf88134682026` |
| Firmware | QEMU's bundled OpenSBI fw_dynamic (the native platform's firmware, the same hash as in `rebase-qualification.json`) | `2335a28682ddf65e` |
| rootfs | **deviation:** the QEMU kit's rootfs (the collaborator's `62049800...` is not here) | `9903242c37a72425` |
| SDKs | this branch's `build-sdk.sh`, compiler `cc-prmerged` (the compiler that built today's SPEC kits); exchange SDK and `-DCAPSTONE_MUSL_NATIVE_IO=ON`, identical libc archive | runtime `9469fe943bc35d68` / `0fb401d92738e35c`, libc `68c9c5bd22249eee` |

## Step 0: the epoch and bitmap words move to +2336/+2344, not +2320/+2328

The escape published the epoch at +656 and the tag/raw-tag bitmaps at +664. On C.5a, +656..+912 is the
context's FP save area. The plan proposes +2320/+2328 as an example, from the RTL's free range
+2320..+4095.

**+2320/+2328 is not free in the model's v5/v6 transport image.** The image's FP policy is twelve u64s
from `NATIVE_FP_STATUS` = +2240, so it covers +2240..+2335 (capstone-qemu `capstone_supervisor.c`,
`FP_STATUS_FIELDS`). An export (CSRUNV action 4) writes `rebias_overflow` over +2320 and
`rebias_underflow` over +2328. The next import (action 5) compares the image's epoch copy at +1072 with
the frame word and refuses. The native adapter requires `native_slot_transport=1` and one hardware slot,
so exports are on the normal path, for example perl's 317 child processes. The words go to +2336/+2344
instead, the gap before the immutable escape GPRs at +2400. That gap is free in both layouts. The frame
CSRUNV requires grows from 672 bytes to 2352.

Evidence (capstone-qemu `tests/capstone-checked-copy/run.py`, the 35 ISA cases, run under the shared lock):

| QEMU | epoch/kinds | result |
|---|---|---|
| d2bac4e08a21, `8f3f9118b1f57791` | +2336/+2344 | 35 PASS, `checked-copy ISA failures=0` |
| same source at 2320/2328 (negative control, build assert disabled), `b6ee5dc00bb393b4` | +2320/+2328 | `FAIL transport`, `FAIL transport_forged_frame`, `FAIL transport_retired`, `checked-copy ISA failures=3` |

The new build assert (`QEMU_BUILD_BUG_ON`, FP policy end <= epoch word) was shown to fire with the
offsets at 2320: `static assertion failed: "not expecting: NATIVE_FP_STATUS + 8 * (...) > CHECK_EPOCH_OFFSET"`.

Kernel side: `CV_FRAME_CHECK_EPOCH` (2336/8) and `CV_FRAME_CHECK_KINDS` (2344/8) replace every
`frame[82]` and `frame[83]`: 18 readers in `checked-copy-dom.h`, `native-process-dom.h` and
`native-syscall-dom.h`. No other module code reads +656..+671. `clear_thread_frame`'s
`memset(frame, 0, 656)` clears the tagged slots and is left as is.

The native path's own gate on the step-0 platform, property ON (`run-musl-io.py io`, one hardware slot):
`MUSL_IO checks=170 failures=0` (exchange), `checks=176 failures=0` (direct), both `checks=4 failures=0`,
`native integration checks=51 failures=0`, `native entry failures=0`, `native task failures=0`, and the
verdict `PASS musl IO, scalar shape, route boundaries, cached identity, authority, entry and task
lifecycle`. Control: the same kernel, which reads +2336, on the QEMU that publishes at +2320 fails at the first
service, `capstone_task: fault rc=-116 kind=3 cause=11` (ESTALE) and `MUSL_GATE_EXIT:139`. So the gate
does test that the two offsets agree.

**Second collision, recorded and left as it is:** the transport image's epoch copy (+1072) and caller
privilege (+1080) lie inside the RTL's caller snapshot (+1024..+1976). They matter for export/import
images, not for the direct ABI.

## Step 1: CSCHECKR/W emulated by firmware (measurement only)

The native platform boots QEMU's bundled OpenSBI (057eb10b6d52), not the Capstone monitor, and its M mode is
not in capability mode. The emulation therefore goes into that OpenSBI, as
`tests/native-task/cscheck-emu/opensbi-057eb10b-cscheck-emu.patch`, built by `build-firmware.sh` in the same
folder (the build reproduces `fw_dynamic.bin` `e03268f9c8e64620` bit for bit). It follows the CSMINT
emulation's pattern: decode mtval on the illegal-instruction trap, emulate, write rd, step mepc. The trap is
custom-2 (0x5b), funct3 1, funct7 0x25/0x26, taken from S; anything else on custom-2 stays illegal. M reads
everything physically: frame, descriptor and lifetime table.

The checks follow capstone-qemu `helper_cscheck` in its refusal order:
1. U or V caller, a descriptor or frame not aligned RAM outside the firmware: 29.
2. Zero epoch, an epoch other than the frame word at +2336, slot outside 1..31, reserved word non-zero,
   satp other than frame+0, or srevroot other than frame+8: 29.
3. Untagged at the escape (bit `slot` of +2344): 24.
4. Lifetime record not VALID or FREE (paged table at the current srevroot, 3-level 4:9:9:8 walk, one 64-bit
   read of the flags): 25.
5. Type other than NONLIN: 26.
6. Read or write permission missing: 27.
7. Offset overflow or bounds, or an address outside Sv39: 28.
8. Otherwise descriptor[4] = cursor + offset and rd = 0.

The bounds decode is `cap_compress.c` `decode()` for 30-bit ids.

**Measurement-only shortcuts, to be stated wherever the number is cited:**
1. The capability is the MUTABLE frame slot +64+16r (the compressed form the escape saved), and its tag is
   the escape's classification bit at +2344. C.5a has no hidden escaped snapshot, so trusted frame rewrites
   are not excluded.
2. The epoch is the frame RAM word at +2336, not a hart register. On this QEMU the escape publishes it,
   because the context transport is on. On C.5a the kernel would have to write it. The emulation cannot see
   paused/terminal state or resume invalidation.

The satp/srevroot comparisons use the kernel-written entry words +0/+8. The RAM check is the virt DRAM
window [0x8000_0000, +64 GiB) with the firmware image excluded.

The firmware also adds an experimental SBI extension 0x08435343: fid 0 probe, fid 1 counter, fid 2 reset.
It counts entries, rd = 0, and each cause 24..29. Kernel side, measurement only, default off:
- `capstone_vm.checked_copy_emu=1` admits a model whose device tree says `checked-copy-abi` 0 and transport
  v5, provided the firmware answers the probe. It prints `MEASUREMENT ONLY: CSCHECKR/W emulated by firmware`.
- `/sys/module/capstone_vm/parameters/native_counters` reads CSRUNV escapes by kind, counted in
  cv_step_locked's loop so faults resolved in place are included. It also reads kind-3 services by route
  (register, exchange, other) and the firmware's counters.
- `run-musl-io.py` gains `--checked-copy-model off` and `--counters`.

The native path's gate (`run-musl-io.py io`, one slot) on Image `28e5cc224b5bd776`, firmware `e03268f9c8e64620`:

| Model | Firmware | Result | Emulation counters at the end |
|---|---|---|---|
| `x-capstone-checked-copy=true` (positive control) | emulating | PASS; 170/176 musl-IO and 51 integration checks, 0 failures | entries 0, ok 0, every cause 0 |
| `x-capstone-checked-copy=false` | emulating | PASS; the same checks, 0 failures | entries 11,509, ok 11,492, c25 1, c28 16, others 0 |
| `x-capstone-checked-copy=false` (control) | base OpenSBI `2493e91d13d61600`, same source unpatched | FAIL: adapter refuses to load, `.dom` not executable | n/a |

With the property on, the firmware is never entered. With it off, every check is emulated and the gate's
authority refusals still come back as the expected errors. Escapes and routes are the same in both arms
(end: k1 104, k3 767, k4 532/531, register 227, exchange 393, other 147). During the IO phase the gate
made 424 kind-3 services and 10,189 emulated checks, so one service can issue many checks: the copy
re-authenticates every chunk. Per-call check counts therefore belong in the SPEC readings.

The emulation is ready for the RTL lane's review against the RTL frame layout.
