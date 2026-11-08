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

## SPEC measurement: method and pre-registered prediction (written before the runs)

**Arms.** One binary per benchmark, built with the native-IO SDK: the tree-board build's own compile and
link lines (capstone-lifetimes arm sources, the sources today's kits were built from), with that SDK's
`capstone-cc` in place of sdk-lifetimes'. perlbench `b42a89c7d06c3db1`, cpugcc `5fd1d99966f9ae77`.
- **before = exchange:** `capstone_vm.native_io=0`, every native route off. The SDK sees no v3 feature and
  sends every call through the exchange buffer. This is the native branch's own exchange route.
- **after = direct:** `native_io=1` plus `native_{locks,stat,seek,fd,close,path,clock}=1`, the
  collaborator's mask 479.
- Both arms run with `x-capstone-checked-copy=false` and `capstone_vm.checked_copy_emu=1`, so every check
  traps to the emulating firmware `e03268f9c8e64620`.
- Everything else is identical: one supervisor slot, transport on, `dom_null=1`, `dom_local_tmp=1`, 2 GiB,
  icount 0.

**Today's launcher route** (dev's adapter and capstone-vexec, not this branch) is kfast's `k0-perl` /
`k0-gtr`. It is a reference row only: a different kernel, adapter, binary and SDK.

**Stages.**
- perl test = kfast's perl stage inputs: makerand.pl, then test.pl driven by the plain RV64 perl, which
  execs the native perl for each test file.
- gcc train01 = kfast's gtr stage input.
- Runner and analyzer: `tests/native-task/cscheck-emu/run-native-stage.py` and `analyze.py`.

**Instruments.**
- capmix3 (shift 24) totals between HINT markers before and after each command, split into firmware
  ([0x8000_0000, 0x8100_0000)), kernel and user.
- `native_counters` deltas: escapes by kind, kind-3 services by route, emulation entries and causes.
- Definitions: exits = all CSRUNV escapes. Delegated calls = kind-3 services of the exchange and register
  routes. Instructions saved per delegated call = (total_before - total_after) / delegated calls.
- Machine-wide totals include the plain perl driver and Linux work, which are the same in both arms.

**Prediction, perl test (both commands summed):**
- rc 0 in both arms, specdiff 2/2 in both. Output hashes equal across the arms and equal to the launcher
  run's `test.out`/`makerand.out` hashes.
- Delegated calls 90k-120k, against kfast k0's 105,239 launcher-served requests. Equal across the arms
  within 1 %.
- direct arm: 55-80 % of calls on the register route. exchange arm: register = 0.
- Exits (all kinds) equal across the arms within 3 %.
- Emulation entries > 0 in both arms: 3-8 per delegated call in exchange, 1-4 in direct.
- Firmware instructions per emulated check: 300-1,200.
- **Saving: 4,000-10,000 instructions per delegated call**, 0.3-1.5 % of the run's total. Basis: the QEMU
  syscall table's 15.9k exchange versus 4.2k specialised read and 9.6k generic dispatcher, plus the
  emulation difference.

**Prediction, gcc train01:**
- rc 0 and specdiff 1/1 in both arms; the `.s` hash equal across the arms.
- About 150 delegated calls (kfast k0-gtr's histogram: 122 of them writev, which has no direct route). Fewer
  than 30 on the register route.
- Saving per call is not resolvable from the total. Saving under 0.01 % of the run.

## The pre-registered perl stage was void; the replacement stage, registered before its runs

`perl-exch-off`, `perl-dir-off` and `perl-dir-on` (the stage registered in b0dc39bd66f3) ran makerand.pl
correctly. test.pl, though, reported `Failed 317 tests out of 0`, and the adapter's counters did not move
during that command (`@@CNT 1 begin` = `@@CNT 1 end`, k3 125 in both). No native perl was ever started.
Cause: a native exec's grant root is its cwd (`resource-dom.h` `dom_resources_init`, `get_fs_pwd`). t/TEST
`chdir 't'` and execs `../../sw/perl` from `t/` with `-I../lib`. That cwd has no `.capstone-tmp`, so the
`dom_local_tmp` grant refuses the exec, and `../lib` would lie outside the grant anyway. The plain-perl driver
of today's board stage therefore cannot run on the native backend. These three runs are void and are not
used. The gcc runs of that registration are unaffected.

**Replacement perl stage** (`run-native-stage.py`, perl part, the collaborator's qualified Perl recipe):
- SPEC's own two invocations by the native perl itself, run from the staged directory as `./perlbench_r`.
- test.pl's 317 children start through native spawn and the pinned native hush shell (BusyBox 1.37.0 port,
  this branch, built with the native-IO SDK). The shell is exposed as `/bin/sh` through `dom_shell`.
- Plus `dom_null=1`, `dom_local_tmp=1`.
- `supervisor_slots=1`, the default, is no longer spelled out, because the kernel keeps 511 command-line
  bytes. The runner now asserts that limit.
- The shell and the children's spawn and pipe services now run on the native backend, so the workload
  differs from kfast k0's.

**Prediction for the replacement stage** (perl test, both commands):
- rc 0 and specdiff 2/2 in both arms; `makerand.out` hash `539d02dc...` in both, as in the launcher run.
- Delegated calls 100k-200k (the driver now reads 317 child pipes, and the shell's calls count too).
  Equal across the arms within 1 %.
- Exits equal across the arms within 3 %.
- direct arm: 40-80 % of calls on the register route.
- Saving 4,000-10,000 instructions per register-routed call. Per delegated call 2,000-8,000.
- Emulation entries > 0 in both arms; firmware instructions per check 300-1,200.

**gcc train01 results** (registered in b0dc39bd66f3):

| | exchange | direct |
|---|---|---|
| rc, specdiff | 0, 1/1 | 0, 1/1 |
| `.s` sha256 | `c786c07f51d56232` | `c786c07f51d56232` (= launcher run k0-gtr) |
| total instructions | 17,495,899,853 | 17,495,878,377 |
| user / kernel / firmware | 14,834,266,116 / 2,656,942,873 / 4,690,864 | 14,834,248,818 / 2,657,000,213 / 4,629,346 |
| exits (k1, k3, k4, k5) | 135,259 (698, 46,608, 87,902, 51) | 135,258 (698, 46,608, 87,901, 51) |
| delegated calls (register / exchange) | 156 (0 / 156) | 156 (11 / 145) |
| other kind-3 services (map, nodes, ...) | 46,452 | 46,452 |
| emulation entries (all ok) | 4,271 | 4,172 |

The direct ABI saves 21,476 instructions in the run, 0.0001 %, about 2k per moved call, and only 11 calls
move. gcc's exits are faults (87.9k) and mapping services (46.5k). Only 156 of its 135k exits are delegated
calls, and 122 of those are writev, which has no direct route. As predicted, the saving is not resolvable
per call from the total. Firmware per emulated check is at most 1,098-1,110 instructions; the bucket also
holds timer SBI calls.

## perl test results (replacement stage, registered in d0703d0f27d5)

Both commands summed. Model off, emulating firmware `e03268f9c8e64620`, Image `28e5cc224b5bd776`, QEMU
`8f3f9118b1f57791`, perlbench `b42a89c7d06c3db1`, hush `tmp/hush-native` (native-IO SDK).

| | exchange (before) | direct (after) | difference |
|---|---|---|---|
| rc (makerand, test) | 0, 0 | 0, 0 | |
| specdiff | 2/2 | 2/2 | |
| makerand.out / test.out sha256 | `539d02dc10fabf70` / `b41ee90a9ea04241` | the same | (= launcher run k0-perl) |
| total instructions | 139,201,559,100 | 136,802,276,650 | **-2,399,282,450 (-1.72 %)** |
| user / kernel / firmware | 107,300,558,834 / 30,272,534,466 / 1,628,465,800 | 107,034,380,984 / 28,940,168,092 / 827,727,574 | -266,177,850 / -1,332,366,374 / -800,738,226 |
| exits (all kinds) | 720,613 | 691,554 | -29,059 (-4.0 %) |
| of which kind 1 / 3 / 4 / 5 (test.pl) | 11,914 / 315,368 / 389,397 / 2,741 | 11,895 / 286,136 / 389,593 / 2,741 | |
| delegated calls (register + exchange) | 159,288 (0 + 159,288) | 130,054 (102,809 + 27,245) | -29,234 |
| other kind-3 services (map, nodes, spawn plumbing) | 156,205 | 156,205 | 0 |
| emulated checks (all rd = 0) | 2,643,137 (16.6 per call) | 1,349,015 (10.4 per call) | -1,294,122 |
| guest seconds (icount clock), test.pl | 134.6 | 132.2 | |

**Instructions saved per delegated call:**
- 15,063 per call of the exchange arm (2,399,282,450 / 159,288);
- 18,448 per call of the direct arm;
- 23,337 per register-routed call.

These figures include 0.80e9 instructions of firmware emulation that the direct arm avoids by issuing fewer
checks. They are measurement-only: the emulation reads the mutable +64+16r slot and the frame-word epoch.

**Against the prediction** (d0703d0f27d5):
- Held:
  - rc, specdiff and hashes;
  - calls 100k-200k;
  - register share 79.1 % (inside 40-80 %);
  - emulation entries > 0 in both arms;
  - firmware per check about 613 (inside 300-1,200).
- **Failed:**
  - calls equal within 1 %: the direct arm makes 18.4 % fewer delegated calls;
  - exits equal within 3 %: 4.0 % fewer;
  - saving per register call 4k-10k: 23.3k;
  - saving per delegated call 2k-8k: 15.1k-18.4k.
- The direct ABI therefore removes calls and exits as well as making each call cheaper. Which calls
  disappear is UNRESOLVED: the runs have no route trace. Chunking of large transfers through the exchange
  buffer, against the direct ABI's 64 KiB single call, is a hypothesis that has not been tested.

**Exits per run are dominated by work the direct ABI does not touch.** Of 691,554 direct-arm exits, 389,593
are faults (kind 4, mostly resolved in cv_step_locked) and 156,205 are mapping and other services. Only
130,054 are delegated calls. Today's launcher route on the same test (kfast `k0-perl.analysis`, a
different kernel, adapter, binary and test.pl driver) shows 146,020 escapes for test.pl: kind 1/3/4 =
11,478 / 112,149 / 22,393.

**Positive control at SPEC scale and the emulation's cost.** The direct arm with `x-capstone-checked-copy=true`
(`p2-dir-on`) gives:
- rc 0, specdiff 2/2, `test.out` `b41ee90a9ea04241`;
- **emulation entries 0**;
- firmware 13,432,504 instructions (timer SBI work), against 827,727,574 with the model off;
- the same calls (130,054, 102,809 register) and exits within 31 (691,585).

Total 135,986,852,905 against 136,802,276,650 with the model off, so one emulated check costs **604
instructions** machine-wide (603.6 of them in firmware). A trap round trip on silicon costs cycles beyond
that. The exchange arm with the model on was not run. Scaling its 2,643,137 checks by 604 suggests about
1.6e9 instructions of saving with in-hardware checks, about 10k per exchange-arm call. That figure is an
estimate, not a reading.
