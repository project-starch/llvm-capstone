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

## RTL frame-layout review (rtl lane, 8 Nov, read against capstone-ariane c5a-wt; 1fc2cd5aa is its ancestor)

- **Offsets agree.** satp at +0 and srevroot (CSR 0x5C1) at +8 (switcher `runv_load_advance`), PCC at +64, x1..x31
  at +80 with stride 16 (+64+16r), so refusing slot 0 is right. +2336/+2344 is free on silicon: the caller snapshot
  ends at +1983 and the caller FP block (+2048, stride 8) at +2311; nothing writes +2312..+4095. Bounds, perms
  (hi[33:30], R = bit 2, W = bit 1) and id (hi[63:34]) field positions match.
- **Difference 1, type numbering.** The RTL stores its own enum in the saved word: NOT_CAP 0, LINEAR 1, NONLIN 2
  (ariane_pkg.sv:662-671; the tag is cap_type != NOT_CAP, ex_stage.sv:966). QEMU stores LIN 0, NONLIN 1 (cap.h:29-31).
  A C.5a port needs CAP_TYPE_NONLIN = 2, or every check refuses with 26. Any module code that decodes the type field
  of a saved slot reads numbers one higher on silicon. (kfast, perf/kernel-fastpath, reads only integer frame words:
  the escape kind and the registers, not type fields.)
- **Difference 2, the lifetime table.** C.5a refuses paged tables (capstone_rev_node.anvil:298-301): only a flat
  RECYCLE table, idx = id[17:0], gen = id[29:18]. The emulation's 4:9:9:8 walk would return 25 on every check
  there. A C.5a check is one header read plus one record read with a generation compare, cheaper than the 604
  instructions measured here. The capability-mode monitor's minting and the trap round trip are unmeasured.
- C.5a's escape writes real tags for the slots and no bitmap, so a port could read the tag by an LDC through a
  minted frame capability instead of the +2344 bitmap.
- Steps 2-4 are on HOLD (runtime and rtl lanes agree): 1.72 % on perl and nothing on gcc, before any silicon cost.
  A hardware check cuts the exchange arm (16.6 checks per call) more than the direct arm (10.4), so the 15.1 k
  saving per call would shrink to about 11-12 k. The kfast module's board number is the input to reopen on.

## C.5a route, probe 1: pinned arenas, no transport (branch native/c5a-route)

On C.5a the adapter's admission fails three ways (the RTL lane, above): transport, the paged table, and the
copyable profile. Also, the unpinned backing it requires uses CPTAG, CLDRAW and CSTRAW
(`linux-6.1-unpinned.patch:202-248`, `capstone_vm.c:1031/1308`), which the boards' RTL does not decode.
CLDRAW/CSTRAW cannot be emulated one instruction at a time either. The monitor saves S's registers as 8-byte
integers (`capstone_csmint.S`: x[n] at a0 + 8n), so opaque cargo can never return to an S register.
**Unpinned backing as written cannot work on C.5a.** The route survives only with pinned arenas.

`capstone_vm.c5a_model=1` (MEASUREMENT ONLY, default off) admits no transport and `unpinned=0`.
`run-musl-io.py --c5a-model` runs the model with raw moves and transport off. Result: QEMU d2bac4e08a21,
checked-copy ON, one slot, Image `b28fc66c17704252`, `gates/c5a-probe-on`:
- `MUSL_IO checks=170/176 failures=0`, `native integration checks=51 failures=0`,
  `gcc coverage checks=20 failures=0`: single-task native I/O works pinned.
- Then `entry-driver` printed `PASS first native task ready` and died on its 30 s alarm (`Alarm clock`,
  `MUSL_GATE_EXIT:142`) while starting a SECOND native task. **Correction (9 Nov):** this line first named
  `task-driver`. The string is entry-driver's, and task-driver's binary does not contain it; the kernel-port
  lane found this (native/board-6.4 b05aa17aa521). Both drivers start a second live native task, so the
  finding is unchanged. Without transport, `slot_acquire` never evicts
  (`capstone_vm.c:680`), and a new context waits for the slot (`:2953-2963`), so two live native tasks
  deadlock on one slot.
- **On C.5a's model only one native task can be live at a time.** The collaborator's perl recipe has three
  live at once (test.pl in the native perl, the hush shell, the test child), so it cannot run there.
  gcc train01 (one task) can. Perl test can run only with today's board structure (a plain driver that
  execs one native perl at a time). That structure needs the exec grant root to be wider than the cwd
  (`resource-dom.h` `dom_resources_init`), which is a change to the grant model.

## C.5a route, probe 2: the flat table on QEMU, and where the work stops

**Additions, MEASUREMENT ONLY and default off:**
- `c5a_model=1` accepts no copyable-profile node (build id width, compressed bounds).
- With `make CV_FLAT_INDEX_BITS=16|18` it takes a flat RECYCLE table, allocated contiguously per context.
- `dom_grant_root=DIR` gives an exec whose cwd lies beneath DIR that directory as its grant root, keeping
  the cwd. This is for the board's one-native-task perl structure.

**gcc train01, direct arm, checked-copy ON, flat 16-bit table, no transport, pinned, one slot.** Run on
QEMU d2bac4e08a21 with Image `18cffd642b651c59` (`spec/out-c5a-gtr-dir-on`): rc 139 after 11.26e9
instructions, `capstone_task: fault rc=-14 kind=2 cause=30`, 0 emulation entries. The flat build never
collects (`collect()` returns at once on flat tables), and 65,536 ids do not hold gcc train01. The launcher route
completes the same input on the 18-bit twin (kfast k0-gtr). On QEMU, an 18-bit flat table exists only on
the C.5a twin (capstone-qemu virtual/flat-index-18 46b69c03a4 + the Q-34 pin). That twin has no CSCHECK,
transport or raw moves. Its legacy profile keeps CAPENTER, so it boots the capability-mode monitor. The
copyable QEMU (dd041/d2bac4e) has native CSCHECK but cannot boot that monitor (CAPENTER removed). **No single
QEMU has both native CSCHECK and C.5a's table.** The "check native, firmware entries 0" control therefore
reads only on the copyable QEMU's 16-bit table, which gcc train01 exhausts. The monitor emulation can be
validated only on the twin, with the model off.

**Not started, recorded for the next step:**
1. The CSCHECK emulation in the capability-mode monitor (csbi-batch d9da397, beside `csmint_dispatch`).
   Planned design:
   - Physical capabilities via `CSM_PHYS`, no Sv39 walk (rs1/rs2 are PAs).
   - Tag by LDC through a minted frame capability.
   - Type/bounds/perms by LCC selectors 1/3/4/5. The TYPE query agrees on RTL and QEMU (NONLIN reads 1,
     Q-35), so the stored-enum difference is avoided.
   - id = hi >> 34; flat liveness: header, then record root + 16 * (id & 0x3ffff), VALID && !FREE, word1
     generation [59:48] == id >> 18.
   - Software epoch: the kernel writes +2336 after each escape when the model publishes none.
   - The classification word does not exist on C.5a, so the direct route needs an SBI "kinds" call per
     service (8 LDCs in M). That cost lands only on the direct arm.
2. The QEMU monitor build for the twin (kit firmware `fw_jump.elf` came from an opensbi-T tree).
3. **The board kernel.** The board kit's Image is Linux 6.4.14 from the shared caplifive-system bake
   (`build-fpga/.config`). The native route needs CONFIG_RISCV_CAPSTONE_TASK and the native/helper patches,
   written for 6.1. "Swap only module/launcher/firmware" cannot hold: the kernel changes, and either the
   patches are ported to 6.4.14 or a 6.1 FPGA kernel is built. Neither has ever booted on the board.

## The C.5a / E5 path (lead's decision: native goes to silicon, pinned arenas, rd-only slot detach/adopt)

**Adapter** (`c5a_model=1`, MEASUREMENT ONLY):
- CSRUNV 4/5 are the RTL's rd-only slot detach/adopt (rd 0 or -1). No image is written or read: no magic
  at +1024, no epoch at +1072, and import writes satp to +928 (the switcher's saved satp) instead of the
  image's +1192. A resident context needs no ASID rebind: +928 is refreshed before every resume.
  `native_slot_transport=1` enables it.
- Flat RECYCLE table, `make CV_FLAT_INDEX_BITS=20`. 2^20 records = 16 MiB come from CMA (boot with
  `cma=`); the buddy allocator is used below its largest order.
- The copyable-v1 node is read, holding truthful values: id-bits 30, exact-bounds 0, page 4096,
  granule 16, checked-copy-abi 0, context-transport-version 0. `checked_copy_emu=1` admits abi 0.
- The kernel writes the epoch at +2336 after every escape. The classification word (+2344) is fetched
  on demand, once per escape, by CSKINDS rd, frame_pa (custom-2 funct7 0x27, MEASUREMENT ONLY). The
  monitor emulates it with an LDC of each slot. Only the direct route and native spawn/pipe read it, so
  its cost lands on the direct arm only (`kinds_queries` counts it).
- Firmware entries: the monitor stamps descriptor word 5 with `0x4353434845` on every check it handles.
  The adapter counts the stamps (`emu_fingerprints`); a processor-performed check never writes word 5.

**Monitor** (capstone-sbi `native/cscheck-emu`, `capstone_cscheck.S`, `-DCAPSTONE_CSCHECK_EMU`): the hook sits
before the CSMINT one in `_handle_non_ecall`. Physical capabilities come from CAPCREATE per page (as CSM_PHYS).
- The stored type is read before any load: `CSC_STORED_NONLIN` 2 on the RTL, 1 in QEMU (Q-35). The check
  never LDCs a slot that might be LINEAR.
- Tag by LDC, then LCC TYPE (7 = untagged; NONLIN reads 1 on both machines).
- Flat liveness: header RECYCLE set and PAGED clear; 2 <= index < next and < capacity; record word 1
  VALID and not FREE; generation [59:50] == id >> 20.
- LCC for perm, cursor, base and end; bounds; Sv39.
- **QEMU does not decode CAPCREATE, so this file cannot run on any QEMU.** It assembles (291 instructions),
  but it is unexecuted until an RTL-simulation arm runs it, as CSMINT's emulation was validated
  (va-csmint-emu).

**QEMU twin** (capstone-qemu `native/c5a-twin20-check`, on `virtual/flat-index-20` 23f9df04e5):
`x-capstone-cscheck-frame=true` models the monitor's rules (frame-word epoch, mutable slot, stored-type
pre-check, flat liveness), so the kernel path can be gated on the twin. With it, CSCHECK never reaches
firmware. The detach/adopt CSRUNV 4/5 come from the rtl lane, not yet delivered.

**Working check, before registration:** gcc train01 direct on twin20.
- Setup: Image `d7712c4127479056` (flat 20), twin20 + model, the QEMU kit's monitor `fw_jump.elf`, the C.5a
  dtb, 2 GiB with cma=256M, one slot, no transport.
- rc 0, specdiff 1/1, total 16,591,075,992 instructions.
- 156 calls (11 register, 145 exchange), kinds_queries 11, emu_fingerprints 0.

lbm test direct: rc 0, specdiff 1/1, 332 calls (327 register).

### Pre-registered prediction for the twin20 runs (written before them)

**Setup:** the binaries above (`nat-*` builds of the capstone-lifetimes sources, native-IO SDK);
exchange = native_io 0, direct = mask 479; check model on (QEMU, no firmware).

**Predicted for every run:**
- rc 0 and specdiff pass in both arms; output hashes equal across the arms.
- emu_fingerprints 0 (the positive control: the check never reaches firmware). kinds_queries 0 in
  exchange and <= register calls in direct.
- Exits equal across the arms within 5 % (perl's 4 % drop was call-count driven).

**Saving per run:**
- gcc train01: under 0.01 %.
- lbm test: 0.02-0.2 %.
- nab test: under 0.1 %.
- xz test: 0.1-2 %.
- x264 test: 0.5-3 % (most write calls; its in-guest image validation is not run, so its specdiff is
  expected to be unavailable).

**Ranking by saving:** x264 > xz > lbm, nab > gcc.

## twin20 results: exchange vs direct at test size, the two-live-task gate, and the native train stages

**Exchange vs direct** (registered in f93d97008f2c). Setup: twin20 + `x-capstone-cscheck-frame`, Image
`d7712c4127479056`, one slot, no transport. Every run: rc 0, specdiff pass in both arms, emu_fingerprints 0,
kinds_queries = 0 in exchange and equal to the register calls in direct.

| bench | exchange instr | direct instr | saving | exits | calls (register) |
|---|---|---|---|---|---|
| gcc train01 | 16,588,854,663 | 16,590,963,109 | -2.1e6 (-0.013 %) | 135,206 / 135,206 | 156 (11) |
| lbm test | 7,960,542,674 | 7,954,328,287 | 6.2e6 (0.078 %) | 4,056 / 4,049 | 336 -> 332 (327) |
| nab test | 7,970,132,919 | 7,969,948,918 | 0.18e6 (0.002 %) | 2,132 / 2,126 | 34 -> 30 (15) |
| xz test (12 cmds) | 61,749,225,882 | 61,749,768,863 | -0.5e6 (-0.001 %) | 48,111 / 48,111 | 216 (36) |

**Against the prediction:**
- Held: lbm (0.02-0.2 %), nab (< 0.1 %), equal exits, the fingerprint control, kinds_queries.
- Failed: xz (predicted 0.1-2 %, read ~0).
- gcc: |-0.013 %| is just outside the "< 0.01 %" bound. Its negative sign, with only 11 register calls,
  puts the run-to-run noise at about +-2e6 instructions, so nab and xz are also inside the noise.
- x264 was not measured. The first attempt ran the generator as the native ldecod, which faults with cause
  28 where today's stage runs a plain ldecod. The rerun was stopped after 85 min to yield the shared lock.

**At test size no candidate has a measurable native saving.** Delegated calls are 30-340 per run.

**Two live native tasks on twin20 with the rtl lane's detach/adopt** (local twin 5de8f6b012 cherry-picked on
the check model, Image `0ee07f455a215c6e` with the adopt-kind gate). `run-musl-io.py io --c5a-model --twin`
PASSes:
- 170/176 musl-IO and 51 integration checks, `native entry failures=0`, `native task failures=0`;
- `detaches=2 adopts=2 adopt_kind_refused=0`;
- no twin `ADOPT-KIND-MISMATCH` or `ADOPT-UNPARKED` line, which the runner now asserts.

Probe 1's deadlock is gone.

**Native train stages** (`make-native-stage.py`: a cp -rL copy of the stages-train-fair stage with the
launcher replaced by a direct exec):
- **perlbench:** the same udcnofp objects as the frozen train image (perf/heapx/objs-udcnofp), relinked with
  the native-IO SDK built by the same compiler (llvm-c1): `ef38a768580f370e`.
- **mcf:** the tree-boarddevmk sources replayed with that SDK: `d5e5dc49eec9be7c` (stripped as the frozen
  ones).
- The train commands are the stage's own, each a single native perl or mcf process. Unlike test.pl, train
  needs no native shell or spawn, so the command structure matches Plain's.
- The stage runs on a private tmpfs, because the native grant admits tmpfs, ext4 and 9p only.
- Short twin: perl `perfect.pl b 3`, direct, with transport. rc 0, specdiff 1/1, 19.6e9 instructions,
  187 calls (132 register), no ADOPT lines.

**Table size per stage.** The compiler lane's high-water runs: mcf train peaks at 221,072 pages and 14,065 live
nodes, and was SIGKILLed with a 2^20 table on kit-f55. perlbench needs about 415 k live nodes. Each native
stage therefore carries its own boot parameters (`cscheck-emu/stages/<stage>.params`):
- mcf: `capstone_vm.node_flat_records=262144` (2^18, 4 MiB);
- perlbench: `capstone_vm.node_flat_records=524288` (2^19, 8 MiB).

The native route sizes its table per boot (module parameter, read at load), not per run within a boot.
The flat table now comes from CMA first, with the buddy allocator as fallback: a multi-MiB buddy block can
be missing on a fragmented 936 MiB guest.

The twins run at the board's 936 MiB with a device tree dumped at that size
(`cscheck-emu/twin20-936-c5a.dts`). A dtb dumped at 2 GiB on a 1 GiB guest faults the QEMU monitor at boot
(cause 24 at 0xffffffc6feffe000).
