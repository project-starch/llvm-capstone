# Current recommended next step

This file intentionally contains the **current** recommendation only.
It is expected to change over time and should be updated when the project state changes.

## Current recommendation

The previous runtime blocker has been removed, and the first real HostCall stdout proof is now validated. The next smallest meaningful step toward the real goal (running serious software such as SPEC-like tests, FFmpeg, sqlite, libpng, etc.) is now:

> keep the same working stdout request/response flow, but make the payload sharing rule stricter: the domain writes the payload, the helper only reads it after return, and the runtime revokes that borrowed access automatically when the round completes

In short: **the project now knows that a host-side service can be requested and serviced correctly. The next smallest milestone is to prove that the same flow also works under a tighter ownership rule that is closer to what real libc/OS boundary calls will need.**

## Same recommendation in simpler words

Right now the project has proven that:

- the domain can ask for one tiny host-side service,
- the helper can do that work,
- the domain can come back and verify the answer.

That is good news, but the current proof is still a little too permissive because
the payload buffer is broadly shared.

So the next step is **not** “add many more services” yet.
The next step is:

> keep the exact same stdout example, but make the payload buffer ownership tighter so the domain produces the bytes, the helper consumes them once, and the runtime revokes that borrowed access automatically after the round

Why this is the next step:

- it is small,
- it reuses the already working test,
- it removes one important source of future ABI sloppiness,
- and it moves the design closer to something that can scale to real host-side I/O.

## Same recommendation in more technical terms

The current validated HostCall v0 stdout probe uses two shared regions:

- metadata: `INOUT + SHARED`,
- payload: currently also `INOUT + SHARED`.

The metadata region should stay shared for now because both sides inspect and update
the protocol state across both `call_dom()` rounds.

The payload region is different:

- Round 1: the domain writes the bytes,
- after `DOM_RETURN(HC_V0_RET_PENDING)`, the helper only needs to read them,
- after the helper performs the write-like host service, that borrowed access should
  no longer remain broadly live.

So the technical next step is to re-run the same stdout proof with the payload moved
toward a directional borrowed rule (`OUT + BORROWED`) while keeping metadata shared.

## Why this is now the right next step

The previously active blocker was:

- the local image behaving like stock OpenSBI,
- host-side `SBI_EXT_CAPSTONE` calls returning `error=-2`,
- shared-region mutations not becoming visible,
- and split `null_blk` either crashing or failing to create the device.

That blocker is now gone.

The source-backed validated results are:

- `build/local.mk` is present again and points Buildroot at:
  - `components/linux`
  - `components/opensbi`
- `make build A=opensbi-rebuild` regenerated:
  - `components/opensbi/lib/sbi/sbi_capstone_dom.c.S`
  - `components/opensbi/lib/sbi/capstone_int_handler.c.S`
- the OpenSBI rebuild log shows `opensbi custom Syncing from source dir .../components/opensbi`
- the shared-region probe now reaches:
  - `word after call 1 = 0x1111111111111111`
  - `word after call 2 = 0x2222222222222222`
  - `success`
- baseline `null_blk` works,
- the first HostCall stdout proof now:
  - shares fixed-width metadata and payload regions,
  - returns `HC_V0_RET_PENDING` from the first `call_dom()` round,
  - lets the host helper print the payload in ordinary Linux userspace,
  - returns `HC_V0_RET_DONE` from the second `call_dom()` round,
  - and finishes with `phase = HC_V0_PHASE_DONE`,
- split `null_blk` now:
  - creates `/dev/nullb0`,
  - completes the marker-based I/O path,
  - and completes `rmmod null_blk` successfully.

So the current bottleneck is no longer the firmware/runtime bring-up path itself.
The next unknown is whether the same protocol still works once data ownership becomes stricter.

## Why this directly helps the end goal

The end goal is not just to print one string. The end goal is to support real
software that eventually needs host-side services such as file I/O or other OS-
boundary work.

That future only becomes practical if the project can prove three things in order:

1. the control-transfer protocol works at all,
2. the data-sharing rules are safe and disciplined enough to scale beyond a toy example,
3. additional services can be added without redesigning the ABI every time.

Step 1 is now proven.
The current next step is the smallest concrete move on Step 2.

## Key terms used in this recommendation

- **permissive**: broader sharing than strictly necessary; easier for bring-up, weaker for long-term discipline,
- **stricter**: narrower access and clearer ownership rules,
- **ownership discipline**: an explicit rule saying who writes, who reads, and when that access should end,
- **disciplined protocol**: a protocol whose state transitions and buffer usage are intentionally constrained and easy to audit,
- **payload becomes one-direction borrowed**: the payload is produced on one side, consumed on the other side, and is not meant to remain broadly shared after the round.

See also `capstone/agent-handoff/current/runtime-terms-glossary.md`.

## Plan from the current point to a serious workload

### In simple words

1. **Tighten the existing stdout proof**
   - same demo, fewer permissions.
2. **Add one or two more tiny host services**
   - enough to prove the ABI is not special-cased for one hardcoded string.
3. **Define a small stable host-service layer**
   - something file/I/O shaped, not “one protocol per libc symbol”.
4. **Build a tiny domain-side runtime/libc story around that layer**
   - keep simple helpers local, cross the boundary only for real OS work.
5. **Run one serious but still manageable target first**
   - `sqlite` is the most sensible first serious workload.
6. **Only then move to heavier programs**
   - `ffmpeg` and especially SPEC are later, because they stress more subsystems.

### More technical phased plan

#### Phase 0: keep the current baseline green

Required gates stay:

- `run-shared-region-probe.sh`,
- `run-hostcall-stdout-probe.sh`,
- `run-nullblk-baseline.sh`,
- `run-nullblk-split-io.sh`,
- `run-nullblk-split-rmmod.sh`.

#### Phase 1: tighten ownership on the existing stdout proof

Goal:

- keep metadata as `INOUT + SHARED`,
- move payload toward `OUT + BORROWED`,
- prove the same two-round flow still works.

Exit criterion:

- same wrapper still passes,
- no stale-read / revoke / capability fault regression appears.

#### Phase 2: prove that the ABI generalizes beyond one toy opcode

Goal:

- add one or two more coarse operations,
- keep the same metadata shape if possible,
- avoid symbol-per-libc growth.

Good candidates:

- small buffered write-like call,
- minimal file open/read/close-style service,
- tiny status/query operation.

Exit criterion:

- at least one second service works on the same ownership model,
- the ABI still looks like a reusable service protocol rather than a one-off demo.

#### Phase 3: define the minimal domain-side runtime/libc boundary

Goal:

- identify what stays local in the domain (`memcpy`, `strlen`, formatting helpers, etc.),
- identify what crosses to the host (file/device/stdout-like services),
- document the first stable service subset.

Exit criterion:

- the project has a small, explicit list of local helpers vs host services,
- the next application step does not require inventing the ABI from scratch again.

#### Phase 4: first serious workload = `sqlite`

Why `sqlite` first:

- much smaller dependency surface than `ffmpeg`,
- easier to reason about than SPEC packaging/workflow,
- stresses real file I/O and libc behavior without immediately requiring the whole multimedia stack.

Exit criterion:

- a meaningful `sqlite` scenario runs in the intended execution model,
- failures can be attributed to concrete missing services rather than total ABI ambiguity.

#### Phase 5: broaden to heavier workloads

- `ffmpeg` after the file / buffering / process-environment surface is stronger,
- SPEC-like runs after toolchain/runtime/libc behavior is much more mature and repeatable.

## What the next step is, and why it is next

### Simple answer

The next step is to **tighten the payload-sharing rule in the already working stdout
probe**.

It is next because it is the cheapest change that improves the architecture rather
than just adding more demo code.

### More technical answer

The current stdout HostCall proof has already validated control transfer,
helper-side servicing, and re-entry. The highest-value remaining uncertainty at this
layer is buffer ownership semantics.

Changing the payload from broad shared access toward borrowed one-direction access:

- tests revoke behavior on the live runtime,
- exercises the permission model that later file/I/O services will likely need,
- preserves the same wrapper and success markers,
- and avoids widening the ABI surface before the ownership model is credible.

That is why it is a better immediate step than jumping straight to `sqlite`,
`ffmpeg`, SPEC, or a broad libc port.

## Concrete form of the next step

1. Keep `capstone/caplifive-buildroot/build/local.mk` present so future rebuilds continue to use the local Capstone-enabled Linux/OpenSBI overrides.
2. Treat the OpenSBI/runtime bring-up path as restored.
3. Use the working shared-region probe, the validated HostCall stdout wrapper, and the working baseline/split `null_blk` wrappers as the runtime baseline.
4. Keep the metadata region shared (`INOUT + SHARED`), because both sides still need to inspect it across both rounds.
5. Change only the payload region to a stricter one-direction borrowed handoff for the same `WRITE_STDOUT` proof:
   - domain writes the bytes once,
   - host reads them once after return,
   - automatic revoke semantics should make the data-flow contract stricter and more realistic.
6. Revalidate the same two-round wrapper after that permission change before adding any new opcode or libc-facing surface.
7. Only after that tighter proof is green should the project broaden into richer host calls, micro-libc work, or larger applications.
8. When changing `components/opensbi`, keep using the validated rebuild sequence:
   - `make build CAPSTONE_CC_PATH=... A=opensbi-rebuild`
   - then rebuild any kernel modules/packages whose vermagic must match the active kernel.

## What not to regress

Do **not** accidentally drop back to stock OpenSBI by deleting or bypassing:

- `capstone/caplifive-buildroot/build/local.mk`

If that file disappears again, the earlier wrong-firmware symptoms are expected to come back.

