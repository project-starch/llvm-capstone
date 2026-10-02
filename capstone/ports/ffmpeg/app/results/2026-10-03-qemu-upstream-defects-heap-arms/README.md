# Two upstream FFmpeg defects on the capability machine: level0, shrink, sublet (2026-10-03)

**Question.** Fixtures 24 and 25 reproduce two defects that the triage found **live at the 9.0.1
pin**. Does the revoking heap catch them, and do the unprotected arms let them through?

**Pre-registration.** Both fixtures and all six predictions were pushed in `95aa3d340f80`, before
either image was built. Nothing in that file has been edited since, including the one cell that
differs.

## Verdict

**Five of six cells exactly as pre-registered; the sixth differs in a predicted value for a reason
entirely in the derivation, with the mechanism conclusion unchanged.**

| arm | fixture 24 — `vvc/thread` | fixture 25 — `ops_dispatch`, interior pointer |
|---|---|---|
| `level0` | RETURN `180015b` — **as predicted** | RETURN `190017b` — predicted `190015b` |
| `shrink` | RETURN `180015b` — **as predicted** | RETURN `190017b` — predicted `190015b` |
| `sublet` | **FAULT temporal, cause 24** — as predicted | **FAULT temporal, cause 24** — as predicted |

**The faults are attributed, not merely observed** — by hand, against the rule the port's
classifier applies, **not by the classifier itself** (see "What the oracle could not judge").
Each fixture published the address it was about to touch, and QEMU's diagnostic names the same
value:

    FFAPP-FIX 24 target=c4802000   <->  cincoffset with an UNTAGGED rs1 ... val=0xc4802000
    FFAPP-FIX 25 target=c4802020   <->  cincoffset with an UNTAGGED rs1 ... val=0xc4802020

Both faults follow the fixture's own `touch` line, and **neither section contains a `returned` or
`mark` line at all** — which is the rule `safety-verdict.py:112` actually enforces, stricter than
"nothing after it". The value comparison is the rule that matters: an untagged-operand fault counts
as temporal **only** when the untagged value is the fixture's printed target, because without that a
null dereference would read the same.

One correction to how this was first written: the ordering is **structural, not temporal**.
`check-safety.py` concatenates stdout then the QEMU slice, so a diagnostics-channel fault is always
after the touch line by construction, and neither channel carries timestamps. The load-bearing
checks are the absent `returned` line and the matched value, not the order.

## What the oracle could not judge, and why

These six cells were run through `tests/runtime-qemu/run-domain-smoke.py` by hand, not through
`ports/common/application/check-safety.py`. Re-running the recorded artifacts through the real
classifier afterwards:

- **the four RETURN cells pass the real oracle**, including its exit-code corroboration
  (`result['value'] == mark & 255`: 91 = `0x5b`, 123 = `0x7b`);
- **the two FAULT cells cannot be judged by it on this platform at all.** `check-safety.py:40`
  requires a `fault` field in `capstone-job`'s JSON, and this build of `capstone-job` emits only
  `{"version":1,"kind":"%s","value":%d}` — there is no `fault` field in the binary. The oracle's
  signal/11 branch is therefore **structurally unreachable** here, not merely skipped.

So for the two faults the pc↔diagnostic correspondence and the cause were checked by hand. They
hold, and more strongly than claimed above: `pc=0xc020ffbc` disassembles to the `cincoffset` inside
`ffapp_fix_touch` in both images, and each fault record's `sha256=` matches its `.dom` byte for
byte. But a hand check is not the gate, and this bundle does not contain a run of the gate.

**The arms differ in the one thing they are supposed to.** The capability the fixture holds:

    level0   bounds=[c0253220,c03d3220)  len-from-cursor=1522896   the whole arena
    sublet   bounds=[c4802000,c4802040)  len-from-cursor=64        the object alone

and on `sublet` that alias is dead at the touch, while on `level0` and `shrink` the read completes
and returns the **new owner's** byte with `same-address=1` — the storage really was freed and
reissued.

## The differing cell, and why it is not a mechanism finding

Fixture 25 returned `0x7b` where the prediction said `0x5b`. The cause is in the prediction's
arithmetic, not in the machine: `fill(p, v0, n)` writes **`v0 + i`**, an incrementing pattern, and
fixture 25 reads at **offset 32** of the reissued 64-byte object, so the new owner's byte there is
`0x5b + 32 = 0x7b`.

**The control for this is a native re-run, not fixture 24.** An earlier version of this file said
fixture 24 matching "isolates the error to the offset rather than to the fill" — that is wrong, and
it is the same class of mistake as the prediction itself: fixture 24 reads at offset 0, where
`v0 + 0 == v0`, so it behaves identically under a constant fill and an incrementing one. It isolates
the *reuse*, nothing about the fill. The real control is `native-pair.c` re-run with the fixture's
incrementing fill, which prints `read=7b` for case 25 and `read=5b` for case 24 — see the native
bundle's own correction.

The prediction file keeps the value it was given. What the cell establishes is unchanged and is what
the fixture was for: the stale **interior** pointer read the new owner's data at its own offset.

**And the registered hypothesis about interior pointers held.** The expect file predicted that
fixture 25's interior pointer "changes nothing, because revocation is per object and not per alias,
and a disagreement between 24 and 25 is itself the result". There is no disagreement: both fault on
`sublet`, both complete on the unprotected arms, and 25's capability is the same object's bounds
with the cursor 32 bytes in.

## What this does and does not say

- **It is the CONTROL half, not a detection claim.** Both defects are plain `malloc`/`free`
  use-after-free, which ASan also reports — measured the same day in
  `../2026-10-02-native-upstream-defects/`, where both buggy arms report `heap-use-after-free` and
  neither fixed arm does. The value here is that the 2×2 is now measured on *real upstream defects*
  rather than on shapes we invented, in both rows.
- **Nothing about nested allocators.** Neither defect is in FFmpeg's pools.
- **QEMU only.** The temporal faults are the emulator untagging a revoked capability on reload
  (Q-11); the deployed silicon lets such an access retire.
- **N = 1 per cell.** Six cells, one boot per arm, two fixtures per boot.

## The platform these ran on, which is not the default one

The images are application-SDK images and need the delegated runtime's **process ABI** — SBI
functions `0x21`–`0x2b`. The buildroot platform installed on this host provides none of them, which
is recorded in
[`docs/history/02-10-2026_20-30-00_application-images-blocked-on-monitor-process-abi.md`](../../../../../docs/history/02-10-2026_20-30-00_application-images-blocked-on-monitor-process-abi.md),
which now carries a correction block for the four claims this run refuted. That document said the
implementation was nowhere on this machine. **That was wrong, and this run is the correction:** it
exists, built, in the delegation lane's tree, and the pieces are a matched pair built within a
minute of each other on 2026-10-01.

| piece | what was used | why not the default |
|---|---|---|
| monitor | `deleg-gate2/opensbi-T/.../fw_jump.elf`, 41 `context_step` symbols; its `capstone-sbi` is **byte-identical to the pinned `4674ab6a`** | the installed `fw_jump.elf` has **0**; it dispatches 14 capstone SBI functions and no `PROCESS_*` |
| QEMU | `deleg-gate2/qemu-12/build/qemu-system-riscv64` — a checkout at **exactly the branch's pin `674cdab03c3e`** | with the stock QEMU the monitor faults early in its own address range and the guest never reaches the launcher (observed once, N = 1, serial log not kept: an observation, not evidence). `CAPSTONE_SUPERVISED_CALL` is a **monitor-side** `#ifdef`, not a QEMU one |
| kernel module | built out-of-tree from buildroot `8fd1ea1249a9` | the installed `capstone.ko` has **zero** module parameters; this one has `parm=process_cache_bytes` |
| launcher | `capstone-exec`, `capstone-job` built against the pinned `libcapstone.c` | absent from the rootfs entirely |
| compiler | `7d01722aab88`, Release + assertions | `dev`'s compiler has no intcap extensions, which the application SDK requires |
| kernel, rootfs | the installed ones, unchanged | — |

The run goes through `tests/runtime-qemu/run-domain-smoke.py` over the serial console rather than
`capstone-vm`, because the rootfs has no dropbear. The guest-side command is the one `capstone-vm`
issues: `capstone-job … -- capstone-exec -- <image>`.

**Corrected, and it strengthens the result rather than weakening it.** An earlier version of this
file called the platform "assembled from one lane's build trees, not from this branch's submodule
pins", and said the pins disagree with each other. **That was wrong.** The pins agree; it is the
main clone's *working trees* that lag them:

| component | this branch pins | main clone has checked out | what this run used |
|---|---|---|---|
| QEMU | `674cdab03c3e` | `deb7d757` | **`674cdab03c3e` — exactly the pin** |
| buildroot → `components/opensbi` → `capstone-sbi` | `4674ab6a` | `2c49c41c` | **byte-identical to `4674ab6a`** |

So these cells ran on the platform this branch pins, assembled by hand because the main clone's
submodule checkouts are stale — not on a platform of someone's invention.

**What remains genuinely weak**, and is the reason this section still exists: the outer OpenSBI
revision that `deleg-gate2/opensbi-T` was built from is **not recorded** — that tree is not a git
checkout, and only its `capstone-sbi` subtree could be matched against the pin. Until that is
resolved the run is not reproducible from the branch alone, which is what the platform script in
this bundle's follow-up is for.

Files: `result-lines.txt` (all six cells), and the staged platform at
`/tmp/capstone/pinned-platform/` with the per-arm run directories.
