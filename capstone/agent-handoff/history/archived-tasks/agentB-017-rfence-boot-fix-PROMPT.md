# Follow-up prompt for Agent-B — task 017: fix the RFENCE boot gap (scoped green light)

*Paste below the line into `claude-b`. Self-contained. The board run reached the
monitor boundary: our image boots OpenSBI + Linux but never reaches a shell because
the FPGA monitor lacks the SBI RFENCE extension. This authorizes the fix — scoped,
not a blank check.*

---

You are Agent-B, continuing task 017 in `/home/alexey/dev/llvm-capstone-b`,
branch `capstone-bootstrap-b`.

```bash
cd /home/alexey/dev/llvm-capstone-b && git fetch origin && git pull --rebase
source capstone/tests/capstone-test-env.sh
```

## Where the run stopped

Clean load + reset, OpenSBI and Linux 6.4.14 both boot, reach `Run /init`, then the
console floods with `remote fence extension is not available in SBI v1.0` and resets
~9 s into userspace, looping. Your root cause is accepted: the `genesys-testing`
OpenSBI doesn't implement SBI **RFENCE**; exec'ing `/init` needs an icache flush
that Linux routes through `remote_fence_i`, the call fails, the icache is never
flushed, the core faults at `/init`'s entry and resets. QEMU boots the same image
because its OpenSBI has RFENCE.

You are authorized to fix this so the board boots to a shell and the sweep runs —
**within the scope below.**

## Step 1 — confirm the blast radius (do this first, read-only)

Establish **where the OpenSBI that runs actually comes from**:

- If OpenSBI is **embedded in the `fw_payload.bin` we build and upload** (LINUX_PAYLOAD
  flow — this is what your phase-3b build implies, "our OpenSBI"), then the fix is a
  **rebuild of our own image** — no board-resident firmware is touched. This is
  in-scope.
- If instead the board runs a **persistent, board-resident OpenSBI/monitor** that our
  `fw_payload` does *not* supply, then fixing it would mean **re-flashing shared board
  firmware** — that is **out of scope** (see carve-outs). In that case **stop and
  report**; it needs the monitor owner.

## Step 2 — investigate *why* RFENCE is off (read-only, zero risk)

In `caplifive-opensbi` / `caplifive-sbi` on branch `genesys-testing`, determine
whether RFENCE is:

- **(a) off by build config** — a platform defconfig / Kconfig / `platform.mk` that
  disables the RFENCE extension (OpenSBI ships it **on** by default). Trivially and
  safely re-enabled. **OR**
- **(b) deliberately stubbed/removed in code** with Capstone-specific handling —
  i.e. someone removed or no-op'd the RFENCE implementation on purpose.

**If (b): STOP and report.** A deliberate removal is a red flag — the Capstone
domain/capability model may be unable to safely service a remote fence, and forcing
it on could yield a broken monitor or invalid measurements. That case needs the
monitor owner, not a unilateral flip.

If (a), proceed to Step 3.

## Step 3 — the fix (our-image rebuild only)

Preferred, then fallback — **both only change the image we upload**, and both are on
the **boot/exec path only**, so they do **not** touch the userspace malloc/free
measurement loop: the cycle numbers stay valid.

1. **Rebuild the fw_payload's OpenSBI with RFENCE enabled** (cleanest). **Critical:**
   keep the **`genesys-testing` Capstone monitor** (it carries the revoke-cost
   region-share ABI you already verified — `REV_TRANSFERRED`/`REV_SHARED`/
   `DPI_REGION_SHARE`). **Do not** substitute stock/upstream OpenSBI to get RFENCE —
   that would drop the Capstone domain support and your `.dom`s won't run (or would
   run invalidly). The fix is "genesys-testing **plus** RFENCE," nothing less.
2. **Fallback — rebuild Linux to use a local `fence.i`** instead of SBI
   `remote_fence_i` (valid on this single-hart board). Kernel build/flag change;
   only affects our image.

Re-verify the image the way you did before: decompress the initramfs and confirm the
six `/root/rtl-smoke/*.dom`/`*.user` are present, and record the new sha256.

## Step 4 — run the sweep (unchanged gate)

Re-run the same gated flow from the board-run brief: **Stage 0** (image staged /
sha) → **Stage A** (offline mock conformance green) → **Stage B** (live read-only,
payloads match `socketio-api.md`) → then Lock → power → load → reset → **now boots to
a shell** → run the `.user`+`.dom` pairs → capture UART `RESULT` → `--parse-uart` →
release Lock. Board URL supplied at paste time; runtime only.

Report the **cycle** breakdown next to the QEMU reference:
- **Revoke-cost:** bump / norevoke / revoke cycles/op and the revoke-at-free delta
  (QEMU: bump 7 / norevoke 60 / revoke 65 → **+5 instr, O(1)**); state whether the
  RTL delta confirms O(1) in real cycles.
- **Borrow-cost:** raw / borrow / copy cycles/op (also fills the C1 spatial number).

## Carve-outs (this is what "scoped" means)

- **Do NOT re-flash the board's persistent bitstream/firmware** or swap the
  collaborator's served monitor. If the fix would require that (Step 1 case 2), stop
  and report — that is shared infrastructure others use.
- **Do NOT commit changes into the `caplifive-opensbi` / `caplifive-sbi` submodule
  history.** Keep the RFENCE enablement (or kernel `fence.i` change) as build
  flags / a patch / overlay in our own staging, so it's reproducible without
  rewriting the collaborator's repo.
- If Step 2 shows a deliberate RFENCE removal (case b), **do not force it on** —
  flag it.

## Deliverables

- Boot-to-shell fix documented (which path, why RFENCE was off, how re-enabled),
  reproducibly (build flag / patch in our staging).
- The cycle-accurate numbers, or — if you hit case (2)/(b) — a precise report of
  exactly what needs the monitor owner.
- New image sha256 + initramfs contents re-verified.
- History note → `capstone/agent-handoff/history/DD-MM-YYYY_HH-MM-SS_fpga-rfence-boot-fix.md`.
- Commit to `capstone-bootstrap-b`; report readiness for the results merge into
  `capstone-bootstrap`.

## Guardrails (unchanged)

- Beyond the scoped OpenSBI/kernel build change above, still **no `llvm/`, RTL, or
  submodule-history changes**; the driver stays additive test tooling.
- Token never enters the tree or any log; good-citizen on the shared board (Lock,
  short hold, clean state, back off if it's in use).
- Commit on `capstone-bootstrap-b`, exact paths, **no `Co-Authored-By:`**, no
  worker/agent identity in commit messages, no debug/report files.
