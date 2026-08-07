# R-16 — the domain never returns from its FIRST entry (`SHA5` stall)

> ## RESOLVED 2026-08-04 by `caplifive_fixed_forward.bit`. Kept as a bitstream acceptance test.
>
> R-16 was the same **capability operand-forwarding bug** as R-14 (`capstone-ariane
> 7aac52f93`, "Fixed an operand forwarding bug", `issue_read_operands.sv`: capability-metadata
> forwarding was selected by an over-broad `check_cap_op`, narrowed to `check_fwd_rs1`).
>
> The reproducer below entry-stalled **8/8** on `working-caplifive-captype-fixed.bit` and
> **enters** on `caplifive_fixed_forward.bit`, on a boot whose control both entered and
> returned.
>
> We never root-caused R-16 ourselves. Eight axes were eliminated (below) and none separated
> entering from stalling images; the mechanism came from the RTL side. That is worth stating
> plainly, because the eliminated-axes list reads like progress and was not.
>
> **Why this package still exists:** it converts "does this bitstream have the forwarding
> fix?" into one boot — for a bitstream that has not been checked yet.
>
> **UPDATE 2026-08-07: `caplifive_65536_nodes.bit` DOES carry the fix.** This paragraph used to
> say its status was "unconfirmed"; that has been stale since 2026-08-06.
> `ref/SILICON-BLOCKER.md:347` records `k1200` — the R-14 acceptance rung, which fails on unfixed
> silicon and returns 4 with the fix — returning **4 on both bitstreams**, across four boots with
> byte-identical firmware and a passing control in each. `ref/known-good-controls.md:16` lists it
> the same way.
>
> One hop remains between that and "R-16 is fixed on the resident silicon": `k1200` tests **R-14**,
> and the claim that R-14 and R-16 are the same defect is an inference (see the header above), not
> a measurement on this bitstream. The direct R-16 test is the `sb1`/`sb0`/`f10` set below, which
> has never been run on 65536.
>
> **Why this matters more than a doc fix.** An entry stall on the resident bitstream is therefore
> *probably not R-16*, so "redraw and retry" is probably the wrong response to it. On 2026-08-07
> five stalls were treated as R-16 on the strength of the stale sentence above, and at least two of
> them turned out to be a staged `.dom` that was never packed into the initramfs — now blocked by
> gate C14 in `preflight-board-run.sh`. Check membership before blaming the silicon.

---

## What the failure looks like

The monitor completes the region share and hands off; the domain never comes back. The last
UART line is `SHA5:xxxx`.

`SHA5` = "about to leave M-mode for the domain"; `SHA6` = "the domain returned from the share
entry" (`sbi_capstone.c`). A stop between them exonerates the monitor: the domain died on its
**first** entry, which is where the glue builds the capability table (one `split` per global)
and runs `__capstone_cap_init`.

**The domain's own code never runs.** So a stalled run carries *no* information about the code
under test and must never be recorded as a result for it. This is what made R-16 expensive: it
biased *which constructs could be measured at all*, and for a while every minimisation arm was
being silently thrown away.

### The classification rule — get this right or the verdict is worthless

| evidence in the run-scoped transcript | meaning |
|---|---|
| `SQ: obs=<n>` / `RESULT … retval=` | a result — record it |
| `SQ: G/enter` present, no `H/return` | entered and **wedged** — a real, bisectable result |
| **no `SQ: G/enter`, ends at `SHA5:`** | **entry stall = R-16** — says nothing about the code |
| no boot banner / JTAG errors | infra — retry |

`SHA5` last does **not** by itself mean an entry stall: a domain that enters and wedges
immediately also leaves `SHA5` last. **Distinguish on `SQ: G/enter`.** Getting this backwards
is the single easiest way to mis-report this issue.

## The reproducer

| image | build | old bitstream | new bitstream |
|---|---|---|---|
| `sb1` | `SQLITE_STATIC_BUILTINS=1` | entry-stalled **8/8**, no `SQ: G/enter` ever | **enters** (`SQ: G/enter`) |
| `sb0` | `SQLITE_STATIC_BUILTINS=0` | entered | enters |
| `f10` | known-entering control | entered | enters |

`SQLITE_STATIC_BUILTINS=1` is the **R-14 workaround** (`build-sqlite-silicon.sh:75`), so R-16
was created by the R-14 workaround — the two issues are the same underlying defect reached
from two directions.

Selector `:0` returns before any ladder code runs, so it isolates entry from everything else.

### Building it

Images are **not committed**: each is ~1.5 MB, and the package policy in `../README.md` is
that multi-MB SQLite builds ship as source plus recipe. `src/` carries the pieces the stall
actually lives in — the interp glue that builds the cap table, its generator, the domain
header and the controller.

```bash
source capstone/tests/capstone-test-env.sh
SQLITE_STATIC_BUILTINS=1 OUT_DIR=/tmp/capstone/sb1 \
  bash capstone/benchmarks/sqlite/build-sqlite-silicon.sh     # the reproducer
SQLITE_STATIC_BUILTINS=0 OUT_DIR=/tmp/capstone/sb0 \
  bash capstone/benchmarks/sqlite/build-sqlite-silicon.sh     # the control
```

`IMAGE-HASHES.txt` pins the exact binaries that produced the 2026-08-04 verdicts. A rebuild may
differ in hash while being the right image — build output is not bit-reproducible here — so
**identify a build by size and carve count, not by filename and not only by hash**:

| build | size | carves |
|---|---|---|
| `SQLITE_STATIC_BUILTINS=1` | 1,551,512 | 179 |
| `SQLITE_STATIC_BUILTINS=0` | 1,607,832 | 179 |

Read the carve count from the `.capstone_gp_initdesc` header (u64 `count` at offset +8):

```bash
python3 - <<'PY'
import struct, subprocess
p = "/tmp/capstone/sb1/sqlite_silicon.dom"
out = subprocess.run(["llvm/cmake-build-debug/bin/llvm-readelf","-SW",p],
                     capture_output=True, text=True).stdout
off = size = None
for l in out.splitlines():
    if "initdesc" in l:
        f = l.split(); off = int(f[5],16); size = int(f[6],16)
print("carves =", struct.unpack_from("<Q", open(p,'rb').read()[off:off+size], 8)[0])
PY
```

> **The staleness trap — this cost a board session on 2026-08-04.** A `sqlite_silicon.dom`
> left staged in the buildroot overlay was a `STATIC_BUILTINS=0` build even though the builder
> now defaults to `1`. Its `SQ: G/enter` was read as "R-16 is gone" when it was simply the
> shape that always entered. Always re-derive the identity of a staged image before drawing a
> verdict from it.

### Running it

See `run.sh`, or `../../agent-handoff/ref/HOW-TO-LAUNCH-ON-FPGA.md` and the `board-run` skill.
Domains are **baked into the buildroot image** — never shipped over UART.

```bash
export FPGA_URL="$(cat ~/.claude-c/secrets/fpga-console-url)"   # secret; never commit or echo
export FPGA_FW=.../opensbi-custom/build/platform/fpga/ariane/firmware/fw_payload.bin
export SQLITE_STAGE_DOMS="/test-domains/f10.dom:0,/test-domains/sb1.dom:0"
export PROBE_SCOPED_OUT=/tmp/capstone/r16.txt
python3 capstone/tests/rtl-smoke/fpga_driver/run_sqlite_stages_fpga.py
```

Three rules, each of which produced a wrong verdict when skipped:

1. **A known-entering control runs FIRST in every boot.** The control itself fails roughly 1 in
   5 — and 2 of 3 boots in the 2026-08-04 session. **A boot whose control fails is VOID** and
   carries no verdict about anything. `run.sh` gates on this and retries.
2. **At most one expected-to-stall domain per boot, placed LAST.** A stalled or wedged domain
   takes the core with it; everything after it is collateral, not a result.
3. **Retrying the same binary is futile — REDRAW instead.** Rebuild with a harmless constant
   varied so the code under test is byte-identical across draws, and `sha256sum` the set to
   prove the draws differ. Three boots were once spent retrying one binary.

Keep `ENTRY_STALL_S` ≥ 260 in `board-watchdog.sh`: the JTAG upload is 133–227 s of entirely
legitimate UART silence, and a lower threshold aborts healthy runs mid-upload. The old 45 s
default produced a session's worth of false "will not boot" diagnoses.

## What was eliminated and never explained

All measured on the old bitstream, none of which separated entering from stalling images:

| axis | result |
|---|---|
| image size | ruled out (1087 KB pair) |
| carve count | ruled out (192 / 3072 B) |
| size + carve **conjunction** | ruled out |
| `dom_data` geometry | ruled out — byte-identical blob/cap-table/storage/stack/globals-offset across an entering/stalling pair |
| blob size on the ladder path | ruled out |
| the loader | ruled out |
| the interp-glue pad bug | ruled out — fixed it; R-16 unchanged |
| `BUILTIN_LIMIT` | ruled out |

The one clean lever was `SQLITE_STATIC_BUILTINS`: `=0` entered 8/8, `=1` stalled 8/8, with the
only measured difference being the blob (75120 vs 84336 bytes). That is what made it look like
a code-shape property rather than a pipeline hazard.

Two further observations that were true and are worth keeping, because they constrain any
future explanation:

* **It is a property of the IMAGE, not the board or firmware** — proven by running a
  known-entering control first in the same boot on the same firmware.
* **But not strictly per-image either:** the same binary in the same boot entered for `:0` and
  stalled on a later selector. So it tracked which *invocation* ran too, and "deterministic per
  image" overstates it. This is consistent with a pipeline forwarding hazard and was, in
  hindsight, the strongest hint available.

## References

- `capstone/agent-handoff/ref/SILICON-BLOCKER.md` — full measurement trail, the retractions,
  and the 2026-08-04 bitstream change.
- `capstone/agent-handoff/ref/ISSUES.md` — R-16 entry.
- `capstone/agent-handoff/ref/HOW-TO-LAUNCH-ON-FPGA.md` — board procedure and driver contract.
- `.claude/skills/board-run/SKILL.md` — the decision procedure, including this classification.
- `../ARCHIVED/R14-frame-pad/` — R-14, the same defect reached from the other direction, with
  the ~10 KB one-variable pair that is cheaper to run as a bitstream check.
