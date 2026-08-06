# Known-good controls

**A boot whose control fails carries no verdict about anything.** So "which rung is a valid
control" has to be a looked-up FACT, not a judgement made while setting up a run.

On 2026-08-06 `matmult_int` was used as "the canonical known-good control". It is a **documented
silicon MISCOMPILE** (`ISSUES.md:28`, unmeasurable under R-1; commit `03ca1ea85873` records
1166210317 vs the correct 774662735). That single error produced a confident bitstream-regression
claim that had to be retracted, plus a reflash cycle and several boots. Hence this file.

`preflight-board-run.sh` BLOCKS a run whose first rung is not listed here.

| rung | oracle | verified on | last verified | notes |
|---|---|---|---|---|
| `k800` | 4 | `caplifive_fixed_forward.bit`, `caplifive_65536_nodes.bit` | 2026-08-06 | Returned 4 in every boot on both bitstreams. The default control. |
| `k1200` | 4 | `caplifive_fixed_forward.bit`, `caplifive_65536_nodes.bit` | 2026-08-06 | R-14 acceptance test: fails on unfixed silicon, returns 4 with the operand-forwarding fix. Doubles as the "does this bitstream carry the fix" probe. |
| `r14lp` | 4 | `caplifive_fixed_forward.bit` | 2026-08-04 | From the R-14 frame-pad package. |
| `r14sl` | 4 | `caplifive_fixed_forward.bit` | 2026-08-04 | Same package. |
| `gpn2` | 3976364985 | both | 2026-08-06 | Was a C-14 wedge; returns correctly since the movc fix. |
| `gpn4` | 3360062749 | both | 2026-08-06 | As above. |
| `gpw8` | 671377293 | both | 2026-08-06 | As above. |
| `gpw16` | 2928574773 | both | 2026-08-06 | As above. |
| `locfl3head` | 26 | `caplifive_65536_nodes.bit` | 2026-08-06 | The C-14 acceptance test, and the reason it is listed: the SAME rung built ~1 h earlier WEDGED. Returning 26 is what proves the destructive-`movc` fix on silicon, so it doubles as "does this compiler still carry the fix". Re-verify after any change to `CapstonePostRAExpandPseudoInsts.cpp`. |
| `fdreg1` / `fdreg2` / `fdreg3` | 2456 / 2609 / 2736 | `caplifive_65536_nodes.bit` | 2026-08-06 | Static array of {string ptr, function ptr}: cap-init, then derived-capability stores into a global, then indirect calls through the array. Ascending, so they double as a bisection. |
| `fdreg2p` | 2769 | `caplifive_65536_nodes.bit` | 2026-08-06 | `fdreg2` padded past gp index 128, i.e. the `lui/addi/cincoffset/ldc` cap-table path. Pair with `fdreg2w` (2609), the unpadded rung at the same 32 KiB code window, or the window change is a confound. |

## NOT controls — do not use

| rung | why |
|---|---|
| `matmult_int` | Documented silicon miscompile (R-1); `cscall` hangs at every reachable config. Wedges on BOTH bitstreams. Fine as an observation, never as a control. **Still true after the C-14 fix (2026-08-06), despite one boot where it returned 774662735 correctly:** the fix is BYTE-IDENTICAL at -O0 (`md5 71975c8c` with it on and off), so the recorded -O0 miscompute cannot be a C-14 instance and is untouched; and the -O1 pass moved the entry VA as well as the codegen, with the matched fix-off control present in the image but never run. See `history/06-08-2026_21-10-00_matmult-int-is-not-cleared-by-the-c14-fix.md`. A control needs a track record; one boot cannot supply one. |
| `bnds` / `bnd2` | Diagnostic probes with NO fixed oracle -- they encode runtime capability state (`bnds` returns end−cursor+10, `bnd2` an encoded verdict). Their `.oracle` files say 0, which is meaningless. |
| any freshly built image | R-16 is per-image: an image with no track record cannot separate "this image failed" from "the boot failed". |

## Rules

* The control runs **FIRST**. A control that follows the subject cannot exclude a
  position-dependent fault, and the monitor's exact-fit spin is exactly that.
* The control's own failure rate is roughly 1 in 5. A boot whose control fails is **VOID**.
* Quote the control's verdict alongside every result.
* Re-verify a row before relying on it after a bitstream change, and update `last verified`.
