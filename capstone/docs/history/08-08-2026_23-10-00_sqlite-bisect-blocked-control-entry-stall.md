# SQLite bisection blocked by an OPERATOR ERROR — ROOT-CAUSED, 2026-08-08

> ## ROOT CAUSE: the control was launched under the WRONG HOST. Not the board, not the image size.
>
> `run_sqlite_stages_fpga.py` accepts `host|selector:dom`; with no `host|` prefix it falls through
> to `HOST` (`:36` = `/test-domains/sqlite_host.user`). `k800` is a **ladder** rung and requires
> `lpc`. All four VOID boots ran `sqlite_host.user k800.dom`.
>
>     lpc k800 /test-domains/k800.dom          RETURNED 321 / 321
>     sqlite_host.user /test-domains/k800.dom  RETURNED   0 / 5
>
> **Mechanism.** `sqlite_host.c:116,118,136,140` does **two** `create_region` / two
> `shared_region_annotated`; `ladder_perf_ctl.c:367,373` does **one**, and its header says "shares
> it once — that share IS the domain entry". Every share is an entry
> (`sbi_capstone.c:1104-1107` brackets it with SHA5/SHA6). The markers show the domain **entered
> and returned** on share 1, then died on share 2:
>
>     SQ: E/share1 ... SHA5 SHA6 ECSZ     <- entered AND returned
>     SQ: F/share2 ... SHA5 <silence>     <- second entry, never returned
>     per run, all four: A/dom-ok=1 E/share1=1 F/share2=1 G/enter=0 SHA5=2 SHA6=1
>
> `k800.dom` has an unguarded `__test_reentry` that byte-identically re-runs `_start`'s cap-table
> builder (the interp glue guards its rebuild with a `built_flag`; the generic glue does not), and
> the image contains **no `mtvec`**, so a domain-side fault becomes a whole-core wedge. Trap byte
> `0x9a` = **mcause 26 = INVALID_CAPABILITY** (`ex_stage.sv:469`, `riscv_pkg.sv:350`) — a
> capability fault, NOT an R-16 forwarding stall. Bit-identical across three firmware images.
>
> **The driver's "R-16 entry stall / no SQ: G/enter" label is therefore WRONG here**, and it is
> what I trusted for four boots.
>
> ### The fix is one token
>
>     SQLITE_STAGE_DOMS="/test-domains/lpc|k800:/test-domains/k800.dom,/test-domains/bl0.dom"
>
> ### RETRACTION OF A RETRACTION — mine
>
> I proposed "wrong control for this driver", then **withdrew** it citing `sqpc.log`'s
> `### TEST 1/4 END k800:/test-domains/k800.dom rc=0 ###` as proof k800 works under this driver.
> It does — launched with **`lpc`** via the `host|` prefix. Same driver, different HOST. I checked
> the driver name and not the host, and discarded the correct answer. The lesson is narrow and
> worth keeping: when a domain fails to run, identify the **host binary** that launched it, not
> the driver script.
>
> ### What this un-blames
>
> Image size, the 2 MiB padding step, staging a 1.5 MB domain, R-16, and the board are all
> **exonerated**. The 17:39-vs-later correlation was coincidence: every later boot happened to
> use the sqlite driver, and every one of those defaulted the control to the wrong host.


## (original note, superseded above)

# SQLite bisection BLOCKED: the control entry-stalls whenever a SQLite-sized domain is staged

**No SQLite verdict was obtained. Four boots, four VOID.** Recorded so the next session does
not repeat them, and because the pattern itself may be the more useful finding.

## What was attempted

Clamp bisection on `BUILTIN_LIMIT` — how many entries `sqlite3InsertBuiltinFuncs` walks — to
localise the wedge inside `sqlite3RegisterBuiltinFunctions()`, which is the **live**
localization (`SILICON-BLOCKER.md:3-22`). Five variants built and verified: `L=0,1,8,32,64`,
distinct hashes, each with the repaired clamp gate confirming `clamped x1 / unclamped x0`.
The array is 64 entries (`sqlite3RegisterBuiltinFunctions.aBuiltinFunc`, 9216 B = 64 x 144).

`L=0` is the decisive first point: zero builtins registered, loop body never runs. Returns
=> the wedge is in the loop and the axis is valid. Wedges => the axis is dead.

**It never ran.** In all four boots the CONTROL `k800.dom` was reported
`WEDGED (no return)` with **no `SQ: G/enter`** — created but never entered, R-16 entry stall —
so the driver correctly refused to attribute anything, and `bl0` never executed.

## The pattern, which is the actual observation

| boot | image contents | control |
|---|---|---|
| 17:39 | 7 ladder rungs (13-33 KB each), no SQLite domain | **GREEN**, all 7 entered and returned |
| 4 later boots | ladder rungs **+ a 1.5 MB SQLite domain**, then fewer, then `k800`+`bl0` only | **VOID** every time |

Three of those four used **distinct firmware images** (sha `50e0e850`, `b064dc8b`, `d5e57590`),
so this is not R-16 per-image randomness, and it is not the documented ~1-in-5 control flake.
Every boot containing a SQLite-sized domain voided; the one without it was flawless.

## Ruled out, each checked rather than assumed

* **Missing files.** The decompressed initramfs inside `fw_payload.bin` contains
  `sqlite_host.user`, `lpc`, `k800.dom` and `bl0.dom`, with `busybox` as a positive control and
  a bogus name absent as a negative control.
* **Wrong control for the driver.** Briefly asserted and **withdrawn** — `k800` returned `rc=0`
  under this same driver and the same `sqlite_host.user` earlier the same day
  (`/tmp/capstone/mtv/sqpc.log`).
* **Firmware staleness.** Rebuilt `linux-rebuild` then `opensbi-rebuild` in order each time;
  hash changed each time.
* **Bitstream mismatch.** Runs pinned to `caplifive_65536_nodes.bit` via `FPGA_BITSTREAM`.

## A wrong turn worth recording

The 2 MiB firmware growth was attributed to staging TWO 1.5 MB domains, so one was removed.
That could not have worked: the 28 ladder rungs together are only ~400 KB, and the entire jump
came from a **single** SQLite domain. Measuring the components first would have shown that
removing `bl64` changes nothing. It cost a boot.

## Next step, and it is NOT another bisection attempt

Isolate whether a large domain in the initramfs is what breaks entry:

1. Stage the SQLite domain **alone** (no ladder rungs, no control) and see whether it enters.
   Trades away the control deliberately — the question is about entry, not about the workload.
2. If it also stalls, the blocker is staging/layout, not SQLite, and the bisection cannot
   proceed by this route at all.
3. Compare against the 17:39 firmware, which is known-good and whose exact size (15369224) is
   recorded — a diff of what changed in the image between then and now.

**Open question for the project lead:** whether anything changed board-side after 17:39. A
systematic control failure across three distinct images is explained far better by that than by
anything in these builds.

The five clamp variants are built and kept at `/tmp/capstone/sq-bisect/L{0,1,8,32,64}/`, so the
bisection can start the moment a boot yields a valid control.
