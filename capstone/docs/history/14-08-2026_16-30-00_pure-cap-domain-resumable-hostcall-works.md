# A pure-capability domain CAN make a blocking hostcall and resume, 2026-08-14

**Result, QEMU, verified.** A `.dom` compiled for `capstone64-unknown-elf` yields
to the host mid-execution, the host services a HostCall v0 request, and the
domain resumes at the instruction after the yield with its C frame intact.

```
__CAPSTONE_QEMU_BOOT_CONTROL_OK__
yield-probe: round 1 before yield
yield-probe: round 2 AFTER RESUME, stack intact
yield-probe: DONE after 2 serviced request(s), domain entered domain_main 1 time(s)
__CAPSTONE_YIELD_PROBE_PASSED__
```

Harness: `capstone/musl-capstone/yield-probe/run-yield-probe.sh`.

**Why it matters.** A syscall must return to the instruction after itself. The
shared entry glue instead RESTARTS `domain_main` on every entry, and the existing
HostCall v0 probes sidestep the question by running an S-mode payload in a nested
domain (`create_dom(sbi.dom, x.smode)`), which is not pure-capability and so
cannot carry a libc. Without resumption there is no POSIX layer for domains.

## How the check distinguishes resume from restart

A restart and a resume both look like "the domain ran again", so three
independent discriminators had to hold at once:

| evidence | rules out |
|---|---|
| round 2 sends a DIFFERENT message | a restart would re-send message 1 forever |
| a local set before the yield is checked after it (`MARKER-LOST` otherwise) | a lost or reset stack |
| `domain entered domain_main 1 time(s)` | restarting and happening to reach the same place |

## The mechanism

`RETURN` sets `pc.cursor` to `x[rs1]` and only then swaps the C-effective
registers out, so `x[rs1]` is where THIS domain resumes on its next `CALL`.
Passing a label inside the yield routine instead of `__test_reentry` is the
entire difference from the shared glue.

**`sp` does NOT survive the boundary; `gp` does.** The first attempt assumed the
register swap restored the whole C state and saved only `ra`. Measured result:

```
[CAPSTONE] Cap mem access requires capability: pc = ..., rs1 = x2, imm = 0
[CAPSTONE] domain halted by capability fault: cause = 24
```

`rs1 = x2` is `sp`, at the first sp-relative `ldc` after resume: sp returned as a
plain integer. The gp-relative access two instructions earlier did not fault, so
gp survived -- consistent with the monitor handing gp to the domain on entry.

The remedy was already encoded in the shared glue and was there to be read: it
stashes sp into `cscratch` before returning and `__test_reentry` recovers it with
`ccsrrw(sp, cscratch, x0)`. cscratch is the designated carrier for a capability
across a domain boundary. So sp travels in cscratch and everything else travels
on the stack that sp then finds again (`ra`, `gp`, `s0`-`s11`).

cscratch needs no save/restore: `_start` reads it with `ccsrrw(sp, cscratch, x0)`,
leaving null, and the resume path does the same.

## Two process notes from the same session

**A tiny domain hits the open `helper_cssplit` fragility.** The probe's first
image was 1232 bytes and the boot produced nothing. `ISSUES.md` records the
monitor splitting the code capability at what looks like a fixed `0x1000`
offset, with the countermeasure "`__pad` keeps image > 0x1000". Padding to 5376
bytes cleared it (`Segment size = 1500` in the log). The padding carries
`retain` plus a live reference because `--gc-sections` would otherwise drop it
and silently restore the failure.

**Boot-to-login on this host straddles 8 minutes.** At
`--timeout-multiplier 4` (login timeout = 120 x multiplier) two of three runs
returned `__CAPSTONE_INFRA_FLAKE__ phase=boot-login` and one succeeded. Default
raised to 8. The harness names the phase, which is what made this cheap to tell
apart from a domain stall.

**`--success-marker` applies to EVERY `--guest-command`.** Batching the boot
control as a separate command made the control fail for missing the probe's
markers, and the probe then never ran. Both must share one command.

## Next

`runtime/hostcall.c` maps Linux syscall numbers onto HostCall v0 opcodes and is
written but not yet exercised. The link probe already showed that
`write(1, ...)` against the partial musl archive leaves
`__capstone_hostcall` as the ONLY undefined symbol, so the remaining step is to
run musl's own `write()` through this yield.
