# The same eight defects, with the fault delivered instead of halting

`matrix.tsv`: **16/16 arms passed**, on top of PR #52's cooperative client fault
recovery. Same eight defects, same domain program, same paired design as
`../20260918-qemu/`. What changes is what happens after the fault.

| | without delivery | with delivery |
|---|---|---|
| emulator message | `domain halted by capability fault` | `domain capability fault **delivered**` |
| QEMU | exits; guest never returns | keeps running |
| launcher process | — | dies by `SIGSEGV`, guest exit **139** |
| evidence of survival | none possible | a command runs *after* the fault |

So the claim moves from **detection** to **containment**: not merely "the stale
access was refused", but "the access was refused, the offending process died,
and the machine carried on".

## What the runner now requires

For a delivered fault the classifier demands `__EXIT_CODE__139` in the serial.
That marker can only appear if the guest shell survived the fault and ran another
command, so it is the containment evidence rather than a restatement of the
fault.

It deliberately does **not** require the guest runner's own exit status to be 0.
That status stays 1, and correctly so: the generic runner waits for the normal
completion marker, which a killed launcher never prints. Requiring 0 would reject
exactly the arms that demonstrate containment — this cost one debugging cycle and
is recorded so it does not cost another.

## The precondition that is not in any branch

Fault delivery needs a **matching emulator**, and it is not the one the
repository pins:

| | revision | has delivery |
|---|---|---|
| repo submodule pin | `deb7d757` | **no** |
| locally built QEMU | `408fd8394503` | **no** |
| required | `77d69353b7`, branch `runtime/1-domain-trap-delivery` | yes |

`runtime/domain-faults.md` states it plainly: *"an old emulator still stops on
their faults"* — which is exactly what we measured before rebuilding. Both
negatives above were checked by ancestry, not assumed.

So merging PR #52 alone does not give anyone fault delivery. The emulator has to
be built separately from that branch and selected through `CAPSTONE_QEMU_BINARY`.
Built here with the same options as the existing build:
`--target-list=riscv64-softmmu --disable-docs --disable-werror`.

Domains and guest are built with `-DCAPSTONE_DOMAIN_FAULT_RECOVERY=ON`; recovery
is opt-in and off by default.

## Cost

94 seconds for 16 arms, against 70 without delivery. Delivery does **not** make
this experiment faster — at this size it costs a little, because the guest now
does more after each fault instead of the emulator exiting. Its value is the
containment claim. An earlier estimate in this work put a boot at two minutes;
that was wrong by about thirty times and is corrected here and in the sibling
results directory.

## Limits, from the runtime contract

Cooperative, not monitor-enforced containment of malicious C-mode code. No
firmware or kernel change. No FPGA claim. Recovery state is 256 declared bytes
outside the application budget. See `capstone/runtime/domain-faults.md`.
