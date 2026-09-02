# S-12: store-buffer pressure is not the missing trigger either

## The measurement

Four domains in one boot — possible only because the fault now returns instead of taking the core
— control first, then three pressure variants of the shape rung, two draws:

| rung | pressure added before the shape | draw 1 | draw 2 |
|---|---|---|---|
| `k800` (control) | — | oracle 4, got 4 | oracle 4, got 4 |
| `s12sb8` | 8 scalar stores, conflicting lines | `20770` | `20770` |
| `s12sb32` | 32 scalar stores, conflicting lines | `20770` | `20770` |
| `s12cb8` | 8 **capability** stores (occupy the serialising DYN unit) | `20770` | `20770` |

`20770` = `0x5122` is "the loop completed without faulting". 4096 iterations per arm per draw, so
**24,576 further executions under pressure, zero faults**, on top of the unpressured rung's 12,288.
36,864 in total, in a real capability domain on silicon, at the exact shape SQLite faults on.

The capability-burst arm matters separately from the scalar ones: an STC occupies the DYN unit,
which serialises one op in flight, and a scalar store does not. Both behave identically here.

## What the search has now eliminated

| variable | verdict | evidence |
|---|---|---|
| instruction shape alone | not sufficient | 3 Verilator variants; precondition counter (its own positive control) fires every iteration, outcome 0 |
| register relation | **NECESSARY** | board ladder, localised to one byte, 2x2 dissociation |
| register identity / ABI class | not the cause | D3 wedges on x5, D4 clean on x14 |
| producer distance | not necessary, not sufficient | D2 wedges at distance 7, D1 clean at distance 1 |
| stored value | not the cause | null in every arm |
| capability domain context | not sufficient | 12,288 executions, zero faults |
| store-buffer pressure | **not sufficient** | this run, 24,576 executions, zero faults |

The mechanism's own stated requirement was a store-buffer-stalled STC. Adding that traffic
directly, in the domain, does not produce the fault. Either the stall is not being created by this
burst — plausible, and not separately instrumented on silicon — or the stall is not the trigger.

## What remains, in the order worth testing

1. **Slot provenance.** SQLite's load and store slots are frame offsets `s0-0x70` and `s0-0x120` on
   a MONITOR-CARVED STACK. This rung uses a static buffer reached through the cap table. Those are
   capabilities of different provenance — different bounds, possibly different revocation nodes —
   and nothing has tested whether that matters. This is the largest remaining structural difference
   and the next variant to build.
2. **Dependency depth.** In SQLite the loaded capability comes from a spill written 18 instructions
   earlier by an incoming argument; here it is stored and reloaded immediately.
3. **Scale.** A 4600-instruction function after ~1060 cap-table carves, against a four-instruction
   loop — instruction-cache behaviour and pipeline occupancy differ in ways a burst does not model.

## Method note

The 5-domain version of this run was BLOCKED by preflight: the monitor spins at about the 5th
`create_dom`, so slot 5 carries no verdict. There is an override and it was not used — the arm was
dropped instead, because the unpressured baseline is already measured at 3/3 clean and reading an
untrustworthy slot to save one boot is how a number that looks like data gets into the record.
