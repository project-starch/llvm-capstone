# R-35 — on silicon a REVOKED capability still reads AND writes the storage its object has given up, at every age, and the access does not trap

**Status (2026-09-19): REPRODUCED ON THE BOARD, twice, with controls. The mechanism is NOT identified,
and the two obvious candidates are both already fixed in the RTL this bitstream is labelled from — which
is the reason this folder exists rather than a line in an existing one.**

Sibling issues, so a reader who arrived with the wrong symptom is redirected now:
`../R34-lsu-exception-lost-on-immediate-grant/` is the LSU dropping exceptions it generates, and its fix
(`c77c65324`) **is an ancestor of this bitstream's RTL `054cea69b`** — verified by `git merge-base
--is-ancestor`, so R-34 does not explain this. `../RTL-cap-mcause-off-by-one/` (R-24) is the capability
causes aliasing the core's `DEBUG_REQUEST = 24`, which would route a trap into debug mode instead; the
same commit says it "move[s] the debug sentinel off 24, and put[s] the capability causes on the spec's
base", so R-24 does not explain this either. **This folder is one issue: a capability check that does
not fire on silicon.**

## What was observed

The harness retains one alias per allocation, runs the allocator to its buffer limit, and then probes
the aliases it kept. Board transcript, `results/board-full-probe.r1-lines.txt`, image
`35fb3fec3196841b`, bitstream `caplifive_m1_054cea69b.bit`:

    R1 m1 end arm=pressure stop=buffer alloc=43297 minted=43328 revoked=43297 retained=43296 released=0
    R1 m1 stale-take go
    R1 m1 stale-deref ok k=0     reuses_since=2706 byte=16 live_byte=16 is_live_data=1
    R1 m1 stale-deref ok k=21648 reuses_since=1353 byte=16 live_byte=16 is_live_data=1
    R1 m1 stale-deref ok k=43295 reuses_since=0    byte=31 live_byte=31 is_live_data=1
    R1 m1 stale-write ok
    R1 m1 stale-write readback via_live_alias=165 wrote=165

Read in order:

- **The reference is revoked.** `revoked=43297` — every alias probed belongs to an object whose handle
  was given back. `k=0` was retained at the first allocation and its storage has since been handed to a
  new object **2,706 times**.
- **It still reads, at every age.** Oldest, middle and newest all return a byte.
- **What it reads is the CURRENT OCCUPANT'S LIVE DATA, not residue.** `live_byte` is recorded by the
  harness as the arm runs — the value the most recent allocation wrote into that storage — and
  `is_live_data=1` says the stale read returned exactly that. So this is not a stale-bytes leak; it is a
  working pointer into somebody else's object.
- **It also WRITES.** `0xA5` (165) stored through the revoked reference is then read back **through a
  live alias to the same storage** and is there. The revoked reference can corrupt the current occupant.
- **Nothing trapped.** No wedge; `k800 retval=4` on both the opening and closing control.

## The controls, because a clean result has to be distinguishable from an instrument that never fired

- **The emulator REFUSES the identical instruction, in the identical image.** `results/emulator-refuses.txt`:
  `domain halted by capability fault: cause = 24`, at `pc` offset `0x4354`, which disassembles to the
  `lbu` of the first probe. Cause 24 is defined in `capstone-qemu/target/riscv/op_helper.c` from the
  spec as *"`x[rs1]` is not a capability"*. So the architectural intent is a trap, and the probe
  demonstrably fires.
- **A LIVE alias through the same code path succeeds on the same silicon** (boot of image
  `aed492ab985653f3`, `mepc` landing on the later mint, not on the dereference). So the path is not
  broken in a way that would make everything succeed.
- **An earlier probe using `mrev` proved nothing and is recorded so nobody repeats it:** `mrev` requires
  `CAP_TYPE_LIN` and an alias is `NONLIN`, so it is a type error on *any* alias, live or stale. Both
  faulted; the detector separated neither.

## The two live hypotheses, and the one question that discriminates them

Both R-34's and R-24's fixes are ancestors of `054cea69b`. Therefore either:

**(a) The flashed bitstream is not built from `054cea69b`.** The board exposes **no digest** for a
resident image — `GET` returns 405, there is no download route, and `flash_state` carries only
state/name/epoch — so the image is known by filename alone. This project has been bitten by a label
naming different content before (R-29). If the flashed build predates `c77c65324`, R-34 and R-24 come
back as the explanation and nothing here is new.

**(b) The untagged-base check does not fire on the LSU path.** `UNEXPECTED_OPERAND` is raised in
`core/capstone_flu_unit.anvil.sv`; a plain `lbu`/`sb` through a capability base is handled by the
load/store unit, and whether that path checks the tag at all is a separate question from the two fixed
issues. If so this is a new, and severe, gap: revocation does not deny access.

**The discriminating question is for the board/RTL lane, and this folder cannot answer it:
was the currently flashed `caplifive_m1_054cea69b.bit` built from `054cea69b`, i.e. with `c77c65324`
in?** A yes makes (b) the answer and this a new defect. A no makes it an old one.

## Reproduce

    # the probe image: off-by-default guards, so no measured image moves
    OUT_DIR=<dir> DOMAIN_BASE_VA=0x410000 \
      R1_EXTRA_DEFS="-DM1_MAXRET=43296 -DM1_STALE_DEREF=1 -DM1_STALE_MINT=0 -DM1_STALE_WRITE=1" \
      bash capstone/sublet/r1/build-r1-silicon.sh        # -> 35fb3fec3196841b

    # emulator: must halt at cause 24, offset 0x4354  (the control that proves the probe fires)
    R1_DOM=<dir>/r1_slots_pools.dom R1_HOST=<sqlite_host_rr.user> OUT=<out> \
      bash capstone/sublet/r1/run-r1-qemu.sh \
      "--arm pressure --series m1 --pattern shared --cap 64 --budget 60000 --stale-take" 2097152

    # board: one invocation, drivers/lists/m1-staletake.txt, via board-r1e4.sh

`M1_STALE_MINT=0` is what makes the domain **return** instead of wedging; with the mint present the
fault destroys the output buffer and the only evidence is a latched `mepc`.

## What this is NOT evidence of

The bitstream is an integration branch whose timing does not close (WNS −8.307, 90,379 failing
endpoints). Nothing here separates a design property from an artefact of this build, and no
measurement in this folder was taken on a timing-clean image.
