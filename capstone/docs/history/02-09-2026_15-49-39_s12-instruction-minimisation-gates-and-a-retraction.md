# S-12 instruction minimisation: three gate defects, one retraction, and where the campaign stands

## The retraction, first

**The "S-12 signature with no reload preceding it" does not exist. Withdrawn.**

It was recorded from board draw 3 of a ladder run as: a fault at `cincoffsetimm a4, s0, -0x100`,
`mcause 25`, `tval 0`, with a valid capability in the slot — a instance of the S-12 signature whose
operand came from the long-lived frame pointer rather than from a load. It was load-bearing: it was
the reason the pre-registered POS rule was widened away from a single fault address, and the reason
every load-adjacent mechanism family looked refuted.

Checked against the artifact it came from:

- draw 3 ran image sha256 `e6effb53…` = `min-aggressive.dom`, **not** the n=4 truncated build it was
  attributed to. Draws 1 and 2 of the same ladder ran `wrongbase.dom` (`7e6c33c7…`), a third image
  again. The three draws were never a controlled set.
- in that image the latched `mepc 0x828f4830` maps to VA `0x104830`, and the instruction there is
  **`[42] ldc a1, 0x0(a3)`** — not a `cincoffsetimm`, and not the instruction recorded.
- `min-aggressive.dom` has `[20] cincoffsetimm a3, s0, -0x90` NOPed out. `a3` was therefore never
  defined and held whatever it contained on entry. The fault is a load through an undefined base
  register: an artifact of a def-use-open patch in our own experiment.

So there remains exactly **one** well-attested S-12 signature: `cincoffsetimm a4, a4, 0xb0` at VA
`0x104814` consuming `a4` written by `ldc a4, 0x0(a0)` one instruction earlier, with the register
file and the source memory slot both holding a valid capability at the wedge. Nothing on the table
requires two mechanisms, and the load-adjacent families are **not** refuted. The POS rule should
revert to keying on the fault site.

What would have caught it: the claim named an instruction, and the instruction was never read back
out of the image the draw actually ran. Two of three draws in the set had a different sha256 than
the one assumed, which a per-draw sha check already printed and nobody compared.

## EJF: NEG-other, and why

`EJF` = NOP {2, 25…33}, 10 instructions. Three draws, all identical, all deterministic:

    mcause 25   mepc 0x828bb298   tval 0x82b9ce30   DBAS 0x82800000   ->  VA 0xcb298

VA `0xcb298` is `ldc a1, 0x10(a1)` in `sqlite3TableColumnToIndex` — a different function. Non-zero
`tval`, deterministic across draws, wrong site: **NEG-other**, and the cut broke the program rather
than testing it.

## Three gate defects, each of which had already passed something it should have refused

**1. `whole_function()` stopped at the first following symbol.** `.Lpcrel_hi8047` sits at `0x107450`,
*inside* `sqlite3WhereCodeOneLoopStart`, so closure analysis covered 2866 of the function's **4600**
instructions. Any use in the last 38% was invisible. Fixed to break only on a non-`.L` symbol.

**2. The register closure gate cannot see memory, and the whole window is memory.** Every candidate
instruction initialises a local, and a local is consumed through a frame slot. A frame-slot closure
gate was added: a removed store whose slot is read later, with no surviving store in between, is
refused.

**3. That new gate was inert, and returned clean.** Its `known` map seeded `{"s0": 0}` and then
processed the prologue, where `[3] movc s0, sp` redefines `s0` from an untracked base and **drops
s0 from the map**, which never recovers. It tracked **zero** frame accesses across 4600
instructions and passed E, F and G as safe. Only a positive control caught it — the counter is now
seeded after the prologue, the prologue shape is asserted, and *tracking zero accesses raises an
error instead of reporting a clean cut*. Working, it tracks 1120 accesses.

Also added: `[1]` and `[2]` are refused as callee-saved spills. Verified in the disassembly rather
than inferred — `[4596] ld ra, 0x7e0(sp)`, `[4597] ldc s0, 0x7d0(sp)`, single `ret` at `[4599]`.

## A fourth defect, in the verdict path itself

`DBAS` was extracted with `grep -ao "DBAS:…" | tail -1` over the whole log. Two things break that:
the console **replays the previous boot** on connect (one 3936-character line in `ejf-2.log` carries
two DBAS values from a different session), and UART output is **chunk-split at arbitrary
boundaries** — the test domain's line was emitted as `'DBA'` and then the rest, so neither piece
matched. It did not fail; it fell back to the **control domain's** DBAS, `0x82400000`, under which
the faulting `mepc` maps ~5 MB outside the image. That is the shape that converts a genuine
in-function positive into a NEG-other.

`probes/s12-verdict.py` reassembles the UART payloads, scopes to the test stage, and reports every
field as UNKNOWN rather than defaulting. On `ejf-2.log` it recovers `DBAS 0x82800000` and
`VA 0xcb298`, which matches the disassembly independently.

## Where the campaign stands

The exhaustive single-instruction sweep over `[1]`…`[33]` leaves **three** admissible cuts:
`[28] sw a4, 0x0(a5)` (slot `s0-0x10c`), `[30] sw a4, 0x0(a5)` (`s0-0x110`), and
`[33] stc a4, 0x0(a5)` (`s0-0x120`). All 30 others are refused.

`[33]` is the priority cut: it is the null-capability store immediately before the faulting
`ldc`/`cincoffsetimm` pair, i.e. the direct deletion test of the register-match correlation that
has been retracted and re-derived twice.

These three pass only because the frame-slot scan is **linear and ignores control flow**: each slot
is re-stored 520–1220 instructions downstream (`[755]`, `[740]`, `[1252]`), and whether those
stores execute on this input is a question about the program. That is the one direction in which
the gate is unsound — permissive. `probes/s12-funcgate.py` settles it empirically instead of
statically: run the variant under QEMU and require byte-identical result lines. QEMU never
reproduces S-12, which is exactly what makes it the right instrument — any difference it reports is
caused by the cut and nothing else. It is the gate that would have refused EJF before it cost three
draws.

The SQL input needs no further reduction: `dd2_join.test` is 13 lines — one `CREATE TABLE` of five
integer columns and one two-way self-join on an empty table. That, paired with the byte-identical
`.dom` (`69fe70b7…`), already is the handoff repro.
