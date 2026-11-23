# Board bitstream registry: the registered default and fallback

The single record of which sealed bitstream is the default for NEW board work and which is the registered fallback.
Under the delegated rule (CLAUDE.md "Autonomy"; the lead, 2026-11-21: decide between graded seals with the rtl and
runtime lanes, act, report), the default is agreed between those two lanes and recorded here by the board lane.
Pre-registered chains keep the bitstream they name, whatever this file says.

## Current (from 2026-11-23)

| role | bitstream | sha256 | seal record |
|---|---|---|---|
| **default** (new work) | `caplifive_…-pagedtbl-r76-r75d_1c5d31474.bit` (R-75 (d)) | `2d2ba92c2e193dc4fcf368b9159ee454d2281c40d20a63ef9a57e6a7418eaf16` | `docs/plans/2026-11-22-r75d-gate.md` |
| **fallback** | `caplifive_…-pagedtbl-r76_9e801ab5e.bit` (bitstream 1) | `01707b372e93ab1c49890a8917395d28dc7dc6addc8f8ade44fa275ff1486f83` | `docs/plans/2026-11-20-bitstream-1-full-build.md` |

The full names are the seal records' (the feature prefix `caplifive_ids55-native-…-tagoverlap6c` is elided here).

**Scope.**
- Paper chains stay on P1 (d83d3a633).
- Every pre-registered chain keeps the bitstream it names.
- Nothing is reflashed for the change; it applies to new work only.

**Not candidates.**
- The B5b-1 seal `…-pagedtbl-r76-r75d-b5b1_d0afdb40f.bit` (`docs/plans/2026-11-23-b5b3-full.md`) waits on two things.
  - Its CSMINT kit must read.
  - It uses the Sublet format: a different id format and table. So it also waits on the native route moving to Sublet
    (the rtl lane's correction, 2026-11-23).

**Grounds**, agreed 2026-11-23 by the runtime lane (proposer) and the rtl lane (confirmed directly to the board lane):
- R-75 FIXED ON SILICON on (d): board 2, booking 1437, `docs/plans/2026-11-22-r75kit-board2.md`; ISSUES at dev
  819780239744.
- Its seal PASS on every key against bitstream 1's (`2026-11-22-r75d-gate.md`).
- The runtime lane's perf null on (d) in bitstream 1's band on all three rows (its Train PREREG Addendum 219 part 2;
  smoke 9/9; specdiff PASS). The board lane has not read that reading: the addendum's pre-registration is on
  spec-silicon (612b946c6883), and the part 2 reading was not on a pushed branch when this was written.

## History

| from | default | fallback | agreed by |
|---|---|---|---|
| 2026-11-23 | R-75 (d) 1c5d31474 | bitstream 1 9e801ab5e | runtime + rtl, delegated rule |
| 2026-11-21 | bitstream 1 9e801ab5e (P1 + R-76) | P1 d83d3a633 | the lead ("Yes, bitstream 1"), then delegated |
| before | P1 d83d3a633 | | |
