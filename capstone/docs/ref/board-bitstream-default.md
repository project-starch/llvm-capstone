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
- The runtime lane's perf null on (d): Addendum 219 part 2, read in Addendum 227 (spec-silicon cd8148f9554e,
  `ports/spec-cpu2017/results/2026-11-08-train/PREREG.md`), chain r75n-b2, kit bkit106, board 2. Smoke 9/9 on every
  boot and specdiff PASS on every row. Each row is inside bitstream 1's band (Addendum 204):
  - perlbench makerand 275.30 s (273.74-276.50);
  - gcc train01 1,762.22 s (1,759.78-1,766.84);
  - mcf 5,943.91 s (5,931.77-5,955.55).

## History

| from | default | fallback | agreed by |
|---|---|---|---|
| 2026-11-23 | R-75 (d) 1c5d31474 | bitstream 1 9e801ab5e | runtime + rtl, delegated rule |
| 2026-11-21 | bitstream 1 9e801ab5e (P1 + R-76) | P1 d83d3a633 | the lead ("Yes, bitstream 1"), then delegated |
| before | P1 d83d3a633 | | |
