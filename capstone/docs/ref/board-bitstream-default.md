# Board bitstream registry: the registered default and fallback

The single record of which sealed bitstream is the default for NEW board work and which is the registered fallback.
Under the delegated rule (CLAUDE.md "Autonomy"; the lead, 2026-11-21: decide between graded seals with the rtl and
runtime lanes, act, report), the default is agreed between those two lanes and recorded here by the board lane.
Pre-registered chains keep the bitstream they name, whatever this file says.

## Current (from 2026-11-28)

| role | bitstream | sha256 | seal record |
|---|---|---|---|
| **default** (new work) | `caplifive_…-r75d-b5b2n-crev4b-wap-flu-r80-r77_b42c8c644.bit` (the Line B Full seal, final candidate) | `6a09f34d50842e01deff19296c79c5e06021730ce0da37ec01d0fd86270b190a` | `docs/plans/2026-11-25-crevoke-4b-rtl.md`, "The final candidate b42c8c644: sealed and graded" |
| **fallback for Line B images** (recovery only) | `caplifive_…-r75d-b5b2n-crev4b-wap-flu-r80_a5c45a46d.bit` (b42c8c644 without R-77's PRE fix) | `9b46b14fb6252873696acc9762d7aecbc53271567447460207737c2366f1e36e` | `tests/rtl-smoke/b5b2-seals-board-2026-11-25/` |
| **fallback for legacy-format kits** (recovery only) | `caplifive_…-pagedtbl-r76-r75d_1c5d31474.bit` (R-75 (d), the previous default) | `2d2ba92c2e193dc4fcf368b9159ee454d2281c40d20a63ef9a57e6a7418eaf16` | `docs/plans/2026-11-22-r75d-gate.md` |

The full names are the seal records' (the feature prefix `caplifive_ids55-native-…-tagoverlap6c` is elided here).

**Two fallbacks, because the formats do not mix** (the rtl lane): Line B images (the trap-route kernel, Sublet tables,
CAPENTER-free monitors) do not run on 1c5d31474, which predates the Sublet format, and the legacy kits (classic monitor,
CSRUNV contexts) do not run on b42c8c644 or a5c45a46d.

**Both fallbacks are for recovery only** (the runtime lane, 2026-11-28, for both lines):
- a fallback means the board is recoverable and known to boot;
- a Line B session never continues on a fallback: if b42c8c644 does not come up, the session STOPS;
- it never moves onto another seal;
- Line B kits never go to the P1 line.

**What a5c45a46d has read on silicon, and what it has not.** Board 3, 2026-11-25: its bare Tier 1 kits, counted across the
session and the reruns of its loads that failed to start (`tests/rtl-smoke/b5b2-seals-board-2026-11-25/README.md`):
- b5b2n 20/21. The one MISS, `15-b5b2-cptag`, is reading 21's instrument artifact, already classified on dev (H-fetch).
- crevoke4b 16/16: every image read AS PREDICTED at least once.
- wapflu 1/1.
- R-80's fixed arm (ldcexc2) AS PREDICTED.

**It has no Linux boot on record.** The spectest kit's Amendment 7 pre-registered a SPEC session on it, but that session
moved to b42c8c644, and no board capture reads a5c45a46d back. So the Line B fallback has no proven Linux boot. A
recovery onto it has to establish one before anything relies on it.

**Named seals, not candidates.** Each runs only where a pre-registration names it:
- 14f2ef81a, +Spatial (b42c8c644 + CAPSTONE_SPATIAL_ONLY), the campaign's CS arm;
- 796af23d9, stage 2 (N), the compiler lane's stage-2 sessions; it does not replace b42c8c644 for the campaign (the
  Line B plan's 28 Nov audit, decision 3);
- 8db8f9cda, rvbase (`caplifive_rvbase-matched-ic32k-dtlb64_8db8f9cda.bit`), the Plain arm. A manual seal: its
  finish.verdict reads COLLECT-FAILED, because the riscv-base tree has no collector; SEAL.txt carries the seal line.

**Scope.**
- Every pre-registered chain keeps the bitstream it names.
- Nothing is reflashed for the change; it applies to new work only.

**Grounds**, agreed 2026-11-28 by the board lane (proposer), the runtime lane and the rtl lane:
- b42c8c644 is sealed and graded, and its Tier 1 kits read on boards 2 and 3 (`docs/plans/2026-11-25-crevoke-4b-rtl.md`,
  "on silicon: its Tier 1 kits on two boards"; records `tests/rtl-smoke/b5b2-seals-board-2026-11-25/`).
- The campaign's CF arm ran on it: the same image within 0.06 % on every stage across boards 1, 2 and 3.
- Open rows that touch b42c8c644 (the rtl lane, 2026-11-28); none blocks it as the default:
  - S-35, mcf's silent stop with an LDC at the LSU head: seen once, on 904cd9c6b, never on b42c8c644. R-80's fix is
    in b42c8c644, but its silicon reading does not discriminate: the unfixed 28fcbb646 control was never run, and the
    fixed arm was never read on b42c8c644 itself. Every campaign mcf row on b42c8c644 PASSES.
  - R-77: FIXED IN RTL in b42c8c644, not measured on silicon; no kit value moves with it.
  - R-84: the liveness query answers live for a retired generation. Its only Sublet sender is LCC selector 0, which
    grants no authority, and the access path excludes the retired word.
  - I-17: the JTAG debug-module busy timeout. Infra, seen on all seals.
  - Interpretation, not a defect: CREVOKE is interruptible and chunked on b42c8c644, so a kit that timed CREVOKE on an
    earlier line reads differently here.

## History

| from | default | fallback | agreed by |
|---|---|---|---|
| 2026-11-28 | b42c8c644 (Line B Full) | Line B images a5c45a46d; legacy-format kits 1c5d31474; both recovery only | board lane (proposer), runtime, rtl; delegated rule |
| 2026-11-23 | R-75 (d) 1c5d31474 | bitstream 1 9e801ab5e | runtime + rtl, delegated rule |
| 2026-11-21 | bitstream 1 9e801ab5e (P1 + R-76) | P1 d83d3a633 | the lead ("Yes, bitstream 1"), then delegated |
| before | P1 d83d3a633 | | |
