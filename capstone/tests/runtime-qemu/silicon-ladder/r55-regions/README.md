# R-55 (regions): the FPGA monitor's region table under a full-scheme bounds encoder

`replay.py` replays the monitor's region table (`split_out_cap`, `create_region`, `create_domain`, the boot carves of
`cap_env_init`) through a recorded R-43 a2 carve sequence. It uses `../check-repr.py`'s literal port of the RTL
encoder. It reports:
- the slot id each region receives;
- OVERLAP: pairs of a carved domain block or region, and another live slot whose DECODED bounds meet it. Every live
  slot can be loaded into the CPMP for S-mode, so a nonzero count means Linux can reach that memory;
- pieces that are still inexact.

| run | expected |
|---|---|
| `python3 replay.py a2-0568f93a9.json --format cursorless` | every id of 0568f93a9's a2 run (14 tests), OVERLAP 0 (control) |
| `python3 replay.py a2-respin-afcd19bea.json --format full` | every id of the respin's a2 run (9 tests, table full at 96), OVERLAP 508 (control) |
| `python3 replay.py a2-0568f93a9.json --format full --policy exact --region2-end 0xC0000000` | the new policy: 111 slots, OVERLAP 0, no inexact piece |

- `extract.py` turns a board transcript (`boot-raw.txt`) into the JSON sequence.
- `edit-monitor.py` and `edit-wrapper.py` are the scripted source edits (capstone-sbi r55/exact-regions and
  r55/board-exact; caplifive-opensbi likewise).
- `gate-exact2.sh` compiles both through capstone-c and requires QEMU's assembly to be byte-identical.
- The plan doc's "R-55 (regions)" sections hold the pre-registration and the result.
