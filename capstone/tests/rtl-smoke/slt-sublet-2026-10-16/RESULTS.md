# SQLLogicTest on the Sublet bitstream: results (2026-10-16 23:16 to 2026-10-17 02:12)

Pre-registered in README.md (lane 0d68d03ea85c, before the boots). **Every pre-registered reading holds: all seven
files match the native baseline on `caplifive_sublet_d93a8d66b.bit`, k800 4 in every boot, no refusal.** Result lines:
`results/slt-sublet.result-lines.txt`, one attempt per file, with its firmware hash and the runner's time.

| file | records | silicon `SLT-SUMMARY` up to ` completed` (= native) | runner |
|---|---:|---|---:|
| negctl | 23 | stmt 9 pass / 2 fail, query 7 / 5, skip_cond 2, parse_err 1 | 448 s |
| aggfunc | 80 | stmt 12 / 1, query 57 / 10, skip_cond 1 | 444 s |
| select1 | 1,031 | stmt 31 / 0, query 1,000 / 0 | 750 s |
| select2 | 1,031 | stmt 31 / 0, query 1,000 / 0 | 689 s |
| select5 | 1,436 | stmt 704 / 0, query 732 / 0 | 1,770 s |
| select3 | 3,351 | stmt 31 / 0, query 3,320 / 0 | 1,255 s |
| select4 | 3,857 | stmt 1,025 / 0, query 2,648 / 0, skip_big 184 | 4,621 s |

- **Both positive controls reproduce their native failures exactly.** negctl has 2 + 5 plus one parse error; aggfunc
  has 1 + 10. So a clean file is clean, not a comparator that cannot fire.
- **select4's 2,648 passing queries** include the 31 empty results that §7c's runner never compared. They are now
  compared on silicon too.
- **The three classes with reduced stacks** (s3 and s4 at 1,507,328 B, s5 at 458,752 B) completed their files on
  silicon as under QEMU.

## What did not run, and why: refusals before any board time

Ten attempts stopped before a boot, each at a gate, and none spent a board session. The driver's own entry-VA check
and the preflight both run before the JTAG load.
1. **The driver's entry-VA check** (negctl, aggfunc, select1, select2, first attempts). Another campaign's
   `speedtest1.dom` was in the overlay at 0x10000, beside the SLT domain. The driver now retires the other campaigns'
   overlay files for the boot and puts them back after.
2. **Preflight C16** (select5, select3, select4). The images had no QEMU pass record: `run-sqlite-slt.sh` validates
   but does not record. Each image was given its record from its own QEMU run, equal to native.
3. **The preflight's overlay hygiene gate** (negctl, aggfunc, select1, retry 1). Other campaigns' native and host
   binaries in the overlay. These are now retired too.
   - The same gate also refused select3/4/5's case files as "not used" by the boots that used them. It read only the
     first word of `--slt <path>`. Fixed on dev, 8134ad9a99f4, tested both ways.
   - select5's retry then passed with its 702 KB case file in place, which tests the fix end to end.

**Not established:** one draw per file, as §7c. This is a correctness run, so no cycle counts are read.
