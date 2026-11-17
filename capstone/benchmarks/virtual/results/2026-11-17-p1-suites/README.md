# The embedded suites on P1: all 103 programs, T1 and T3, native route

Bitstream P1 (caplifive_…_d83d3a633, monitor 22a907f), board 3, kit bkit91, seven boots, 2026-11-17. This is the
suites' rerun on the prototype of the paper's SPEC section, so that every Sublet number comes from one prototype.
Bitstream 3 (`../2026-10-27-t3all-bitstream3/`, `../2026-10-28-t3heap-bitstream3/`) is the correctness reference
only. No P1 ratio is compared with a bitstream-3 ratio: the bitstream, the SDK revision and the instrument all
changed (see the pre-registration).

| kit | processes | programs | window (+08) | kit SHA256SUMS (first 16) | firmware (first 16) | cpio (first 16) |
|---|---|---|---|---|---|---|
| p1k1 | 54 | 5 | 18:01–18:13 | 2d51a8aa258c90fa | 05c201827debc0c8 | 8e2e31baa531bc90 |
| p1k2 | 40 | 5 | 18:13–18:25 | b3f7add3fa7bb4cb | 3ae9afdde6e3a7cf | 8e15c437a67786b3 |
| p1k3 | 40 | 5 | 18:25–18:36 | 6bb43c3d8ecac3c0 | 92a3c515d92bd90c | 4df6c55112d71eb0 |
| p1a1 | 190 | 22 | 18:36–19:03 | 14173a38479c49d3 | 27549e74be170901 | 73ca68ffcb402be0 |
| p1a2 | 190 | 22 | 19:03–19:32 | b05de35451c85be7 | ce8b1c4d66f1a853 | 0226ab245d397117 |
| p1a3 | 190 | 22 | 19:32–19:59 | 6d3c330b183c6b59 | f723b7b7edcbde59 | e10cc46eefa6d4b0 |
| p1a4 | 176 | 22 | 19:59–20:27 | 8730fe0f733cafd9 | b04eafd8dd9457a0 | 5406a134897b0800 |

**Kits.**
- `perf/p1suite/kits/<kit>`, made by mk-p1kits.py from the bitstream-3 kits.
  - Each keeps its source kit's step order with every T2 row dropped, and the same W/I.
  - 880 processes in total.
  - Per the pre-registration, T1 is SDK FRCDKMSspa3 and T3 is FRCDKMSa3, both compiled with llvm-c9 29027126.
- The SHA256SUMS hashes are the pre-registered ones. Each still verifies on disk. Each boot's bake manifest lists
  exactly the kit's 55 or 59 files, plus the 5 files of the native harness.
- bkit91 is `t3-smoke/bkit91-k2n-native-p1-tree-cscfix`, with SHA256SUMS bcc44a118431d470 over 30 files. It verifies.
- dtb 6c6e8bce4e324fd7 and native kernel config 328179a606277155 are the same on every boot.
- Firmware, Image and native kernel hashes differ per boot only because each boot's image embeds its own cpio.

**Pre-registration.** PREREG-t1-t2-sampling.md on spec-silicon has three entries, all on 2026-11-17 (+08):
- **1bb843bef3cc**, "All 103 programs on P1, T1 and T3, native route", committed 16:47, before any of these boots.
- **Erratum 44ced8f013bb**, committed 16:56, before any boot. It withdraws the per-process cross-check
  (BENCH cycles / run_cycles ≥ 1), which was ill-posed. A descriptive reading replaces it, below.
- **Deviation 225fcfbbc8b7**, committed 18:47, after p1k1-3 landed. It records the parsing repair below; the gates
  and bands are unchanged.

**Gates.** Each kit passed both pre-registered gates:
1. p1-check.py exits 0 (`check-<kit>.txt`). Every step had:
   - rc=0, `source=context-csr rdcycle_rdinstret=ok` and `counters=present`;
   - exactly W+I BENCH lines with cycles, instret and ns > 0;
   - no VERIFY-FAILED, and `BENCH-DONE failed=0`;
   - native_counters advancing.

   check= is one value across the two arms, and it equals bitstream 3's for 103 of 103 programs.
2. The boot's native smoke passes: "PASS: all 9 steps" on all seven boots. The smoke's "task: the two accounting
   checks WAIVED (K-min)" line is the kit's standing waiver. The boot record's "SPEC VERDICT … NO RESULT" is the
   verdict for a SPEC stage, which these kits do not have.

parse-board.py read 880 processes and 4,840 iteration rows, with no mid-line heartbeat.

**Parsing repair: kernel lines spliced into BENCH lines.** On the board console, the capstone_vm module's release
line (`[ t] capstone_vm: release retires …`) landed inside BENCH and BENCH-DONE lines. The gate as first committed
(07d3f7279267) therefore failed p1k1-3. p1-suite-native **aacc0105e681** (with 5a98f54fbed1, which prints the count)
fixes the parsing, before the unchanged gate reads the text:
- Each timestamped kernel line that starts mid-line is removed together with its newline, which rejoins the user
  line. Native-smoke's `@@NS-DMESG` quote lines are exempt, because they quote dmesg on purpose.
- p1-check.py and parse-board.py both print their count.

Fragments removed per kit, as the two scripts report them:

| kit | p1-check (gate) | parse-board | of which after a BENCH-DONE line's CR |
|---|---|---|---|
| p1k1 | 38 | 38 | 0 |
| p1k2 | 24 | 24 | 0 |
| p1k3 | 24 | 24 | 0 |
| p1a1 | 92 | 86 | 6 |
| p1a2 | 114 | 114 | 0 |
| p1a3 | 135 | 129 | 6 |
| p1a4 | 96 | 75 | 21 |
| total | 523 | 490 | 33 |

- Every one of the 523 is a capstone_vm release line. Every one sits in a BENCH or BENCH-DONE line after the first
  @@PERF-BEGIN; host prefixes such as `BE` and `BENCH-DO` are those lines cut mid-word.
- **Why the two counts differ by 33.** In 33 cases the kernel line arrived between a complete BENCH-DONE line's CR
  and its LF.
  - p1-check removes CRs first, so it sees the kernel line as mid-line and removes it.
  - parse-board reads with universal newlines, which turn the lone CR into a line end. The BENCH-DONE line is
    already whole, and the kernel line stays a line of its own, which parse-board ignores.
  - Both readings give the same BENCH lines.

Controls for the repair:
- `../../p1/test-p1-check.sh` builds a synthetic transcript:
  - a spliced copy PASSes, with TSVs identical to the clean copy's;
  - a deleted BENCH-DONE still FAILs;
  - the pre-repair scripts fail the spliced case.
- Ten bitstream-2/3 transcripts parse byte-identically with the old and the new parser.
- The seven P1 QEMU transcripts show 0 fragments.

**Files.**
- `runs-<kit>.tsv.gz`: parse-board.py's per-iteration rows, warm-ups included. `steps-<kit>.tsv` is each kit's
  process order, the STEPS file p1-check.py read.
- `runs-<kit>.tsv.ncnt.tsv`: per process, the native_counters deltas beside the summed BENCH cycles.
- `check-<kit>.txt`: p1-check.py's output against the bitstream-3 references (`runs-t3k<n>` for p1k<n>,
  `runs-t3a<n>` for p1a<n>).
- `T1-T3/`: perf-analyze.py `report --arms T1,T3` on the 15 allocating programs (p1k1-3).
- `all103/T1-T3/`: the same over all seven kits, with the usual suites (beebs, beebs-libc, rv8, coremark, ctl).
- `groups/T1-T3/`: the same with beebs-libc folded into beebs, so that BEEBS is one group of 93 programs, as on
  2026-10-28.
- `TRANSCRIPTS.sha256`: sha256 and size of each boot's board.txt, the text both scripts read, and of the
  board-raw.txt it was reassembled from. The transcripts are not committed.

**Result. Every pre-registered band is met, and no refutation fires.** T3/T1 cycles, geometric mean, 95 % CI
(`groups/` for BEEBS, `all103/` for the rest):

| group | programs | T3/T1 | band (pre-registered) |
|---|---|---|---|
| BEEBS (upstream + libc malloc) | 93 | 1.0063 [1.0059, 1.0067] | ≤ +1.5 % |
| — of which upstream | 80 | 0.9999 [0.9996, 1.0002] | within ±0.5 % |
| — of which libc malloc | 13 | 1.0465 [1.0440, 1.0490] | +0.5 % to +8 % |
| RV8 | 7 | 1.0000 [0.9983, 1.0015] | within ±0.5 % |
| CoreMark | 1 | 1.0003 [0.9990, 1.0016] | within ±0.5 % |
| controls (ctl) | 2 | 1.0597 [0.9841, 1.1407] | none |

Each interval lies inside its band, not only the point estimate.

Refutations:
- **No upstream program has a T3/T1 interval that excludes 1 by more than 1 %.** Three upstream intervals exclude
  1 by less:
  - fac 0.9992 [0.9986, 0.9998];
  - miniz 1.0008 [1.0003, 1.0012];
  - ctl-string 0.9872 [0.9750, 0.9996].
- The same rule applied to the 13 libc programs fires on 8 of them, so it can fire.
- No libc group reads below 1.000.

Per-program notes, descriptive only:
- The libc group splits in two:
  - five programs read about 1.000: dtoa, huffbench, miniz, trio-snprintf and trio-sscanf, all -libc;
  - eight read 1.047 to 1.124: the three ctl-*-libc and the five sglib-*-libc.
- The sglib programs execute 0.25–0.4 % fewer instructions under T3 (ins_ratio 0.9962–0.9975) while taking 5–12 %
  more cycles.
- ctl-empty reads 1.1233 [0.9976, 1.2649], the widest interval in the run.
- 15-program view (`T1-T3/`): beebs-libc (10) 1.0608 [1.0574, 1.0640] and rv8 (5) 1.0006 [0.9988, 1.0023].

**Replacement reading from the erratum (descriptive, no prediction, no gate).** Per process, the summed timed
BENCH cycles over the process's run_cycles delta, from the `.ncnt.tsv` files:

| arm | processes | min | p10 | median | p90 | max | above 1 |
|---|---|---|---|---|---|---|---|
| T1 | 440 | 0.0006 | 0.8099 | 0.8330 | 0.8368 | 1.0026 | 4 |
| T3 | 440 | 0.0007 | 0.8154 | 0.8331 | 0.8362 | 1.0008 | 2 |

**Regeneration.** Run from this directory. The output reproduces the three report directories (17 files) byte
for byte. This was checked by rerunning in a fresh directory that held only the seven runs-*.tsv.gz.

    V=../..
    lab='BEGIN{OFS="\t"} {if (NR>1 && $3=="beebs" && $4 ~ /-libc$/) $3="beebs-libc"; print}'
    for k in p1k1 p1k2 p1k3; do zcat runs-$k.tsv.gz | tail -n +$([ $k = p1k1 ] && echo 1 || echo 2); done > run.tsv
    for k in p1k1 p1k2 p1k3 p1a1 p1a2 p1a3 p1a4; do zcat runs-$k.tsv.gz | tail -n +$([ $k = p1k1 ] && echo 1 || echo 2); done > all.tsv
    awk -F'\t' "$lab" run.tsv > main.tsv
    awk -F'\t' "$lab" all.tsv > all103.tsv
    awk -F'\t' 'BEGIN{OFS="\t"} {if (NR>1 && $3=="beebs-libc") $3="beebs"; print}' all103.tsv > groups.tsv
    for x in "main.tsv T1-T3" "all103.tsv all103/T1-T3" "groups.tsv groups/T1-T3"; do set -- $x
      python3 $V/perf-analyze.py report $1 --out $2 --arms T1,T3 --boot 10000 --seed 20261026; done

The rows themselves come from the transcripts through `../../p1/analyze-p1.sh` at 5a98f54fbed1. That script runs
p1-check.py and parse-board.py per kit, and then the steps above. Rerunning p1-check.py on the hashed transcripts
reproduces the seven check-<kit>.txt byte for byte.
