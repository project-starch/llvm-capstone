# Paper board runs on R-42 (`caplifive_r42_6cbdaeeb4.bit`), from 2026-09-29

These are the board runs the Sublet paper's hardware protocols still need that R-42 can run. R-43
false-denies a live capability once a few hundred revocation ids are live, so only low-live-id
programs run here. Every result is R-42 evidence, labelled with its bitstream. None is compared with
another bitstream's number without a bridge: the R-43 redesign changes the load path, and R-45 is
fixed only there.

Each study is pre-registered in this file, and the pre-registration is pushed before its first boot.
The result lines are added beside it afterwards.

## H1 calibration and the linear family: three boots

**Why.** H1 step 5 asks for the timer overhead and the dependent-access latency over three fresh boots,
with median and range. E4 (§7v of the measurements doc) measured both in ONE boot, on
`caplifive_r30r31_1bfff7776`. Boot r42e3b read the linear family on R-42 once.

**What runs, identical in all three boots.** `board-r1e4.sh` with:
- `R1_IMG` = image `cbf8cb41eb56c477`, the E3b linear image, at 0x410000;
- host `2c9e82d101b48160`, and the stock `k800` control `b2d60e525f807ea4` at 0x10000, first and last;
- `R1_LIST` = `drivers/lists/h1-r42-calibration.txt`. That is `e4-calibration.txt` lines 1-4 (the
  latency chase with three timed traversals, then `--calib` in three fresh domains), plus the linear
  series as the last invocation. E4's line 5, the Sublet nodes regression, is **dropped**: it
  churns thousands of revocation ids, and on R-42 the same invocation false-denied (B3, cause 25).

**Emulator record** (`~/capstone-artifacts/h1-r42/`, the image and host copied there by hash). All
three invocation types returned `R1_RC=0`:
- the latency chase gives 15 `R1 lat` lines, 5 instructions per load at every size;
- `--calib` gives `cyc_cyc=1 cyc_cyc2=1 ret_ret=1`;
- the linear series gives 12 `R1 lin` lines (boot r42e3b's QEMU record).

The latency run mints nothing: `minted=0 split=5 mrev=5`.

**Pre-registered readings, per boot:**
- `k800` returns 4 twice with instret 1089, else the boot is VOID. Its cycles are recorded, not gated.
- **Latency:** cycles per dependent load are flat across three traversals. 4 KiB and 16 KiB read the same
  value (in the 32 KiB D$), and 256 KiB and 1 MiB read the same, larger value.
  - E4 on 1bfff7776 read 9.00 and 48.2, with 64 KiB at 40.8 between them.
  - **A uniform shift on R-42 is itself the result** and not a failure: R-35's fix put a revocation check
    on the load path, and R-42 is a different build.
  - Readings that would be new: 4 KiB differing from 16 KiB, a non-monotone curve, or traversals
    disagreeing by more than 0.1 cycles.
- **Timer:** `cyc_cyc` = 2 and `ret_ret` = 1 in each of the three domains, as E4 read.
- **Linear:** arms 0-11 read 7,1,7,7,7,1,7,1,0,1,0,1, as r42e3b did. Arms 8 and 10 read 0, so `tighten`
  and `shrinkto` copy (R-21). The controls read 1.
- No trap and no cause 25. There is one `speedtest1-ran=0x4EB1xxxx` per invocation.

**What these runs do not establish.** A number here is R-42's and nobody else's. It does not carry to
the R-43 redesign without that bitstream's own control pair. A boot whose control fails carries no
verdict, and neither does anything after a boot's first failure.
