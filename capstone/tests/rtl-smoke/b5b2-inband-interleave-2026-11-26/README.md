# The console interleave in SPEC in-band blocks, and its grader fix (pre-registered 2026-11-26, before it grades any row)

**The hazard** (raised by the runtime lane, 2026-11-26, from the nested lane's nested-corpus result, nested-workloads
0e8089933116). On the board's slow UART, a kernel printk (`[ <t>] sublet: fault …`) or a `TRAPTEST: tick` line can land
inside a program's output line. On the twin it does not. A SPEC stage prints its outputs in-band:
- `@@SPEC-FILE <arm> <name>` … `@@SPEC-FILE-END` holds gzip+base64 file bodies;
- `@@SPEC-HASH-BEGIN <arm>` … `@@SPEC-HASH-END` holds `<sha256>  <name>` lines.

This folder records what the reader does with such a line, how often it has happened, and the fix.

**What the current reader does.** The reader is spec-silicon's `run-stage.py read` (wt-spec, sha256 `384ed237…`, as
called by `grade-spec-chain.py` `4fa95eaf…`, "v1"). The cases come from `mk-splice-variants.py` on one clean board capture.
The results are in `controls.result-lines.txt`.
- **Inside a SPEC-FILE block, any interleaved line breaks the decode.** A whole line or a mid-line splice, tick or kernel,
  gives `binascii.Error`, `zlib.error` or `EOFError`. The stage reads MISS (`read rc=1`), which protection did not cause.
- **Inside a SPEC-HASH block, a whole line is dropped harmlessly** by the reader's 2-token filter.
- **A mid-line splice inside a SPEC-HASH block is silent.** The tail of the split line parses as `<truncated hash>
  <name>`: `t1.c` read `e750101d…` for `de7b86d7…`, and the stage still PASSes. Under the x264 criterion of spectest
  Amendment 13 (a data-hash match), that would read as a false MISS.

**How often** (`scan.result-lines.txt`, `scan-inband.py` `a0f8467d…`). The scan covers every board SPEC capture graded
before this commit and the +Spatial twin's x264b log. Five captures carry whole `TRAPTEST: tick` lines INSIDE a
SPEC-HASH block:
- b3chain2's B5T, B5TL0 and B5TL1-g2 (9 each, x264's hash block);
- SP-B5TL1-g4mk on board 1 (2, inside its 557-line hash block);
- the twin's x264b (1).

All of them are whole lines, so the reader dropped them and no reading changed. No foreign line sits in any SPEC-FILE
block, and no capture holds a mid-line splice. So the hazard is live: hash blocks long enough to outlast a 120 s tick
already catch ticks.

**The fix: `grade-spec-chain-v2.py`.**
- The runnable copy is `<mnt-hdd>/capstone/b5b2linux/grade-spec-chain-v2.py`, sha256
  `b976d2884c8c42d5bf85a24ce4d8507ccf8b0c489b192eff8d6965a834ab2f82`.
- The committed copy differs only in the host path on its `H = ` line (`<mnt-hdd>`).
- It is v1 plus one step; `v1-to-v2.diff` shows the whole change:
  - Inside SPEC-FILE and SPEC-HASH blocks only, every `[ <t>] …\n` and `TRAPTEST: …\n` is removed together with its newline,
    so a split line rejoins.
  - The stage's `step.txt` is the result, and `step.raw.txt` keeps the slice as captured.
  - gzip's CRC32 then checks every decoded file.
  - A line still foreign after that makes the stage **NO READING** (unreadable in-band block), never MISS. Foreign means not
    base64 inside a file block, or not `<64 hex>  <name>` inside a hash block.
  - `INBAND <stage>: k interleaved line(s) removed` is printed only when k > 0, so a clean capture grades byte-identically.
- Every other rule of v1, including Amendment 13's x264 rule, is unchanged.

**Controls** (`controls.result-lines.txt`).
- Positive: seven interleave shapes (A-F, H). Each is a v1 MISS or a v1 silent hash error. Under v2 each reads PASS, with
  `step.txt` byte-identical to the clean slice's and every hash equal to the clean read's.
- Negative: G replaces one base64 character with `$`. That is not an interleave, and v2 reads it NO READING.
- Not covered: a corruption that stays inside the base64 alphabet still fails gzip's CRC and reads MISS, as under v1.

**Identity** (`identity.result-lines.txt`). v1 and v2 were run on every real SPEC capture graded so far, plus the twin's
x264b log.
- 12 of 14 board captures give byte-identical output.
- Two (SP-B5TL1-g4mk, b3chain2-B5T-g2) and the twin differ only by the added `INBAND` line (2, 9 and 1 tick lines removed).
- Every verdict is equal.

**Adoption.** This folder is the pre-registration, and it grades nothing by itself. A row is graded by v2 only where the
PREREG that owns the row says so in an amendment pushed before that row is graded:
- the runtime lane's spectest Amendment 14, for its rows;
- the nested lane's amendment of its Full + ports PREREG (c4732c42a52b), for CF/CFP.

v1 is run beside it and recorded, and any verdict difference is reported row by row. Rows graded before such an amendment
keep their v1 verdicts.
