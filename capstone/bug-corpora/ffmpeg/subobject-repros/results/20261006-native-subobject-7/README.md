# Seven more FFmpeg sub-object defects, and ASan measured blind across all six of their shapes

**`runners/run-native.sh` exit 0** over all **ten** cases: every fixed arm prints `VERDICT FIXED`,
every buggy arm `VERDICT DEFECT-REPRODUCED`. Result lines in `result-lines.txt`. The control (fixed)
arm runs first and its failure is treated as infrastructure, not data.

| case | upstream | the crossing | live at `n9.0.1` | buggy / fixed |
|---|---|---|:--:|---|
| 3 | `89de2f0de1` | reads `last[512]` of a `uint8_t[512]` into `last_len`'s low byte | no (fix-reversal) | `term=0x5a00` / `0x0000` |
| 4 | `1a00ea51cb` | `strlen("")-1` wraps to `SIZE_MAX`; the read lands one byte *below* `control_url`, at offset 28 | no | `saw=0x5a` / never formed |
| 5 | `d29ff88422` | writes `tile_sizes[256]` of a `uint32_t[256]` onto `std_ref`'s first 4 bytes | **yes** | `touched=257 out_of_member=1` / `256, 0` |
| 6 | `a809a784ec` | an unbounded `j` walks 8 words off `entry_point_start_ctu` into the enclosing struct | **yes** | `entries=72 past=8` / `64, 0` |
| 7 | `275e217b10` | a copy sized by its *source* length runs 16 bytes past `key_uri` into `key_string` | no | `size_arg=80 past=15` / `64, 0` |
| 8 | `fb862976df` | a guard keyed to a counter the loop never updates; 6 writes past `col_width_val` | no | `columns=26 past=6` / `20, 0` |
| 9 | `ac59fc542f` | a 10-bit sample indexes a 256-entry **carved** sub-slice at 1023 | no | `past_entries=768 in_allocation=1` / index 255 |

**Liveness was read from the pinned source, two-sided, not from ancestry.** For each case the
`n9.0.1` tree was grepped for both the vulnerable construct and the fix marker; every one of the
seven matched exactly one side, so the probes are known to fire. `git merge-base --is-ancestor` was
deliberately not used — it has called backported fixes live before. **Five of the seven are
fix-reversals, and that is not a weaker kind of case:** liveness is *recorded, never required*, and
treating it as a requirement is what held this corpus at three cases.

## ASan is blind to all six shapes, measured two-sided with a firing control

`asan-probe-7.c`, one arm per shape over one `calloc`, built `-O0 -g -fsanitize=address`. Readings in
`asan-arms.txt`:

| arm | the shape it crosses | exit | ASan |
|---|---|:--:|---|
| `member` | write one element past an array member | 0 | **silent** — and `neighbour=1094795585` (`0x41414141`), so the write happened |
| `member-read` | read one element past an array member | 0 | **silent** — `read=90` (`0x5A`), the neighbour's sentinel |
| `underflow` | index underflows out of a member's start | 0 | **silent** — `read=90` |
| `walk` | an unbounded loop walks off a member | 0 | **silent** — `past=8` |
| `copy` | a copy sized by its source length | 0 | **silent** — `past=15` |
| `slice` | a carved sub-slice crossed by a data index | 0 | **silent** — `bumped=1` |
| **`past`** | one `uint32` past the **whole allocation** | 1 | **`heap-buffer-overflow`** |

**The control is why the six silences mean anything**, and each silent arm also *reports the value it
crossed into*, so none of them can pass by the access simply not happening.

**`-O0` is load-bearing**, for the reason the 2026-10-05 bundle records: at `-O1` both arms of the
original probe were silent because the dead stores were optimised away, so the control could not fire
and "ASan is blind" would have been an artefact of the compiler rather than a fact about ASan.

**Allocation here is plain `malloc`, not the port's arena** — deliberately. Inside a case binary the
driver hands `av_malloc` one arena, so ASan sees a single allocation and could not see an allocation
bound even in principle; a silence there would prove nothing. With `malloc` there is a real redzone at
the edge, which the `past` arm demonstrates, so the other six silences are about intra-object
granularity and nothing else.

### A detector error worth recording, because it inverted the result

The probe's **first** run reported a detection on **all seven arms**, including the control-free ones
— which would have read as "ASan catches every sub-object crossing", the opposite of the truth. Two
compounding causes:

1. the probe never `free`d its allocation, so **LeakSanitizer** fired at exit on every arm;
2. the detector grepped for the string **`AddressSanitizer`**, which LSan's own summary line
   (`SUMMARY: AddressSanitizer: 356 byte(s) leaked`) also contains.

So a *positive* reading came out of a check keyed to the wrong spelling — the mirror of a clean zero,
and it needed the same suspicion. What caught it was carrying the **report kind** alongside the
pass/fail column: only `past` ever said `heap-buffer-overflow`. The probe now frees before reporting
and the check is keyed on `heap-buffer-overflow`; `leaks=0` on every arm is recorded in
`asan-arms.txt` as the two-sided confirmation that the leak path is gone.

## What this does and does not establish

- **Does:** seven more defects are real, reproduce from the upstream fix differential two-sided, and
  are invisible to ASan across six distinct crossing shapes.
- **Does NOT measure the Capstone, PoisonCap or CheriBSD arms.** Those are **declared predictions** in
  each `case.json`, recorded before any run so a refutation stays visible. The Capstone seam exists
  and is known — probe cases 40-42 of `ports/ffmpeg/buffer-pool/security-tests`, continuing at 43 —
  but it has not been used for these seven.
- **A caveat those predictions must carry:** the port's `av_malloc`
  (`src/shared/metadata-allocator.c:26`) is an un-narrowed bump/freelist carve from one arena that
  rounds every request to 64 bytes, and `__builtin_capstone_cap_shrink` appears only in
  `src/capstone-domain/payload-capabilities.c:57`, on pool *payload* blocks. So on this harness there
  is **no per-allocation bound on the struct at all**, and a completion is weaker evidence than "the
  bound is the whole allocation" would be.
- **Does not establish** upstream reachability. Each `PROVENANCE.md` names the entry point it believes
  reaches the consumer and says it is a belief, not a measurement.
- **N = 1 per cell.**

Files: `result-lines.txt`, `asan-arms.txt`, `asan-probe-7.c`, `inputs.json`.
