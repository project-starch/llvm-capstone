# Three FFmpeg sub-object defects reproduced, and no configuration catches any of them

**`runners/run-native.sh` exit 0.** All three fixed arms print `VERDICT FIXED`; all three buggy arms
print `VERDICT DEFECT-REPRODUCED`. Result lines in `result-lines.txt`.

| case | upstream | buggy arm | fixed arm |
|---|---|---|---|
| 0 cbs_h265 pic_timing | `8864fd0aec` | DEFECT-REPRODUCED | FIXED |
| 1 Vulkan HEVC ref sets | `68845e26f7` | DEFECT-REPRODUCED | FIXED |
| 2 Vulkan HEVC DPB | `e058af88ab` | DEFECT-REPRODUCED | FIXED |

The oracle is the **upstream fix**, because it is the only one available: every crossing here is
between two members of ONE allocation, so a per-allocation bound is *in bounds* for it. That is the
finding, not a shortfall — `partial²` in the taxonomy, where CHERI and Capstone already share a
verdict.

## ASan is blind to this class, measured two-sided

`asan-probe.c`, two arms over the same struct:

| arm | what it writes | exit | ASan |
|---|---|---|---|
| `inside` | index 600 of `uint16_t[600]`, i.e. the next member | 0 | **no report** — and the neighbour reads back `0x00004141`, so the write did happen |
| `past` | one `uint16_t` past the whole allocation | 1 | **`heap-buffer-overflow`** |

**The positive control is why the first arm's silence means anything.** The first build of this
probe had *both* arms silent — at `-O1` the dead stores were optimised away, so the control could
not fire and "ASan is blind" would have been an artefact of the compiler rather than a fact about
ASan. Rebuilt at `-O0` with `volatile` and a read-back, the control fires and the blindness stands.

## What this does and does not establish

- **Does:** the three defects are real, reproduce from the upstream fix differential, and are
  invisible to per-allocation bounds and to ASan.
- **Does not:** measure the Capstone or CheriBSD arms. Those are **declared completions** in each
  `case.json` and measuring them needs a domain build this corpus does not have.
- **Does not** establish upstream reachability: cases 1 and 2 are Vulkan-hwaccel paths and case 0 is
  reached through the `h265_metadata`/`trace_headers` BSFs or the H.265 encoders.
- **N = 1 per cell.**

A process note worth keeping: the runner runs the **control (fixed) arm first** and treats its
failure as infrastructure rather than data, and that is what caught the first version of case 2 — it
wrote all sixteen extra entries the `DPB[32]` walk permits, ran 128 bytes into a 64-byte member and
so left the allocation entirely. The containment check refused it, and the case was reduced to the
first crossing.

Files: `result-lines.txt`, `inputs.json`, `asan-probe.c`.
