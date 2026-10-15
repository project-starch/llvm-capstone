# FFmpeg carved corpus: crossings between regions carved out of ONE allocation

FFmpeg's NESTED spatial row. Every case is an upstream fix for an access that left a region FFmpeg
had carved out of one allocation by pointer arithmetic -- `ubuf = buf + 18 * linesize`,
`quant_cof[c] = buffer + c * max_order`, VP9's `assign()` -- and landed in a neighbouring region of
the SAME allocation. Each case CHECKs that the whole unreduced overshoot stays inside the
allocation, so the driver refuses (exit 75) a reduction that would escape it.

Siblings: `../plain-heap-repros` (the access leaves the malloc bound itself),
`../subobject-repros` (it leaves a struct member), `../plane-repros` (one `av_frame_get_buffer`
block carved into planes), `../pool-repros` (storage an AVBufferPool handed out).

Thirteen cases from thirteen upstream fixes. Cases 0-11 were mined from the FFmpeg history, 2010-2017 (13
candidates were accepted; the CUDA one is excluded because its block is device memory no arm here can
run). **Case 12 joined on 2026-10-10 from `../subobject-repros` (its case 09)**: vf_thumbnail's
histogram, carved twice by arithmetic out of one `av_calloc`, which that corpus had filed NOT NESTED
against the carve rule this one follows. Twelve are fix-reversals. **Case 11 is LIVE at the n9.0.1 pin by source reading**: the per-type residue
bound its fix added was removed in 2014, and the pinned decoder accepts a legal stream that the
Vorbis I spec says to truncate -- see its `live_proof`; it has not been demonstrated on a stream.

## Shapes

| shape | cases | upstream | the crossing | consequence |
|---|:--:|---|---|---|
| a region carved one row shorter than its writer's row count | 0 | `85407c7e63` | writes U's tenth row, 9 bytes, onto vbuf's row 0 | V overwrites it; the field MC reads V's samples as U's |
| a loop bounded by a constant larger than the carved region | 1 | `699341d647` | writes 24 int32 past a 40-sample decoded[0], into decoded[1] | none on the mono path: decoded[1] is not read there |
| a read bounded by a filter order larger than the carved region | 2 | `cd7524fdd1` | reads 56 int32 past a 72-sample decoded[0], from decoded[1] | none: the loop that would use the delay line does not run when order >= length |
| a loop bounded by the prediction order rather than the block length | 3 | `55937bb4a7` | read-modify-writes 8 int32 past channel 0's samples, into channel 1's carry-over prefix | channel 1's next prediction starts from a decremented sample |
| a region carved for one element and used as a list | 4 | `cd09284924` | writes channel 0's terminating stop_flag into channel 1's 44-byte slot | channel 1's entry replaces the terminator, so channel 0's dependency list runs into channel 1's |
| a count read from the stream in a field wider than the region it indexes | 5 | `9d3032b960` | writes 11 int32 past channel 0's 20-word slice, into channel 1's | channel 0's prediction then uses channel 1's coefficients 20..30 |
| a region sized for one chroma subsampling and written for another | 6 | `2d0bea4719` | clears 16 bytes per superblock column past above_uv_nnz_ctx[0], into above_uv_nnz_ctx[1] | none: zeros, over a region the next statement clears anyway |
| a region carved for one element size and used at another | 7 | `2563a33856` | writes 128 bytes past a 128-byte intra_pred_data[0], into [1] | the next superblock row's luma intra prediction reads chroma samples |
| an offset computed before the stride it depends on was doubled | 8 | `b5ff61695f` | writes 48 bytes past U's 208, into V | V then overwrites U's tail, and the vertical scaler reads V's samples as U's |
| a 2-D block placed by a column offset that only fits a wide stride | 9 | `043bcdcdb0` | writes the prediction's last row, 16 bytes, onto src's first | encode_block() then encodes against a corrupted source row |
| a region carved at one stride and written at another | 10 | `d2213b6493` | writes direction 0's rows 8..15 onto tmp_b_block_y[1] after the linesize doubled | direction 1 overwrites them, and rv4_weight averages direction 1 with itself |
| an end validated against all channels' worth of samples for a per-channel decode | 11 | `68226ed9ec` | adds codevectors 128 floats past channel 0's slice, into channel 1's | channel 1's output carries channel 0's residue |
| an unshifted high-bit-depth sample indexing a 256-entry carved sub-slice | 12 | `ac59fc542f` | bumps one int 767 entries past plane 0's 1024-byte sub-slice, into thread 1's slice | a count lands in another thread's histogram, which merges into the frame's |

## Arms

The block is the platform's calloc/malloc, so every arm whose bound is the ALLOCATION -- Capstone
level0 and Sublet, CheriBSD with revocation, ASan's redzones -- is in
bounds for every crossing here. Field bounds (`capstone-subobject`, `cheribsd-subobject`) narrow a
pointer to a struct member, and a carved region is not one. The remedy is at the carving code:
`shared/driver.c`'s `ffc_carve()` narrows each region to its own extent under
`-DFFC_CARVE_BOUNDS` (`__builtin_capstone_cap_shrink` on Capstone, `cheri_bounds_set` on CHERI) --
the `*-carve-bounds` arms. Their fixed arms run under the same narrowing, so a region length that is
wrong would fault a FIXED arm instead of reading as a catch.

Runners: `runners/run-native.sh` (fix differential), `runners/run-asan.sh` (ASan with positive
controls), `runners/run-cheribsd.sh <arm>` (stock CheriBSD, with a carve control in
every boot), and `tools/run-capstone-domain.py` (Capstone arms).
