# 68226ed9ec — a type-0/1 residue may end past one channel's blocksize/2, so a one-channel submap adds into the next channel's residue

## The defect

A type-0/1 residue may end past one channel's blocksize/2, so a one-channel submap adds into the next channel's residue.

## Upstream defect

- **Fix:** `68226ed9ec` (`68226ed9ecef675895dc55a0c58d587014639a0e`), subject: "vorbis: Fix decoder bug.".
- **Carved object:** vc->channel_residues, one av_malloc(blocksize[1] / 2 * channels floats) at libavcodec/vorbisdec.c:952, carved per submap by ch_res_ptr (:1481, :1560-1562).
- **Consumer:** libavcodec/vorbisdec.c:681-683 (setup) and :1359-1364 (type-1 write) at the fix's parent.
- **The crossing:** adds codevectors 128 floats past channel 0's slice, into channel 1's; it stays inside the allocation.
- **Consequence:** channel 1's output carries channel 0's residue.
- **Live at the `n9.0.1` pin:** YES, by source reading — LIVE, by reading the pinned source, two-sided, and adversarially audited (claim-auditor, 2026-10-09); NOT demonstrated on a stream. The per-type end check this fix added (`res_setup->type == 2 ? ... : 1`) occurs 0 times at n9.0.1: it was removed by 0a266cb55a (2014), and `git log -G'(vr|res_setup)->end' 0a266cb55a..n9.0.1 -- libavcodec/vorbisdec.c` is empty while the same query over all history lists both commits. The setup check left (n9.0.1:libavcodec/vorbisdec.c:724-725) bounds begin <= end and the partition count only. The one runtime bound is `max_output = (ch - 1) * vlen + vr->end` against `ch_left * vlen` (:1425-1449), which for a one-channel submap with two channels left accepts an end of up to 2 * vlen, and the type-0/1 loops (:1482-1506) then write vec[voffs] up to begin + ptns_to_read * partition_size - 1, into the next channel's slice of channel_residues (one av_malloc_array, :1019) and never past it. The Vorbis I spec, section 8.6.2, has the decoder clamp residue_end to blocksize/2 for formats 0 and 1, so such a stream is LEGAL and a conforming decoder truncates. Conditions: a type-0/1 residue on a submap with fewer channels than remain, begin + ptns_to_read * partition_size > vlen, and a neighbour whose floor is used (an unused one is cleared at :1743-1745 before the MDCT). The audit refined the history: the 2012 check bounded end by blocksize[1]/2 while the write uses the CURRENT block's vlen, so short-block packets could cross even while it stood. What would settle it dynamically: a 2-channel stream with two one-channel submaps whose first residue is type 1 with end = 2 * vlen, decoded by an instrumented n9.0.1, against the same stream with end = vlen.

## The fix's own diff, quoted from upstream

```diff
diff --git a/libavcodec/vorbisdec.c b/libavcodec/vorbisdec.c
index 0d4b717e55..8778152264 100644
--- a/libavcodec/vorbisdec.c
+++ b/libavcodec/vorbisdec.c
@@ -679,7 +679,7 @@ static int vorbis_parse_setup_hdr_residues(vorbis_context *vc)
         res_setup->partition_size = get_bits(gb, 24) + 1;
         /* Validations to prevent a buffer overflow later. */
         if (res_setup->begin>res_setup->end ||
-            res_setup->end > vc->avccontext->channels * vc->blocksize[1] / 2 ||
+            res_setup->end > (res_setup->type == 2 ? vc->avccontext->channels : 1) * vc->blocksize[1] / 2 ||
             (res_setup->end-res_setup->begin) / res_setup->partition_size > V_MAX_PARTITIONS) {
             av_log(vc->avccontext, AV_LOG_ERROR,
                    "partition out of bounds: type, begin, end, size, blocksize: %"PRIu16", %"PRIu32", %"PRIu32", %u, %"PRIu32"\n",
@@ -1483,6 +1483,7 @@ static int vorbis_parse_audio_packet(vorbis_context *vc)
     uint8_t res_chan[255];
     unsigned res_num = 0;
     int retlen  = 0;
+    int ch_left = vc->audio_channels;
 
     if (get_bits1(gb)) {
         av_log(vc->avccontext, AV_LOG_ERROR, "Not a Vorbis I audio packet.\n");
@@ -1557,9 +1558,14 @@ static int vorbis_parse_audio_packet(vorbis_context *vc)
             }
         }
         residue = &vc->residues[mapping->submap_residue[i]];
+        if (ch_left < ch) {
+            av_log(vc->avccontext, AV_LOG_ERROR, "Too many channels in vorbis_floor_decode.\n");
+            return -1;
+        }
         vorbis_residue_decode(vc, residue, ch, do_not_decode, ch_res_ptr, blocksize/2);
 
         ch_res_ptr += ch * blocksize / 2;
+        ch_left -= ch;
     }
 
 // Inverse coupling
```
