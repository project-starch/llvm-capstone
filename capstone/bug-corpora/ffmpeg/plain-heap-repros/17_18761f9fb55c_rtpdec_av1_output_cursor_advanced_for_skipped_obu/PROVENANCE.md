# 18761f9fb55c — avformat/rtpdec_av1: fix buffer overflow due to variable confusion

## The defect

When a temporal-delimiter or tile-list OBU is skipped, the reassembly advanced the **output** cursor `pktpos` instead of the **input** cursor `buf_ptr`. No bytes are written for the skipped OBU, yet the destination offset jumps forward by `obu_size`, so the writes that follow land past the end of the allocated packet.

## Upstream defect

- **Fix:** `18761f9fb55c`, *"avformat/rtpdec_av1: fix buffer overflow due to variable confusion"*, `libavformat/rtpdec_av1.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin advances the input cursor. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
            if ((obu_type == AV1_OBU_TEMPORAL_DELIMITER) ||
                (obu_type == AV1_OBU_TILE_LIST)) {
                pktpos += obu_size;
                rem_pkt_size -= obu_size;
```

## The fix

```c
                buf_ptr += obu_size;
                rem_pkt_size -= obu_size;
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a plain
allocation because upstream's is — no pool is in the path.

**Reduced:** no RTP packets and no AV1 OBUs. The two cursors are upstream's, the skip is upstream's, and the next write is the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
