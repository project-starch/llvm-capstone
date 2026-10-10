# 041d4f010e — a stream-supplied header length consumed without checking the bytes left

## The defect

`prores_raw`'s `decode_frame` read `header_len` from the stream and checked only that it was **at
least** 62. Nothing checked it against the bytes actually remaining, so a stream declaring a header
longer than the packet made the parse walk off the end of the packet buffer.

The crossing's length is chosen by the input, which is why the fix bounds the length rather than
clamping an individual read.

## Upstream defect

- **Fix:** `041d4f010e`, *"libavcodec/prores_raw: Fix heap-buffer-overflow in decode_frame"*,
  `libavcodec/prores_raw.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin carries the two-sided guard. A fix-reversal.
- This candidate was named as *"Still unopened"* in `docs/ref/spatial-and-temporal-bug-inventory.md`;
  this case opens it.

## The vulnerable code, quoted from the fix's parent

    if (header_len < 62)
        return AVERROR_INVALIDDATA;

## The fix

    if (header_len < 62 || bytestream2_get_bytes_left(&gb) < header_len - 2)
        return AVERROR_INVALIDDATA;

## What is real here, and what is reduced

**Real:** the guard's shape — a lower bound with no upper one — and the fact that the length is
attacker-chosen.

**Reduced:** no ProRes stream, no `GetByteContext`. The packet is a plain allocation and the parse is
a byte loop. The loop carries **its own stop** 32 bytes past the end so that a run cannot wander:
the case measures *that* the bound was crossed, not how far, which matches the fix bounding the
length rather than the read.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the fix differential — the buggy arm crosses, the fixed
arm rejects the frame and never reads.

**Does not establish** any Capstone or CheriBSD reading, nor upstream reachability of a 70-byte
header on a 48-byte packet.
