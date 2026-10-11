# 8553e6ef57 — a payload buffer allocated without the padding its readers require

## The defect

`cbs_av1`'s ITU-T T.35 metadata handler allocated the payload buffer at **exactly** the declared
payload size. FFmpeg's bitstream readers are allowed to read up to `AV_INPUT_BUFFER_PADDING_SIZE`
(64) bytes past the logical end of a buffer — that is what the constant is for — so an exactly
sized allocation breaks the contract and the reader runs off the end.

The defect is in the **allocation**, not in the reader: the reader is doing what the API permits.

## Upstream defect

- **Fix:** `8553e6ef57`, *"avcodec/cbs_av1: pad the ITU-T T.35 payload buffer"*,
  `libavcodec/cbs_av1_syntax_template.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin carries the padded allocation and the `memset` of the
  padding. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

    current->payload_ref = av_buffer_alloc(current->payload_size);

## The fix

    current->payload_ref = av_buffer_alloc(current->payload_size +
                                           AV_INPUT_BUFFER_PADDING_SIZE);
    ...
    memset(current->payload + current->payload_size, 0, AV_INPUT_BUFFER_PADDING_SIZE);

## What is real here, and what is reduced

**Real:** the constant (FFmpeg's own 64) and the shape — one direct allocation sized to the logical
payload while a permitted reader reaches past it.

**Reduced:** no coded bitstream, no `CodedBitstreamContext`, no AV1 parse. The read is a single
labelled byte at the first offset past the payload, so the crossing is attributable to the probe
rather than to a word-at-a-time reader loop.

## What the run establishes, and what it does not

**Establishes:** the defect reproduces from the upstream fix differential, two-sided — the buggy
arm's read is past the allocation and the fixed arm's is inside, differing only by the padding term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is conditional on the usable size for a 24-byte request.
