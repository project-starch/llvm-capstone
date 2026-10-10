# 789d7b1b1dff — avcodec/ffv1dec: mask the fltmap index on the 8bit remap path

## The defect

`decode_plane` remaps decoded samples through `sc->fltmap[remap_index]`, a `uint16_t` map the function itself asserts holds `mask + 1` entries. The 8-bit path used the decoded sample as the index **unmasked**, while the two 16-bit paths in the same function apply `& mask`. A sample above `mask` — and a sample reaches 65535 — reads past the map.

## Upstream defect

- **Fix:** `789d7b1b1dff`, *"avcodec/ffv1dec: mask the fltmap index on the 8bit remap path"*, `libavcodec/ffv1dec.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin masks the index. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
                    sample[1][x] = sc->fltmap[remap_index][sample[1][x]];
```

## The fix

```c
                    sample[1][x] = sc->fltmap[remap_index][sample[1][x] & mask];
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a plain
allocation because upstream's is — no pool is in the path.

**Reduced:** no FFV1 bitstream and no planes. The map is a bare allocation of mask+1 entries and the index is a sample above mask, with the read reduced to the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
