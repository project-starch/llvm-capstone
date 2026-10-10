# bbdce45fda1e — avcodec/diracdec: Enlarge mctmp to cover the worst-case blheight*ybsep + yblen rows

## The defect

`s->mctmp` is allocated with a height term of `h + MAX_BLOCKSIZE`. Motion compensation writes rows up to `blheight * ybsep + yblen`, which can exceed that, so `mc_row()`'s writes at `mctmp + y*rowheight` run past the allocation. Nothing checked that the allocation's worst case covered the writer's.

## Upstream defect

- **Fix:** `bbdce45fda1e`, *"avcodec/diracdec: Enlarge mctmp to cover the worst-case blheight*ybsep + yblen rows"*, `libavcodec/diracdec.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin allocates the wider margin. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
    s->mctmp     = av_malloc_array((stride+MAX_BLOCKSIZE), (h+MAX_BLOCKSIZE) * sizeof(*s->mctmp));
```

## The fix

```c
    s->mctmp     = av_malloc_array((stride+MAX_BLOCKSIZE), (h + 5*MAX_BLOCKSIZE) * sizeof(*s->mctmp));
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a plain
allocation because upstream's is — no pool is in the path.

**Reduced:** no Dirac stream and no motion compensation. The buffer is each arm's own row count and the row writes are a loop, with the first crossing row probed.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
