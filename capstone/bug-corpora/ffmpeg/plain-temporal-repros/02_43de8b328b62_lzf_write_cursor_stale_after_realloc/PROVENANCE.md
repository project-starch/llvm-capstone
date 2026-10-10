# 43de8b328b62 — lzf: update pointer p after realloc

## The defect

`ff_lzf_uncompress` keeps a write cursor `p` into the output buffer and grows that buffer with `av_reallocp` when more room is needed. `realloc` may move the block, freeing the old one, and the cursor was not rebased — so the `bytestream2_get_buffer(gb, p, s)` that follows writes through a pointer into freed storage. Two sites in the same function.

## Upstream defect

- **Fix:** `43de8b328b62`, *"lzf: update pointer p after realloc"*, `libavcodec/lzf.c`.
- **CVE:** none assigned.
- **Live at our n9.0.1 pin: NO.** The pin rebases at both sites. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
                ret = av_reallocp(buf, *size);
                if (ret < 0)
                    return ret;
            }

            bytestream2_get_buffer(gb, p, s);
```

## The fix

```c
                ret = av_reallocp(buf, *size);
                if (ret < 0)
                    return ret;
                p = *buf + len;
            }
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows it. The object is a plain allocation because upstream's
is — `av_malloc` is malloc plus alignment, with no pool in the path.

**Reduced:** no bytestream and no LZF data. The buffer is grown by a real realloc that is forced to move, and the copy is reduced to a single write through the cursor at the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing because the reduction frees
and re-allocates immediately. Nor upstream reachability of the specific sequence chosen here.
