# pg_basebackup forwards a tar member trailer from an advanced pointer

Upstream commit `f1298a4c20`, "Fix multiple bugs in astreamer pipeline code",
backpatched through 15. The commit fixes five separate things; only the first
is a memory-safety defect and only that one is reproduced here.

Upstream's own words:

> `astreamer_tar_parser_content()` sent the wrong data pointer when forwarding
> MEMBER_TRAILER padding to the next streamer. After
> `astreamer_buffer_until()` buffers the padding bytes, the 'data' pointer has
> been advanced past them, but the code passed 'data' instead of
> `bbs_buffer.data`. This caused the downstream consumer to receive bytes from
> after the padding rather than the padding itself, and could read past the end
> of the input buffer.

The pipeline is named `astreamer` on master and `bbstreamer` on `REL_17_STABLE`;
the pin has the `bbstreamer` spelling and the same defect.

## Why it is in this corpus

`pg_basebackup` is a client program. Its `palloc` is
`src/common/fe_memutils.c`'s, which is `pg_malloc` and therefore `malloc`; the
streamer buffers and the input chunks are libc allocations. The defect is
non-nested.

## Established on the pin by inspection

`src/bin/pg_basebackup/bbstreamer_tar.c:220-228`:

```c
if (!bbstreamer_buffer_until(streamer, &data, &len,
                             mystreamer->pad_bytes_expected))
    return;

/* OK, now we can send it. */
bbstreamer_content(mystreamer->base.bbs_next,
                   &mystreamer->member,
                   data, mystreamer->pad_bytes_expected,
                   BBSTREAMER_MEMBER_TRAILER);
```

`bbstreamer_buffer_until` takes `data` and `len` **by address**.
`src/bin/pg_basebackup/bbstreamer.h:174-196` shows why that matters: at `:194`
it calls `bbstreamer_buffer_bytes(streamer, data, len, target_bytes - buflen)`,
which copies those bytes into the streamer's own buffer and advances `*data`
past them while subtracting the same count from `*len`.

So by the time line 225 runs, `data` no longer points at the padding -- the
padding is in `bbs_buffer`. If the padding ran to the end of the chunk, `len`
is zero and `data` is one past the end of the input. The call then asks the
next streamer to read `pad_bytes_expected` bytes starting there.

Upstream's fix passes `mystreamer->base.bbs_buffer.data` instead.

## Trigger

A tar stream from the server in which a member's padding ends exactly on a
chunk boundary. `pg_basebackup -Ft` with a backup whose file sizes are not a
multiple of 512 produces padding on every member; the overread occurs on the
members whose padding completes a chunk.
