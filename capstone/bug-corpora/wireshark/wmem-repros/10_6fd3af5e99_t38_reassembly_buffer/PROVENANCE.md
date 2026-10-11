# `6fd3af5e99` — a forced-reassembly buffer in pinfo->pool kept by the reassembly table

A use-after-free in Wireshark's T.38 dissector, reproduced against
Wireshark's own unmodified `wmem` block allocators.

## The defect

`force_reassemble_seq` gathers the fragments received so far into a buffer
allocated from `pinfo->pool` and wraps it in a tvb that it stores as the
fragment head's data in the persistent reassembly table. The pool is reset
between packets; the table keeps the tvb. A later frame's `fragment_add_seq`
compares its fragment against the stored data with `tvb_memeql`, a `memcmp`
over freed storage.

## Upstream defect

Upstream fix `6fd3af5e99`, "t38: Allocate forced defragmented memory in
correct scope", on master, first tag v4.3.0; cherry-picked as `c04f268605`
(release-4.0, v4.0.14) and `7be4bbb413` (release-4.2, v4.2.4). Its message:
"Fragment data can't be allocated in pinfo->pool scope, as it outlives the
frame". Reported as #19695 by the ASan fuzz job on master.

- **CVE:** `NO VERIFIED CVE`.
- **Live in our pin:** no. At the 4.6.8 pin `packet-t38.c:362` ties the
  buffer to its tvb with `tvb_set_free_cb(fd_head->tvb_data, g_free)`. The
  pre-fix shape is quoted below from the fix's parent.
- **First shipped:** not established. `git log -S` on the allocation finds
  only the fix commits; the `pinfo->pool` spelling replaced an older scope
  name, and the defect predates the rename.

## The vulnerable code, quoted from the fix's parent

`git show 6fd3af5e99^:epan/dissectors/packet-t38.c`:

```c
358 	data = (guint8 *) wmem_alloc(pinfo->pool, size);
359 	fd_head->tvb_data = tvb_new_real_data(data, size, size);
```

`git show 6fd3af5e99^:epan/reassemble.c`, the read in a later frame:

```c
2064			if (tvb_memeql(fd_head->tvb_data, dfpos,
2065				tvb_get_ptr(tvb,offset,fd->len),fd->len) ){
```

The reported trace:

```
READ of size 1 ... in memcmp
    #2 in tvb_memeql epan/tvbuff.c:2705
    #3 in fragment_add_seq_work epan/reassemble.c:2064
    #6 in dissect_t38_T_field_data epan/dissectors/packet-t38.c:703
0x... is located 32 bytes inside of 41-byte region
freed by thread T0 here:
    ... wmem_free_all
    #6 in epan_dissect_reset epan/epan.c:602
previously allocated by thread T0 here:
    #3 in wmem_alloc wsutil/wmem/wmem_core.c:44
    #4 in force_reassemble_seq epan/dissectors/packet-t38.c:358
```

## The fix

```diff
-	data = (guint8 *) wmem_alloc(pinfo->pool, size);
+	data = (guint8 *) g_malloc(size);
+        tvb_set_free_cb(fd_head->tvb_data, g_free);
```

The buffer leaves the pool and is freed with the tvb that owns it.

## What is real here, and what is reduced

**Real:** the allocator, and the reset that ends the buffer.

**Reduced:** the consumer. The reassembly table is one file-scope head
holding the buffer pointer; the tvb wrapper is elided. The reported instance
had a one-byte buffer, which the case reproduces: one byte from the packet
pool, stored in the head, the reset, then a read of that byte where `memcmp`
begins. No reoccupation is asserted.

## What the run establishes, and what it does not

The `sublet` arm must fault at the labelled read; the `spatial` arm completes.
A one-byte object is also the smallest interval the port's narrowing can
express, so this row doubles as a bounds check on the seam.

## Not yet done

- No `before.c`, so no host `native-detect` arm.
