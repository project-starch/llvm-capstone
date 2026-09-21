# `3c8be14c82` — a packet-scope array kept in a global and read by a later packet

A use-after-free in Wireshark's RPC-over-RDMA dissector, reproduced against
Wireshark's own unmodified `wmem` block allocators.

## The defect

`dissect_rpcrdma` builds the list of write offsets as a `wmem_array` in packet
scope and keeps the pointer in the file-scope global `gp_rdma_write_offsets`.
Nothing clears the global when the packet ends. The packet pool is reset at the
end of dissection, the next packet's dissection reoccupies the same storage,
and a later `process_rdma_list` reads the array's element count through the
global — four bytes at offset 56 of storage that now belongs to something else.

## Upstream defect

Upstream fix `3c8be14c82`, 2023-03-19, "RPCoRDMA: Frame end cleanup for global
write offsets", on master; cherry-picked as `c224405c31` (release-3.6, the
report's branch) and `04e506a337` (release-4.0). The commit message states the
shape in its own words:

> Add a frame end routine for a global which is assigned to packet
> scoped memory. It really should be made proto data, but is used
> in a function in the header (that doesn't take the packet info
> struct as an argument) and this fix needs to be made in stable
> branches.

The fix names #18852. The report reproduced here is #18910 (2023-03-16, fuzz
capture on release-3.6), whose trace hits exactly this global and which was
closed the following day; the issue record carries no explicit duplicate link,
so the identification rests on the identical mechanism and the timeline.

- **CVE:** `NO VERIFIED CVE`.
- **Live in our pin:** no. The fix is in v3.6.13, v4.0.5 and v4.1.0; at the
  4.6.8 pin `reset_write_offsets` is present at `packet-rpcrdma.c:298` and
  registered at `:1618`. The pre-fix shape is quoted below from the fix's
  parent, by line.
- **First shipped:** the global-from-packet-scope store dates from `8f0f691312`
  (2017), first tag v2.6.0.

## The vulnerable code, quoted from the fix's parent

`git show 3c8be14c82^:epan/dissectors/packet-rpcrdma.c`:

```c
253  static wmem_array_t *gp_rdma_write_offsets = NULL;
...
1602                 gp_rdma_write_offsets = wmem_array_new(wmem_packet_scope(), sizeof(gint));
...
1159                if (gp_rdma_write_offsets && wmem_array_get_count(gp_rdma_write_offsets) == wmem_array_get_count(p_list)) {
1160                    p_offset = (guint *)wmem_array_index(gp_rdma_write_offsets, i);
```

The reported trace (release-3.6 line numbers are lower; the code is the same):

```
READ of size 4 ... in wmem_array_get_count wsutil/wmem/wmem_array.c:158
    #1 in process_rdma_list epan/dissectors/packet-rpcrdma.c:992
0x... is located 56 bytes inside of 80-byte region
freed by thread T0 here:
    ... wmem_free_all -> wmem_leave_scope -> wmem_leave_packet_scope
    #8 in epan_dissect_run epan/epan.c:617
previously allocated by thread T0 here:
    ... wmem_array_sized_new -> wmem_array_new
    #6 in dissect_rpcrdma epan/dissectors/packet-rpcrdma.c:1411
```

## The fix

```diff
+static void
+reset_write_offsets(void)
+{
+    gp_rdma_write_offsets = NULL;
+}
...
             gp_rdma_write_offsets = wmem_array_new(wmem_packet_scope(), sizeof(gint));
+            register_frame_end_routine(pinfo, reset_write_offsets);
```

## What is real here, and what is reduced

**Real:** the allocator. `wmem_core.c` and the four allocators from the pinned
4.6.8 release, unmodified but for the guarded authority hooks the port applies.
The packet pool is the production `block_fast` allocator; its reset retains the
first block and reissues its storage, which is why the next packet's first
allocation lands on the same address — the case asserts that before the marker.

**Reduced:** the consumer. Reaching `process_rdma_list` needs an RDMA capture
with a write chunk list across two frames. The case performs the allocator
calls that sequence makes, in the same order: an 80-byte object from the packet
pool, its pointer parked in a static, the pool reset, the next packet's
allocation, then a read of the count field through the static. The reported
trace's `wmem_packet_scope()` was a separate `block_fast` pool in that release;
the seam's packet pool stands in for it with the same reset.

## What the run establishes, and what it does not

On Capstone, deriving an interior pointer from revoked authority faults at the
arithmetic, before any load. The case therefore reads through the pointer the
holder returns, at its first byte, rather than computing the report's field
offset after the reset: both are loads through the same revoked authority, and
the oracle names the load. The port's own fixtures document the same rule.

The `sublet` arm must fault at the labelled read, standing for the trace's
`wmem_array_get_count`. The `spatial` arm completes, reading
the next packet's bytes — the silent corruption production sees.

It does not establish that Wireshark as a whole is protected, or that the
frame-end routine upstream chose is the right fix. It establishes that this
one defect, on this allocator, at this pin, is a revoked read.

## Not yet done

- No `before.c`, so no host `native-detect` arm; ASan sees this only under the
  strict allocator override, which is how it was reported.
