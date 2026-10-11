# `693dc40936` — a pinfo->pool buffer kept by proto data added with the wrong scope

A use-after-free in Wireshark's GeoNetworking dissector, reproduced against
Wireshark's own unmodified `wmem` block allocators.

## The defect

PPP's `remove_escape_chars` unescapes the frame into a `pinfo->pool` buffer
and wraps it in a child tvb. GeoNW, when it meets a secured packet, stashes
that tvb in proto data added with `wmem_file_scope()`, so the entry outlives
the packet. The pool is reset between packets; a later frame retrieves the
entry and reads the header type through the stale tvb. The pre-fix code even
tried to clear the entry by adding a `NULL` one — but `p_add_proto_data`
appends and `p_get_proto_data` returns the first match, so the stale entry was
never hidden.

## Upstream defect

Upstream fix `693dc40936`, "GNW: Ensure that tvbuff proto data has the proper
scope.", on master, first tag v4.1.0; cherry-picked as `086700bcc0`
(release-3.6, v3.6.11, the report's branch) and `e3cfec7313` (release-4.0,
v4.0.3). Reported as #18779 by the ASan fuzz job on release-3.6.

- **CVE:** `NO VERIFIED CVE`.
- **Live in our pin:** no. At the 4.6.8 pin `packet-geonw.c:1924` adds the
  entry with `p_add_proto_data(pinfo->pool, ..., SEC_TVB_KEY, ...)`. The
  pre-fix shape is quoted below from the master fix's parent.
- **First shipped:** not established; the file-scope store predates the
  report.

## The vulnerable code, quoted from the fix's parent

`git show 693dc40936^:epan/dissectors/packet-ppp.c`, the buffer:

```c
5914    buff = (guint8 *)wmem_alloc(pinfo->pool, length);
```

`git show 693dc40936^:epan/dissectors/packet-geonw.c`, the stores, the
retrieval and the read:

```c
1913                    p_add_proto_data(wmem_file_scope(), pinfo, proto_geonw, 0, next_tvb);
...
2116    p_add_proto_data(wmem_file_scope(), pinfo, proto_geonw, 0, tvb);
...
2229        tvbuff_t *next_tvb = (tvbuff_t*)p_get_proto_data(wmem_file_scope(), pinfo, proto_geonw, 0);
2230        if (next_tvb) {
2231            tvb = next_tvb;
...
2234            header_type = tvb_get_guint8(tvb, 1);
...
2262            p_add_proto_data(wmem_file_scope(), pinfo, proto_geonw, 0, NULL);
```

The reported trace (release-3.6 line numbers):

```
READ of size 1 ... in tvb_get_guint8 epan/tvbuff.c:1027
    #1 in dissect_geonw epan/dissectors/packet-geonw.c:2235
0x... is located 35 bytes inside of 55-byte region
freed by thread T0 here:
    ... wmem_free_all
    #6 in epan_dissect_reset epan/epan.c:581
previously allocated by thread T0 here:
    #4 in remove_escape_chars epan/dissectors/packet-ppp.c:5859
```

## The fix

```diff
-                    p_add_proto_data(wmem_file_scope(), pinfo, proto_geonw, 0, next_tvb);
+                    p_add_proto_data(pinfo->pool, pinfo, proto_geonw, SEC_TVB_KEY, next_tvb);
-    p_add_proto_data(wmem_file_scope(), pinfo, proto_geonw, 0, tvb);
+    p_add_proto_data(pinfo->pool, pinfo, proto_geonw, SEC_TVB_KEY, tvb);
-        tvbuff_t *next_tvb = (tvbuff_t*)p_get_proto_data(wmem_file_scope(), pinfo, proto_geonw, 0);
+        tvbuff_t *next_tvb = (tvbuff_t*)p_get_proto_data(pinfo->pool, pinfo, proto_geonw, SEC_TVB_KEY);
-            p_add_proto_data(wmem_file_scope(), pinfo, proto_geonw, 0, NULL);
```

The entry moves to `pinfo->pool`, the same lifetime as the tvb it holds, and
the futile `NULL` entry is dropped.

## What is real here, and what is reduced

**Real:** the allocator, and the reset that ends the buffer.

**Reduced:** the consumer. The tvb wrapper is elided — the read goes through
it to the backing bytes — and the proto-data list is reduced to one file-scope
entry holding the buffer. The case allocates the 55-byte buffer from the packet
pool, stores it, resets the pool, and reads through the retrieved pointer where
`tvb_get_guint8(tvb, 1)` does, at byte 0 rather than byte 1. No reoccupation
is asserted.

On Capstone, deriving an interior pointer from revoked authority faults at the
arithmetic, before any load. The case therefore reads through the pointer the
holder returns, at its first byte, rather than computing the report's field
offset after the reset: both are loads through the same revoked authority, and
the oracle names the load. The port's own fixtures document the same rule.


## What the run establishes, and what it does not

The `sublet` arm must fault at the labelled read; the `spatial` arm completes.
The trace's own read is 35 bytes in, a later access on the same stale tvb; the
case reads where the retrieval first touches it.

## Not yet done

- No `before.c`, so no host `native-detect` arm.
