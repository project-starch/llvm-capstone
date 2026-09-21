# `5a109265a6` — a three-byte address in packet scope kept by pinfo and read at column fill

A use-after-free in Wireshark's USB link-layer dissector, reproduced against
Wireshark's own unmodified `wmem` block allocators.

## The defect

`usbll_set_address` allocates the source and destination address structures
from `wmem_packet_scope()` and hands them to `set_address`, which keeps the
pointer. That scope is torn down at the end of `epan_dissect_run_with_taps`;
filling the columns for the same packet then calls `usbll_addr_to_str`, which
reads the address's flags byte.

## Upstream defect

Upstream fix `5a109265a6`, "USBLL: allocate address in pinfo pool", first tag
v3.5.0. Its message: "The packet pool lifetime is too short for data added to
the pinfo structure". Reported as #17367 by the fuzz job on master; #17368,
filed the same day from another capture with the same trace and the same
fault address, was closed as its duplicate.

- **CVE:** `NO VERIFIED CVE`.
- **Live in our pin:** no. At the 4.6.8 pin `packet-usbll.c:1135-1136`
  allocate from `pinfo->pool`. The pre-fix shape is quoted below from the
  fix's parent.
- **First shipped:** not established; the packet-scope allocation predates
  the report.

## The vulnerable code, quoted from the fix's parent

`git show 5a109265a6^:epan/dissectors/packet-usbll.c`:

```c
288 typedef struct {
289     guint8 flags;       /* flags    - Contains information if address is
                             *            Host, Hub, Device or Broadcast. */
293     guint8 device;
294     guint8 endpoint;
296 } usbll_address_t;
...
627     const usbll_address_t *addrp = (const usbll_address_t *)addr->data;
629     if (addrp->flags & USBLL_ADDRESS_HOST) {
...
662     src_addr = wmem_new0(wmem_packet_scope(), usbll_address_t);
663     dst_addr = wmem_new0(wmem_packet_scope(), usbll_address_t);
...
698     set_address(&pinfo->net_src, usbll_address_type, sizeof(usbll_address_t), (char *)src_addr);
699     copy_address_shallow(&pinfo->src, &pinfo->net_src);
```

The reported trace:

```
READ of size 1 ... in usbll_addr_to_str epan/dissectors/packet-usbll.c:629
    #1 in address_to_str_buf epan/address_types.c:746
    #2 in col_set_addr epan/column-utils.c:1946
    #3 in col_fill_in epan/column-utils.c:2168
    #5 in print_packet tshark.c:4232
0x... is located 32 bytes inside of 43-byte region
freed by thread T0 here:
    ... wmem_free_all -> wmem_leave_packet_scope
    #7 in epan_dissect_run_with_taps epan/epan.c:611
```

## The fix

```diff
-    src_addr = wmem_new0(wmem_packet_scope(), usbll_address_t);
-    dst_addr = wmem_new0(wmem_packet_scope(), usbll_address_t);
+    src_addr = wmem_new0(pinfo->pool, usbll_address_t);
+    dst_addr = wmem_new0(pinfo->pool, usbll_address_t);
```

## What is real here, and what is reduced

**Real:** the allocator; the seam's packet pool stands in for the separate
`wmem_packet_scope()` pool of that release.

**Reduced:** the consumer. A three-byte object from the packet pool, its
pointer parked where the address keeps it, the reset, and a read of byte 0,
the flags. Nothing is allocated in between, as in the report.

## What the run establishes, and what it does not

The `sublet` arm must fault at the labelled read; the `spatial` arm completes
with the old byte. The reported region is 43 bytes because the strict
allocator wraps the object in its header and canaries; the object itself is
three bytes, and the case allocates three.

## Not yet done

- No `before.c`, so no host `native-detect` arm.
