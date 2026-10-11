# `6eab9f83ab` — a packet-scope string handed to col_set_str and printed after the scope ended

A use-after-free in Wireshark's SICK CoLA dissector, reproduced against
Wireshark's own unmodified `wmem` block allocators.

## The defect

`dissect_sick_cola_b_pdu` reads the command name with
`tvb_get_string_enc(wmem_packet_scope(), ...)` and passes it to `col_set_str`,
whose contract is to keep the pointer and copy nothing ("Usually used to set
const strings!"). The scope is torn down at the end of dissection;
`print_columns` then runs `strlen` over `COL_INFO` for the same packet.

## Upstream defect

Upstream fix `6eab9f83ab`, 2025-06-23, "SICK CoLA: uses packet_info pool
memory instead of packet_scope one", first tag v4.6.0. Reported as #20587 by
the ASan fuzz job on master. A second fix landed against the same capture,
`523e6aa11a` "RPC: Use pinfo->pool instead of wmem_packet_scope" (also
v4.6.0); the reported trace is the CoLA path and shows no RPC frame, so the RPC
change is a co-fixed sibling this case does not model.

- **CVE:** `NO VERIFIED CVE`.
- **Live in our pin:** no. At the 4.6.8 pin `packet-cola.c:2304` allocates from
  `pinfo->pool`. The pre-fix shape is quoted below from the fix's parent.
- **First shipped:** never in a release. The dissector merged on 2025-06-15
  and the fix on 2025-06-23, both first tagged v4.6.0.

## The vulnerable code, quoted from the fix's parent

`git show 6eab9f83ab^:epan/dissectors/packet-cola.c`:

```c
2304	col_set_str(pinfo->cinfo, COL_INFO, tvb_get_string_enc(wmem_packet_scope(), tvb, offset, 4, ENC_ASCII));
```

The reported trace:

```
READ of size 14 ... in strlen
    #1 in print_columns tshark.c:4630
0x... is located 0 bytes after 45-byte region
freed by thread T0 here:
    ... wmem_free_all -> wmem_leave_packet_scope
    #8 in epan_dissect_run_with_taps epan/epan.c:680
previously allocated by thread T0 here:
    ... wmem_strbuf_finalize <- get_ascii_string <- tvb_get_ascii_string <- tvb_get_string_enc
    #10 in dissect_sick_cola_b_pdu epan/dissectors/packet-cola.c:2304
```

## The fix

```diff
-	col_set_str(pinfo->cinfo, COL_INFO, tvb_get_string_enc(wmem_packet_scope(), tvb, offset, 4, ENC_ASCII));
+	col_set_str(pinfo->cinfo, COL_INFO, tvb_get_string_enc(pinfo->pool, tvb, offset, 4, ENC_ASCII));
```

Every `wmem_packet_scope()` in the dissector becomes `pinfo->pool`.

## What is real here, and what is reduced

**Real:** the allocator; the seam's packet pool stands in for the separate
`wmem_packet_scope()` pool of that release.

**Reduced:** the consumer. A 45-byte string from the packet pool, its pointer
parked where `COL_INFO` keeps it, the reset, then the read where `strlen`
begins. Nothing is allocated in between, as in the report.

## What the run establishes, and what it does not

The `sublet` arm must fault at the labelled read; the `spatial` arm completes
with the old byte. The four-byte `tvb` read and the 45-byte region were not
reconciled from the trace; the site, pool and fix are certain.

## Not yet done

- No `before.c`, so no host `native-detect` arm.
