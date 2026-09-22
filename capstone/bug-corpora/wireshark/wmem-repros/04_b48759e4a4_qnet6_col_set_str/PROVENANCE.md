# `b48759e4a4` — col_add_str turned into col_set_str on a heap string

A use-after-free in Wireshark's qnet6 dissector, reproduced against
Wireshark's own unmodified `wmem` block allocators.

## The defect

A mass conversion "For literal strings, prefer col_set_str() to
col_add_fstr()" changed this `COL_INFO` store from `col_add_str`, which copies,
to `col_set_str`, which keeps the pointer — but the argument is `val_to_str`,
whose default string is not a literal: it is formatted into
`wmem_packet_scope()`. The scope is torn down at the end of dissection and
`print_columns` runs `strlen` over the column.

## Upstream defect

Upstream fix `b48759e4a4`, 2024-07-26, "qnet6: Do not use col_set_str on the
result of val_to_str", first tag v4.3.1. Its message: "val_to_str (non const)
adds a string in packet scope." Reported as #19960 by the ASan fuzz job the
day after the conversion.

- **CVE:** `NO VERIFIED CVE`.
- **Live in our pin:** no. At the 4.6.8 pin `packet-qnet6.c:4068` uses
  `col_add_str`. The pre-fix shape is quoted below from the fix's parent.
- **First shipped:** never in a release. The introducing conversion
  `9e3e17136c` and the fix are both first tagged v4.3.1; `git blame` at the
  fix's parent attributes the line to the conversion.

## The vulnerable code, quoted from the fix's parent

`git show b48759e4a4^:epan/dissectors/packet-qnet6.c`:

```c
4066  col_set_str(pinfo->cinfo, COL_INFO, val_to_str(qtype, qnet6_type_vals, "Unknown LWL4 Type %u packets"));
```

The reported trace:

```
READ of size 39 ... in strlen
    #1 in print_columns tshark.c:4509
0x... is located 0 bytes after 70-byte region
freed by thread T0 here:
    ... wmem_free_all -> wmem_leave_scope -> wmem_leave_packet_scope
    #8 in epan_dissect_run_with_taps epan/epan.c:665
previously allocated by thread T0 here:
    ... wmem_strdup_printf <- val_to_str
    #8 in dissect_qnet6 epan/dissectors/packet-qnet6.c:4066
```

## The fix

```diff
-  col_set_str(pinfo->cinfo, COL_INFO, val_to_str(qtype, qnet6_type_vals, "Unknown LWL4 Type %u packets"));
+  col_add_str(pinfo->cinfo, COL_INFO, val_to_str(qtype, qnet6_type_vals, "Unknown LWL4 Type %u packets"));
```

The only store-side fix in this corpus: the column copies the string again,
and the allocation's scope is left alone.

## What is real here, and what is reduced

**Real:** the allocator; the seam's packet pool stands in for the separate
`wmem_packet_scope()` pool of that release.

**Reduced:** the consumer. A 70-byte string from the packet pool, its pointer
parked where `COL_INFO` keeps it, the reset, the read where `strlen` begins.

## What the run establishes, and what it does not

The `sublet` arm must fault at the labelled read; the `spatial` arm completes
with the old byte. The 70-byte region against a shorter default string was not
reconciled; the site and fix are unambiguous.

## Not yet done

- No `before.c`, so no host `native-detect` arm.
