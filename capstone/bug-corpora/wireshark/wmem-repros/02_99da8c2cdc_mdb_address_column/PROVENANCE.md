# `99da8c2cdc` — a packet-scope default string kept by an address and printed after the scope ended

A use-after-free in Wireshark's MDB dissector, reproduced against Wireshark's
own unmodified `wmem` block allocators.

## The defect

`mdb_set_addrs` names an unknown peripheral with `val_to_str`, whose default
string comes from `wmem_packet_scope()`, and hands it to `set_address` as
`AT_STRINGZ`, which keeps the pointer. On release-4.4 that scope is torn down at
the end of `epan_dissect_run_with_taps`, before tshark prints the packet;
`print_columns` then runs `strlen` over the address column. No allocation
intervenes between the reset and the read, so the unprotected read returns
the old bytes and the output looks right.

## Upstream defect

Upstream fix `99da8c2cdc`, 2026-05-18, "MDB: Allocate a column string with
pinfo->pool", on release-4.4, first tag v4.4.16. The commit message states the
mechanism and, in passing, why the production allocator never showed it:

> the old val_to_str on 4.4 uses wmem_packet_scope, which has slightly too
> short a lifespan to be used in a column ... It probably has no effect on the
> normal block [fast] allocator.

Reported as #21261 by the ASan fuzz job on release-4.4.

- **CVE:** `NO VERIFIED CVE`.
- **Live in our pin:** no. `wmem_packet_scope()` does not exist at the 4.6.8
  pin (`epan/wmem_scopes.c` holds only the file and epan scopes), and
  `packet-mdb.c:259` allocates the string from `pinfo->pool`. The pre-fix shape
  is quoted below from the fix's parent.
- **First shipped:** the dissector, with this store, arrived in v4.1.0.

## The vulnerable code, quoted from the fix's parent

`git show 99da8c2cdc^:epan/dissectors/packet-mdb.c`:

```c
257      const char *periph = val_to_str(addr, mdb_addr, "Unknown (0x%02x)");
...
263          set_address(&pinfo->dst, AT_STRINGZ, (int)strlen(periph)+1, periph);
...
267          set_address(&pinfo->src, AT_STRINGZ, (int)strlen(periph)+1, periph);
```

`git show 99da8c2cdc^:epan/epan.c`, the scope that ends before printing:

```c
658  epan_dissect_run_with_taps(...)
662      wmem_enter_packet_scope();
665      dissect_record(...);
668      wmem_leave_packet_scope();
```

The reported trace:

```
READ of size 47 ... in strlen
    #1 in print_columns tshark.c:4597
    #3 in process_packet_second_pass tshark.c:3709
0x... is located 0 bytes after 55-byte region
freed by thread T0 here:
    ... wmem_free_all -> wmem_leave_packet_scope
    #8 in epan_dissect_run_with_taps epan/epan.c:668
previously allocated by thread T0 here:
    ... wmem_strdup_printf <- val_to_str
    #8 in mdb_set_addrs epan/dissectors/packet-mdb.c:257
```

## The fix

```diff
-    const char *periph = val_to_str(addr, mdb_addr, "Unknown (0x%02x)");
+    const char *periph = val_to_str_wmem(pinfo->pool, addr, mdb_addr, "Unknown (0x%02x)");
```

The string moves to `pinfo->pool`, which survives until the next
`epan_dissect_reset`, after the packet has been printed.

## What is real here, and what is reduced

**Real:** the allocator. The reported pool, `wmem_packet_scope()`, was a
separate `block_fast` pool in release-4.4 with the same `free_all` as
`pinfo->pool`; the seam's packet pool stands in for it.

**Reduced:** the consumer. The case allocates a 55-byte string from the packet
pool, parks its pointer where the address keeps it, resets the pool, and reads
the first byte where `strlen` would. Deliberately, nothing is allocated between
the reset and the read: that is the reported sequence, and it is why the
unprotected arm returns the old byte.

## What the run establishes, and what it does not

The `sublet` arm must fault at the labelled read. The `spatial` arm completes
with the old byte — which is what production does, and why this defect is
found only under the strict allocator.

Reconciling the trace's 55-byte region and 32-byte read offset with the short
default string was not possible from the generic `print_columns` frame; the
allocation site, pool and fix are unambiguous and the reduction rests on them.

## Not yet done

- No `before.c`, so no host `native-detect` arm.
