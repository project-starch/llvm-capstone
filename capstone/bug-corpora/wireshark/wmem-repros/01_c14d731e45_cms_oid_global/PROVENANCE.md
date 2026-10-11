# `c14d731e45` — a packet-scope OID string kept in a global and read by a later packet

A use-after-free in Wireshark's CMS dissector, reproduced against Wireshark's
own unmodified `wmem` block allocators.

## The defect

`dissect_cms_T_capability` decodes an object identifier into a packet-scope
string and keeps the pointer in the file-scope global `object_identifier_id`.
When a later packet's capability decode raises a non-fatal exception before it
re-sets the global, `dissect_cms_T_parameters` calls
`call_ber_oid_callback` with the stale pointer; `find_string_dtbl_entry` does a
`g_strdup`, and the `strlen` inside it walks storage the next dissection has
already reoccupied.

## Upstream defect

Upstream fix `c14d731e45`, 2022-01-07, "CMS: get rid of globals", on master;
cherry-picked as `b0f679bb4e` (release-3.6, v3.6.2) and `9369af77d3`
(release-3.4, v3.4.12). The commit message states the shape:

> Get rid of the global content_tvb and object_identifier_id in
> the CMS dissector, and put them in a packet scoped proto data
> struct, so that when there's a non fatal exception retrieving
> the OID we don't use the global value from a previous packet
> (or worse, file), since what the content_tvb and object_identifier_id
> pointed to were both packet scoped that could lead to memory
> access violations.

Reports closed by it: #17800, #17809, #17835 and, through the stable
cherry-picks, #17935. The traces of #17809, #17835 and #17935 are identical
down to the frame; #17800 is the `content_tvb` half, not modelled here. This
case reproduces #17835.

- **CVE:** `NO VERIFIED CVE`.
- **Live in our pin:** no. At the 4.6.8 pin the globals are gone and
  `cms_get_private_data` (`packet-cms.c:365`) allocates the holder from
  `pinfo->pool`. The pre-fix shape is quoted below from the fix's parent.
- **First shipped:** the global predates the generated file; the declaration
  line was last re-touched by an earlier partial mitigation in v3.1.1.

## The vulnerable code, quoted from the fix's parent

`git show c14d731e45^:epan/dissectors/packet-cms.c`:

```c
314  static const char *object_identifier_id = NULL;
315  static tvbuff_t *content_tvb = NULL;
...
1718 dissect_cms_T_capability(...) {
1722     offset = dissect_ber_object_identifier_str(implicit_tag, actx, tree, tvb, offset, hf_cms_attrType, &object_identifier_id);
...
1739 dissect_cms_T_parameters(...) {
1742   offset=call_ber_oid_callback(object_identifier_id, tvb, offset, actx->pinfo, tree, NULL);
```

The reported trace (#17835):

```
READ of size 28 ... in strlen
    #1 in g_strdup
    #2 in find_string_dtbl_entry epan/packet.c:1533
    #5 in call_ber_oid_callback epan/dissectors/packet-ber.c:1071
    #6 in dissect_cms_T_parameters asn1/cms/cms.cnf:220
0x... is located 0 bytes to the right of 59-byte region
freed by thread T0 here:
    ... wmem_free_all -> wmem_leave_scope -> wmem_leave_packet_scope
    #8 in epan_dissect_run epan/epan.c:617
previously allocated by thread T0 here:
    ... wmem_strbuf_finalize <- rel_oid_subid2string <- oid_encoded2string
    #11 in dissect_cms_T_capability asn1/cms/cms.cnf:210
```

## The fix

```diff
-static const char *object_identifier_id = NULL;
-static tvbuff_t *content_tvb = NULL;
+struct cms_private_data {
+  const char *object_identifier_id;
+  tvbuff_t *content_tvb;
+};
...
+  struct cms_private_data *cms_data = cms_get_private_data(actx->pinfo);
+  cms_data->object_identifier_id = NULL;
...
-  offset=call_ber_oid_callback(object_identifier_id, ...)
+  offset=call_ber_oid_callback(cms_data->object_identifier_id, ...)
```

The holder becomes per-packet proto data in `pinfo->pool`, cleared before each
retrieval, so no packet can read another packet's pointer.

## What is real here, and what is reduced

**Real:** the allocator, as in every case of this corpus. The reset that ends
the OID string is the production `block_fast` `free_all`, and the next
packet's allocation landing on the same address is asserted before the marker.

**Reduced:** the consumer. The BER decode, the exception that skips the
re-set, and the dissector table lookup are replaced by the allocator calls
they make: a 59-byte object from the packet pool, its pointer in a static, the
reset, one allocation by the next packet, then the read through the static.

## What the run establishes, and what it does not

The `sublet` arm must fault at the labelled read, the first byte of the
`strlen`. The `spatial` arm completes, reading the next packet's bytes.

Three separate fuzz captures reached this one line over two months, on three
branches; that recurrence is why the survey ranked it among the most
defensible rows. The run does not establish anything about #17800, whose
object is a `tvbuff`.

## Not yet done

- No `before.c`, so no host `native-detect` arm.
