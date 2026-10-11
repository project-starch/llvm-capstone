# `a8b16d74e1` — a distinguished name kept by a static whose reset was registered on a copy of pinfo

A use-after-free reached through Wireshark's EAP, TLS and X.509 dissectors,
reproduced against Wireshark's own unmodified `wmem` block allocators.

## The defect

`dissect_x509if_RDNSequence` builds the printable distinguished name in a
`pinfo->pool` string buffer and keeps it in the file-level static
`last_dn_buf`, registering `x509if_frame_end` to clear the static when the
packet ends. EAP, however, dissected its EAPOL payload with a **stack copy** of
`pinfo`, so the frame-end routine was registered on the copy's list and never
ran on the real packet. The pool was reset between packets with the static
still pointing into it; a later `dissect_x509af_SubjectName` appended the
buffer to its item text with `"%s"`, reading 2,569 bytes of freed storage.

## Upstream defect

Upstream fix `a8b16d74e1`, "eap: tweak conversation tracking to avoid breaking
pinfo horribly", on master, first tag v4.1.0; cherry-picked as `2fdd1b9c8d`
(release-4.0, v4.0.3). Reported as #18622 by the ASan fuzz job on master.

- **CVE:** `NO VERIFIED CVE`.
- **Live in our pin:** no. At the 4.6.8 pin `packet-eap.c` no longer carries
  `pinfo_eapol`. The pre-fix shape is quoted below from the fix's parent.
- **First shipped:** the copy was introduced by `ce087027ef` "EAP: massage
  pinfo for EAPOL so the TLS decoder does not get lost", first tag v3.7.0.

## The vulnerable code, quoted from the fix's parent

`git show a8b16d74e1^:epan/dissectors/packet-eap.c`, the copy:

```c
1781  packet_info    pinfo_eapol;
1782  packet_info    *pinfo_conv;
...
1818    memcpy(&pinfo_eapol, pinfo, sizeof(packet_info));
1819    pinfo_conv = &pinfo_eapol;
...
2283            call_dissector(tls_handle, next_tvb, pinfo_conv, eap_tree);
```

`git show a8b16d74e1^:epan/dissectors/packet-x509if.c`, the static, the
allocation and the registration that lands on the copy:

```c
279  static wmem_strbuf_t *last_dn_buf = NULL;
...
289  x509if_frame_end(void)
298    last_dn_buf = NULL;
...
911    last_dn_buf = wmem_strbuf_new(actx->pinfo->pool, "");
912    register_frame_end_routine (actx->pinfo, x509if_frame_end);
...
2032 const char * x509if_get_last_dn(void)
2034   return last_dn_buf ? wmem_strbuf_get_str(last_dn_buf) : NULL;
```

`git show a8b16d74e1^:epan/dissectors/packet-x509af.c`, the read:

```c
323    str = x509if_get_last_dn();
324    proto_item_append_text(proto_item_get_parent(tree), " (%s)", str?str:"");
```

The reported trace:

```
READ of size 2569 ... in printf_common
    #1 in wmem_strdup_vprintf wsutil/wmem/wmem_strutl.c:98
    #2 in proto_item_append_text epan/proto.c:7172
    #3 in dissect_x509af_SubjectName asn1/x509af/x509af.cnf:169
0x... is located 0 bytes to the right of 2600-byte region
freed by thread T0 here:
    ... wmem_free_all
    #6 in epan_dissect_reset epan/epan.c:591
previously allocated by thread T0 here:
    #4 in dissect_x509if_RDNSequence asn1/x509if/x509if.cnf:375
```

## The fix

```diff
-  packet_info    pinfo_eapol;
-  packet_info    *pinfo_conv;
...
-    memcpy(&pinfo_eapol, pinfo, sizeof(packet_info));
-    pinfo_conv = &pinfo_eapol;
-      copy_address_shallow(&pinfo_conv->src, &null_address);
```

The copy goes away; sub-dissectors receive the real `pinfo`, and their
frame-end registrations run on it. The x509if static and its routine are
unchanged — they were correct, and attached to the wrong packet.

## What is real here, and what is reduced

**Real:** the allocator, and the reset that ends the buffer.

**Reduced:** the consumer. The case allocates the 2,600-byte buffer from the
packet pool, keeps it in a static that nothing clears — the effect of the
defeated registration — resets the pool, and reads the first byte where the
`%s` begins. No reoccupation is asserted; the report evidences only the read.

## What the run establishes, and what it does not

The `sublet` arm must fault at the labelled read; the `spatial` arm completes.
Why the static was not refreshed in the crashing dissection is data-dependent
and not reconstructed here; the root cause — the routine registered on a
discarded copy — is established from the fix and the quoted lines.

## Not yet done

- No `before.c`, so no host `native-detect` arm.
