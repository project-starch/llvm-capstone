# `31ab1a0a17` — a CSeq method in pinfo->pool stored into the resend record by a truncation fix

A use-after-free in Wireshark's SIP dissector, reproduced against Wireshark's
own unmodified `wmem` block allocators.

## The defect

A fix for fixed-buffer truncation changed the resend record's `method` from
an inline array, filled by `g_strlcpy`, to a `const char *`, and stored the
`pinfo->pool` string that `proto_tree_add_item_ret_string` returned. The
record lives in file scope, reachable from a static hash. The pool is reset
when the packet ends; the next SIP packet's `sip_is_packet_resend` compares
its own method against the stored pointer with `strcmp`.

## Upstream defect

Upstream fix `31ab1a0a17`, "SIP: Fix heap-use-after-free crash with ASAN",
first tag v4.1.0. Reported as #18735 by the ASan fuzz job on master.

- **CVE:** `NO VERIFIED CVE`.
- **Live in our pin:** no. At the 4.6.8 pin `packet-sip.c:5302` stores
  `wmem_strdup(wmem_file_scope(), cseq_method)`. The pre-fix shape is quoted
  below from the fix's parent.
- **First shipped:** never in a release. The introducing change `358641a5`
  "SIP: Fix fixed buffer UTF-8 string truncation" and the fix are one day
  apart and both first tagged v4.1.0.

## The vulnerable code, quoted from the fix's parent

`git show 31ab1a0a17^:epan/dissectors/packet-sip.c`:

```c
1300 static GHashTable *sip_hash = NULL;           /* Hash table */
1331     const char         *method;
1335 } sip_hash_value;
...
4096                            proto_tree_add_item_ret_string(cseq_tree, hf_sip_cseq_method, tvb,
4097                                                    value_offset + sub_value_offset, strlen_to_copy, ENC_UTF_8,
4098                                                    pinfo->pool, (const guint8 **)&cseq_method);
...
5338         p_val = wmem_new0(wmem_file_scope(), sip_hash_value);
...
5352         p_val->method = cseq_method;
...
5375         (strcmp(cseq_method, p_val->method) == 0) &&
```

The reported trace:

```
READ of size 1 ... in strcmp
    #1 in sip_is_packet_resend epan/dissectors/packet-sip.c:5375
0x... is located 32 bytes inside of 48-byte region
freed by thread T0 here:
    ... wmem_free_all
    #6 in epan_dissect_reset epan/epan.c:591
previously allocated by thread T0 here:
    ... tvb_get_string_enc
```

## The fix

```diff
-            p_val->method = cseq_method;
+            p_val->method = wmem_strdup(wmem_file_scope(), cseq_method);
-        p_val->method = cseq_method;
+        p_val->method = wmem_strdup(wmem_file_scope(), cseq_method);
```

Both stores copy into file scope, restoring the ownership the inline array
had provided.

## What is real here, and what is reduced

**Real:** the allocator, and the reset that ends the string.

**Reduced:** the consumer. The hash and its lookup are reduced to the record;
the case allocates the 48-byte method from the packet pool, stores it in a
file-scope record, resets the pool, and reads the first byte where `strcmp`
begins. No reoccupation is asserted.

## What the run establishes, and what it does not

The `sublet` arm must fault at the labelled read; the `spatial` arm completes.

## Not yet done

- No `before.c`, so no host `native-detect` arm.
