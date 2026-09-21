# `90bb3a5c9e` — an epan-scope name freed individually while the registry still holds it

A use-after-free in Wireshark's XML dissector, reproduced against Wireshark's
own unmodified `wmem` block allocators — and the corpus's recorded
non-detection.

## The defect

`register_dtd` names the protocol it registers after the DTD's root element,
an epan-scope string; `proto_register_protocol` stores that pointer as the
protocol's and the header field's name. A performance change turned the local
`root_name` from an owned copy into an alias of that string while leaving the
`wmem_free(wmem_epan_scope(), root_name)` at the end of the function in place.
The chunk goes to the epan scope's recycler. Every later packet that adds the
protocol item runs `g_strdup(hfinfo->name)` over it.

## Upstream defect

Upstream fix `90bb3a5c9e`, "xml: Fix memory usage", first tag v4.6.0.
Reported as #20664 by the ASan fuzz job on master.

- **CVE:** `NO VERIFIED CVE`.
- **Live in our pin:** no. At the 4.6.8 pin the `wmem_free` line is gone from
  `packet-xml.c`. The pre-fix shape is quoted below from the fix's parent.
- **First shipped:** never in a release. The introducing change `395011db11`
  (in the report's own list of commits from the previous 48 hours) and the
  fix are both first tagged v4.6.0.

## The vulnerable code, quoted from the fix's parent

`git show 90bb3a5c9e^:epan/dissectors/packet-xml.c`:

```c
1453        /* we will use the first element found as root in case no other one was given. */
1454        if (root_name == NULL)
1455            root_name = nl->name;
...
1499    if ( dtd_data->proto_root ) {
1500        root_name = dtd_data->proto_root;
...
1600        if (dtd_data->description) {
1601            full_name = dtd_data->description;
1602        } else {
1603            full_name = root_name;
...
1616        root_element->hf_tag = proto_register_protocol(full_name, short_name, short_name);
...
1630    destroy_dtd_data(dtd_data);
1631    wmem_free(wmem_epan_scope(), root_name);
```

`:709`, and `ftype-protocol.c:54`, the read on every packet:

```c
709     pi = proto_tree_add_item(current_frame->tree, ns->hf_tag, tok->tvb, tok->offset, tok->len, ENC_UTF_8|ENC_NA);
...
54  	fv->value.protocol.proto_string = g_strdup(name);
```

The reported trace describes the address, not the object: its "previously
allocated" stack is a packet-pool `add_layer` object that had recycled the
freed chunk, and its "freed by" stack is that object's reset. The root name's
own free is the line above, taken from the fix.

## The fix

```diff
     wmem_map_foreach(elements, free_elements, NULL);
+    free_elements(NULL, root_element, NULL);
     destroy_dtd_data(dtd_data);
-    wmem_free(wmem_epan_scope(), root_name);
```

## What is real here, and what is reduced

**Real:** the allocator. The free lands in the production `block` allocator's
recycler, and the case asserts that the next request of the same size receives
the same chunk — that is what makes the registry's name alias new data.

**Reduced:** the consumer. A 16-byte name from the epan scope, its pointer in
a static standing for the registry, the free, one allocation that takes the
chunk back, a packet reset that changes nothing here, and a read of the first
byte where `g_strdup` begins.

## What the run establishes, and what it does not

**Both arms are expected, and checked, to complete.** Sublet lends whole
regions; a chunk inside a live 8 MiB block has no epoch of its own, and an
individual `wmem_free` ends none. This row records that boundary with a
reported defect rather than only with a fixture. It says nothing about
mechanisms that act per chunk, which is exactly the comparison the row exists
to make possible.

## Not yet done

- No `before.c`, so no host `native-detect` arm.
