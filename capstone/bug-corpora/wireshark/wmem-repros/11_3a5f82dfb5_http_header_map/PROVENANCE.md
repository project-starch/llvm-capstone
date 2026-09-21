# `3a5f82dfb5` — a header map created in pinfo->pool and saved into the conversation's file-scope data

A use-after-free in Wireshark's HTTP dissector, reproduced against
Wireshark's own unmodified `wmem` block allocators.

## The defect

When a header line arrives before any start line, `dissect_http_message`
creates its header map in `pinfo->pool`, since it cannot yet tell which
direction the header belongs to. If the start line follows in the same
message, the code that would create a file-scope map is guarded by
`if (header_value_map == NULL)` — false, because the pool map exists — and the
pool map is stored into the conversation's file-scope private data. The pool
is reset between packets, other dissection reoccupies the storage, and a
later frame retrieves the map and calls `wmem_map_insert` on it, which first
reads `map->table`.

## Upstream defect

Upstream fix `3a5f82dfb5`, "HTTP: Don't save a pinfo->pool scoped map in file
scoped data", first tag v4.7.0. Reported as #20702 by the fuzz job on master,
under the `simple` allocator override, whose Valgrind output shows another
dissector's allocation sitting in the freed block; #20703, filed two days
after the fix from another capture, was closed as its duplicate. This case
reproduces #20702.

- **CVE:** `NO VERIFIED CVE`.
- **Live in our pin:** no. At the 4.6.8 pin `packet-http.c` creates
  `header_value_map` only from `wmem_file_scope()` (`:1598`) before storing it
  at `:1777` and `:1782`; the stray-header path that creates it in
  `pinfo->pool` does not exist there, so the store cannot carry a packet-scope
  map. The pre-fix shape is quoted below from the fix's parent.
- **First shipped:** the file-scope store dates from `f82ed9a537` (v4.6.0);
  the `pinfo->pool` map that makes it unsafe was added on master afterwards
  and fixed in v4.7.0.

## The vulnerable code, quoted from the fix's parent

`git show 3a5f82dfb5^:epan/dissectors/packet-http.c`:

```c
1120	http_req_res_t *req_res = wmem_new0(wmem_file_scope(), http_req_res_t);
1124	req_res->private_data = wmem_new0(wmem_file_scope(), http_req_res_private_data_t);
...
1808			if (header_value_map == NULL) {
1813				header_value_map_allocator = pinfo->pool;
1814				header_value_map = wmem_map_new(header_value_map_allocator, g_str_hash, g_str_equal);
...
1781					prv_data->request_headers = header_value_map;
1786					prv_data->response_headers = header_value_map;
...
1797			if (header_value_map == NULL && conv_data->req_res_tail) {
1802						header_value_map = prv_data->request_headers;
1804						header_value_map = prv_data->response_headers;
...
3611		wmem_map_insert(header_value_map, wmem_strdup(header_value_map_allocator, header_name), value_bytes);
```

`git show 3a5f82dfb5^:wsutil/wmem/wmem_map.c`, the insert's first read:

```c
298     /* Make sure we have a table */
299     if (map->table == NULL) {
```

## The fix

```diff
-				if (header_value_map == NULL) {
+				if (header_value_map_allocator != wmem_file_scope()) {
 					header_value_map_allocator = wmem_file_scope();
 					header_value_map = wmem_map_new(header_value_map_allocator, g_str_hash, g_str_equal);
 				}
```

A pool map is never promoted to file scope; the start line gets a fresh
file-scope map and the stray fields are dropped.

## What is real here, and what is reduced

**Real:** the allocator, the reset that ends the map, and the reoccupation:
the next allocation from the packet pool is asserted to land on the map's
address before the marker, as the report's allocator log shows for a real
capture.

**Reduced:** the consumer. The map is reduced to its 88 bytes of struct
storage, the private data to two pointer fields in a file-scope object, and
the insert to its first read of the struct. Had that read not
faulted, the next step would have called the allocator through the stale
map's own allocator pointer; the case stops at the first touch.

## What the run establishes, and what it does not

On Capstone, deriving an interior pointer from revoked authority faults at the
arithmetic, before any load. The case therefore reads through the pointer the
holder returns, at its first byte, rather than computing the report's field
offset after the reset: both are loads through the same revoked authority, and
the oracle names the load. The port's own fixtures document the same rule.

The `sublet` arm must fault at the labelled read; the `spatial` arm completes,
reading another object's bytes — the report's crash was the call through a
function pointer read from exactly such reoccupied storage.

## Not yet done

- No `before.c`, so no host `native-detect` arm.
