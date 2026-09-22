# `fb504bc76c` — an authentication plugin name in pinfo->pool kept by the connection record

A use-after-free in Wireshark's MySQL dissector, reproduced against
Wireshark's own unmodified `wmem` block allocators.

## The defect

`mysql_dissect_auth_switch_request` reads the plugin name with
`tvb_get_string_enc(pinfo->pool, ...)` and stores the pointer in the
connection record, which is allocated in file scope and lives for the whole
capture. The pool is reset when the packet ends. The AuthSwitch response — by
protocol, the next packet — compares the stored name with `g_strcmp0`.

## Upstream defect

Upstream fix `fb504bc76c`, "MySQL: Correct scope for tvb_get_string_enc",
first tag v4.1.0. Reported as #19045 by the ASan fuzz job on master; the
closing note states the cause in its own words: the connection data is file
scope and the auth data is used after the current frame, so the string must
be file scope too.

- **CVE:** `NO VERIFIED CVE`.
- **Live in our pin:** no. At the 4.6.8 pin `packet-mysql.c:2095` reads the
  name with `tvb_get_string_enc(wmem_file_scope(), ...)`. The pre-fix shape is
  quoted below from the fix's parent.
- **First shipped:** the discussion attributes the AuthSwitch use to a change
  first tagged v4.1.0, the same release as the fix; the `pinfo->pool` store
  itself is older. Not established beyond that.

## The vulnerable code, quoted from the fix's parent

`git show fb504bc76c^:epan/dissectors/packet-mysql.c`:

```c
1347 typedef struct mysql_conn_data {
1371 	guint8 *auth_method;
1372 } mysql_conn_data_t;
...
4043 		conn_data = wmem_new0(wmem_file_scope(), mysql_conn_data_t);
...
3714 		conn_data->auth_method = tvb_get_string_enc(pinfo->pool, tvb, offset, lenstr, ENC_ASCII);
...
3745 	if (g_strcmp0(conn_data->auth_method,"caching_sha2_password") == 0) {
```

The reported trace:

```
READ of size 1 ... in strcmp
    #1 in mysql_dissect_auth_switch_response epan/dissectors/packet-mysql.c:3745
0x... is located 32 bytes inside of 43-byte region
freed by thread T0 here:
    ... wmem_free_all
    #6 in epan_dissect_reset epan/epan.c:589
previously allocated by thread T0 here:
    ... tvb_get_string_enc
    #9 in mysql_dissect_auth_switch_request epan/dissectors/packet-mysql.c:3714
```

## The fix

```diff
-		conn_data->auth_method = tvb_get_string_enc(pinfo->pool, tvb, offset, lenstr, ENC_ASCII);
+		conn_data->auth_method = tvb_get_string_enc(wmem_file_scope(), tvb, offset, lenstr, ENC_ASCII);
```

Applied at all three stores — greeting, login and AuthSwitch request — so the
string shares the record's lifetime.

## What is real here, and what is reduced

**Real:** the allocator, and the reset that ends the string.

**Reduced:** the consumer. The record is one field in a file-scope object;
the case allocates the 43-byte name from the packet pool, stores it there,
resets the pool, and reads the first byte where `strcmp` begins. No
reoccupation is asserted.

## What the run establishes, and what it does not

The `sublet` arm must fault at the labelled read; the `spatial` arm completes.
The request/response pairing makes the cross-packet read a property of the
protocol rather than of fuzzed state.

## Not yet done

- No `before.c`, so no host `native-detect` arm.
