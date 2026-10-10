# c702b44a01 — USB HID: two OUTPUT fields share one usages array, freed twice

## The defect

`parse_report_descriptor` builds a report descriptor's fields in the file scope. Each main item
appends the current `field` by value to `fields_in` or `fields_out`, and `field.usages` is a
`wmem_array_t *`. After an INPUT item the parser starts a fresh usages array; after an OUTPUT item,
before the fix, it did not. Every OUTPUT item therefore appended the same array, and the error path,
which frees each field's `usages`, freed it once per OUTPUT field. **The corpus's first double free.**

## Upstream defect

- **Fix:** `c702b44a01`, *"USB HID: Fix a double free."* Its message says to make sure a new
  `field.usages` array is allocated for `USBHID_MAINITEM_TAG_OUTPUT` as it is for
  `USBHID_MAINITEM_TAG_INPUT`, and it closes issue #16818.
- **CVE:** none assigned.
- **Live in our pin: NO.** A fix reversal, like every row in this corpus except 14, 15 and 18.

## The vulnerable code, quoted from the fix's parent

`c702b44a01^:epan/dissectors/packet-usb-hid.c` (blob `85a3abf4d0e4`):

```c
    wmem_allocator_t *scope = wmem_file_scope();                                  /* :3301 */
    ...
    field.usages = wmem_array_new(scope, sizeof(guint32));                        /* :3305 */
    ...
                    case USBHID_MAINITEM_TAG_INPUT:                               /* :3325 */
                        ...
                        field.usages = wmem_array_new(scope, sizeof(guint32));    /* :3334 */
                    ...
                    case USBHID_MAINITEM_TAG_OUTPUT:                              /* :3341 */
                        field.properties = hid_unpack_value(data, i, size);
                        if ((defined & HID_REQUIRED_MASK) != HID_REQUIRED_MASK)
                            goto err;
                        wmem_array_append_one(rdesc->fields_out, field);          /* :3347 */
                        defined &= HID_GLOBAL_MASK;
                        break;
    ...
err:                                                                              /* :3464 */
    for (unsigned int j = 0; j < wmem_array_get_count(rdesc->fields_in); j++)
        wmem_free(scope, ((hid_field_t*) wmem_array_index(rdesc->fields_in, j))->usages);
    for (unsigned int j = 0; j < wmem_array_get_count(rdesc->fields_out); j++)
        wmem_free(scope, ((hid_field_t*) wmem_array_index(rdesc->fields_out, j))->usages);  /* :3469 */
```

**Liveness, two-sided.** At `v4.6.8` (`e677bf052328`), the OUTPUT case at `:3781-3794` is followed by
the fix's two lines: `field.usages = wmem_array_new(scope, sizeof(uint32_t));` at `:3790`, then
`first_item = false;`. In the parent, nothing follows the append at `:3347` before `defined &= ...`.
The file scope is BLOCK at the pin, as the port's `src/shared/scopes.c:7` builds it.

## The reduction

The descriptor is INPUT, OUTPUT, OUTPUT, then a malformed item. Each `wmem_array_new` is reduced to
its two allocations: the header, sized as `wmem_array_t`'s members, and a one-element buffer. The
case then makes the three frees the error path makes, in order; the third is the defect.

In the native build, BLOCK's recycler is a circular list written over freed chunks. Re-adding the
shared array's chunk while it is already on the list relinks it to itself, so the INPUT field's
freed array drops off the list. The next descriptor's allocations never get it back, whereas the
fixed sequence reissues it at once. The native fix differential observes exactly that, and it is
deterministic: buggy DEFECT-REPRODUCED and fixed FIXED, three runs each in the 2026-10-11 prototype.

## Where a protected arm must stop

The defective access is the second `wmem_free`, a call into the allocator, so the site is the
allocator's handback, not `wm_defect_probe`.
- Under the chunk port (Capstone, domain and virtual), the site is `wm_chunk_of`'s
  `wm_handback_probe`, labelled `wm_widen_probe`.
- Under PoisonCap, which has the hooks without the chunk port, it is `wm_widen`'s load.

Both are declared in `case.json` (`fault_sites`, `fault_sites_why`) before any protected run.
