# 48a00fd55671 — ftype: rework val_from_unparsed to avoid double free.

## The defect

`val_from_unparsed()` freed the old value up front with `string_fvalue_free(fv)` and then, on the non-byte-string path, delegated to `val_from_string()`, which frees the old value again before assigning. `string_fvalue_free()` does not NULL the field, so the second free hits the same pointer. The fix moves the up-front release into the branch that does not delegate.

## Upstream defect

- **Fix:** `48a00fd55671`, *"ftype: rework val_from_unparsed to avoid double free."*, `epan/ftypes/ftype-string.c`.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The pin's entry point performs no up-front release.

## The vulnerable code, quoted from the fix's parent

```c
	/* Free up the old value, if we have one */
	string_fvalue_free(fv);

	/* Does this look like a byte-string? */
	fv_bytes = fvalue_from_unparsed(FT_BYTES, s, TRUE, NULL);
	...
	/* Just turn it into a string */
	return val_from_string(fv, s, err_msg);
```

## The fix

```c
	fv_bytes = fvalue_from_unparsed(FT_BYTES, s, TRUE, NULL);
	if (fv_bytes) {
		/* Free up the old value, if we have one */
		string_fvalue_free(fv);
	...
	} else {
		/* Just turn it into a string */
		return val_from_string(fv, s, err_msg);
	}
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows. The object is a plain allocation because upstream's is
— this code is outside `epan`'s wmem scopes.

**Reduced:** no field values and no display filters. One allocation stands for the string, the helper is reduced to the free it performs, and the delegate's release is a read through the uncleared field.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing. Nor upstream reachability of
the specific sequence chosen here.
