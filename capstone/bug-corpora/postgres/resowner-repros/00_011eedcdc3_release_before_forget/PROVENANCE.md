# 011eedcdc3 -- the upstream hunk, and what this case reduces

Upstream fix `011eedcdc3fe04000c46881333aaa0ac70a1aa1f`, 2026-06-22, on
`REL_17_STABLE`; first released in **17.11**. Master's copy is `ac6a58a700`.
Subject: *Fix unsafe order of operations in ResourceOwnerReleaseAll().*

## The hunk

```diff
@@ -387,13 +390,20 @@ ResourceOwnerReleaseAll(ResourceOwner owner, ResourceReleasePhase phase,
 			elog(WARNING, "resource was not closed: %s", res_str);
 			pfree(res_str);
 		}
-		kind->ReleaseResource(value);
+
+		/*
+		 * Update stored count to forget the item before calling its
+		 * ReleaseResource method.  This avoids double-free crashes in case an
+		 * error gets thrown within ReleaseResource.
+		 */
 		nitems--;
+		if (using_arr)
+			owner->narr = nitems;
+		else
+			owner->nhash = nitems;
+
+		kind->ReleaseResource(value);
 	}
-	if (owner->nhash == 0)
-		owner->narr = nitems;
-	else
-		owner->nhash = nitems;
 }
```

Upstream's own words for the consequence:

> If ReleaseResource throws an error then the subsequent abort cleanup will
> come back here and try to release that item again, possibly leading to a
> double-free or similar crash, and in any case risking an infinite error
> cleanup loop.  This mistake explains why the pgcrypto bug just fixed in
> 80bb0ebcc led to a crash rather than something more benign.

That last sentence is why this case has a sibling: case 01 is the pgcrypto
half, and the two together are one chain.

## Why the second release is reachable at the pin

Two facts, both read out of `REL_17_5` rather than inferred:

1. `ResourceOwnerReleaseAll` writes the count back only **after** the loop, so
   an error inside `ReleaseResource` leaves `owner->narr` describing an array
   that still contains the item just released.
2. `ResourceOwnerReleaseInternal` sets `owner->releasing = true` inside
   `if (!owner->releasing)` and never clears it. On the second entry the guard
   is skipped -- so `ResourceOwnerSort` does not run again -- but
   `ResourceOwnerReleaseAll(owner, phase, isCommit)` still does.

So the item is found a second time and the kind's `ReleaseResource` runs on it
again. Nothing in the pin stops it: the double-pfree detector `0c8b4e9cfc`
arrived in 17.10 and applies to memory-context chunks, not to this.

## What is real here and what is reduced

**Real, compiled unmodified from the pin:** `resowner.c` itself -- the subject
-- and the seven memory-manager files it allocates its owners and arrays
through (`aset mcxt generation slab bump alignedalloc memdebug`), plus
`src/common/hashfn.c` and `src/port/qsort.c`, which `ResourceOwnerSort` and the
owner's hash table need.

**Reduced:** the resource kind. A `ResourceOwnerDesc` and a payload are what
every consumer of `resowner.c` supplies, and the case supplies one whose
`ReleaseResource` frees its payload and then fails. Free-then-fail is not an
invented order: it is exactly pgcrypto's, where `EVP_CIPHER_CTX_free` precedes
the `ResourceOwnerForget` that raises (case 01).

**Reduced:** the error itself. `elog(ERROR)` in a backend reports and then
`siglongjmp`s to the enclosing cleanup, and `shared/pg_support.c` implements
`errfinish` as that `siglongjmp`. The second release is driven the way
`AbortTransaction` drives it, on the same owner. `shared/driver.c` bounds the
loop at one re-entry; a backend would keep going, which is the "infinite error
cleanup loop" upstream names.

**Not modelled at all:** the lifetime rule under test. It is resowner.c's.
