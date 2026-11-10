# 2aa6be6e64 -- the upstream hunk, and what this case reduces

Upstream fix `2aa6be6e64812ab3421b506c20b0e15078160e75`, 2026-06-22, on
`REL_17_STABLE`; first released in **17.11**. Master's copy is `020426268f`.
Subject: *pgcrypto: avoid recursive ResourceOwnerForget().* Upstream cites
bug #19527.

## The hunk

```diff
@@ -800,5 +800,8 @@ px_find_cipher(const char *name, PX_Cipher **res)
 static void
 ResOwnerReleaseOSSLCipher(Datum res)
 {
-	free_openssl_cipher((OSSLCipher *) DatumGetPointer(res));
+	OSSLCipher *cipher = (OSSLCipher *) DatumGetPointer(res);
+
+	cipher->owner = NULL;
+	free_openssl_cipher(cipher);
 }
```

Upstream's description:

> Raising an error within a function using an OSSLCipher object led to a
> complaint from ResourceOwnerForget and then a double-free crash, because
> ResOwnerReleaseOSSLCipher forgot to unhook the OSSLCipher object from its
> owner.  (The sibling logic for OSSLDigest objects got this right, as did
> every other ReleaseResource function AFAICS.)

## The chain, each link read out of REL_17_5

1. `resowner.h`'s contract for `ReleaseResource` says: *"The resource is
   implicitly removed from the owner, the callback function doesn't need to
   call ResourceOwnerForget."* pgcrypto's callback breaks it.
2. `contrib/pgcrypto/openssl.c:799-803` -- `ResOwnerReleaseOSSLCipher` passes
   the object straight to `free_openssl_cipher` with `od->owner` still set.
3. `contrib/pgcrypto/openssl.c:299-305`:

   ```c
   free_openssl_cipher(OSSLCipher *od)
   {
       EVP_CIPHER_CTX_free(od->evp_ctx);
       if (od->owner != NULL)
           ResourceOwnerForgetOSSLCipher(od->owner, od);
       pfree(od);
   }
   ```

   Note the order. The handle is freed **before** the forget.
4. `resowner.c`'s `ResourceOwnerForget` begins:

   ```c
   if (owner->releasing)
       elog(ERROR, "ResourceOwnerForget called for %s after release started", kind->name);
   ```

   so the recursive forget raises, and `pfree(od)` is never reached.
5. The error escapes `ResourceOwnerReleaseAll`, which had not yet decremented
   the count (case 00). Abort cleanup releases the same owner again, the item
   is still there, and `EVP_CIPHER_CTX_free` runs a second time on a handle
   OpenSSL has already freed.

**This is why the case is non-nested.** `od` itself comes from
`MemoryContextAllocZero` and would be a nested object -- but it is never
freed twice, because step 4 cuts the first pass short before `pfree(od)`. The
object freed twice is OpenSSL's. The corpus's earlier inventory called this
defect "mixed"; the order in step 3 settles it as non-nested, and that was
established by reading the pin, not by reasoning about the two allocators.

## What is real here and what is reduced

**Real, compiled unmodified from the pin:** `resowner.c`. It raises the error
itself, from the `elog` quoted in step 4 -- the case does not inject it. The
seven memory managers, `hashfn.c` and `qsort.c` are real too.

**Transcribed in shape:** the two pgcrypto functions above, with
`EVP_CIPHER_CTX_new`/`_free` modelled by a libc allocation and
`ro_free_probe`. Running pgcrypto in place needs the px framework and OpenSSL;
what the defect needs is a release callback that frees a libc object and then
forgets, and that is what the transcription is.

**Not modelled:** the trigger size. Upstream's reproducer is
`encrypt_iv(repeat('A',1073741308)::bytea, ...)` -- about 1 GB -- but the
defect is not about size. It is "raise an error while an OSSLCipher is live",
and at the pin the recursive forget supplies the error by itself, with no
large input and no OOM. That removes the blocker recorded in
`../HANDOFF.md`, which expected this case to need a cheap error path to be
found first; the error path is the defect.
