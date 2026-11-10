# PostgreSQL ResourceOwner defects

Two upstream double-free defects at PostgreSQL's **resource-owner** boundary,
both live at the 17.5 pin, reduced to programs that run against PostgreSQL's
own `resowner.c` -- compiled unmodified from the pin, gated by sha256.

`ResourceOwner` is the layer above the allocator: it holds references to
objects whose release is a callback, and releases them in phase and priority
order when a transaction or portal ends. That makes it a second lifetime
manager in the same program, and a different boundary from the memory contexts
in [`../mmgr-repros`](../mmgr-repros/README.md).

## The cases

| # | upstream fix | first released | shape | object freed twice | live in pin |
|---|---|---|---|---|---|
| 0 | `011eedcdc3` | 17.11 | double release through an item the owner never forgot | the resource's libc payload | **yes** |
| 1 | `2aa6be6e64` | 17.11 | double release through an item the owner never forgot | `od->evp_ctx`, from OpenSSL's allocator | **yes** |

They are one chain, split the way upstream split it. Case 00 is why an error
during release becomes a double free: `ResourceOwnerReleaseAll` calls the
kind's `ReleaseResource` and only then decrements the count, so abort cleanup
finds the item still there. Case 01 is where the error comes from: pgcrypto's
callback leaves `od->owner` set, so `free_openssl_cipher` calls
`ResourceOwnerForget` during release and real `resowner.c` refuses it with
`ResourceOwnerForget called for ... after release started`. Upstream says so
itself, in 00's commit message: *"This mistake explains why the pgcrypto bug
just fixed in 80bb0ebcc led to a crash rather than something more benign."*

## Why ASan counts here

In `../mmgr-repros` a sanitizer's silence measures nothing, and that corpus
says so at length: `pfree` returns a chunk to its context's free list and never
reaches `free`, so there is no event for ASan or for a malloc-granular revoker
to act on.

**This corpus is the other case, and that is the point of having it.** The
object freed twice is libc's -- OpenSSL's `EVP_CIPHER_CTX` in case 01, a plain
allocation in case 00 -- so both frees are events the system allocator sees.
glibc's own tcache check raises `SIGABRT` at the second free, and ASan names
it with the probe frame. Here a silence from a capability arm would be a
finding rather than a restatement, which is what makes the arms worth running.

It is also the backend's counterexample to "PostgreSQL is all nested". The
`OSSLCipher` struct *is* a memory-context chunk; it is simply not the object
that gets freed twice, because the pin's `free_openssl_cipher` frees the handle
before the forget that raises, so `pfree(od)` is never reached. That order is
quoted in `01_.../PROVENANCE.md`.

## What is real and what is reduced

**Real, compiled from the pin and gated:** `resowner.c` -- the subject -- the
seven memory-manager files it allocates through, `src/common/hashfn.c` and
`src/port/qsort.c`. `check-sources.sh` holds all ten to their sha256 at the
pin and `build-host.sh` refuses to build without it. All ten are
byte-identical in `REL_17_0` and `REL_17_5`, so the memory-context port's
17.0 allocator pin and this corpus's 17.5 defect pin do not disagree about a
single byte this corpus compiles.

**Reduced:** the resource kind, which is what every consumer of `resowner.c`
supplies, and in case 01 the two pgcrypto functions, transcribed in shape with
the EVP handle modelled by a libc allocation. Running pgcrypto in place needs
the px framework and OpenSSL; what the defect needs is a release callback that
frees a libc object and then forgets.

**Reduced, and faithful:** the error. `elog(ERROR)` in a backend reports and
then `siglongjmp`s, so `shared/pg_support.c` implements `errfinish` as that
`siglongjmp` and `shared/driver.c` drives the second release the way
`AbortTransaction` does. The driver bounds the re-entry at one; a backend would
keep going, which is the "infinite error cleanup loop" upstream warns about and
which case 01's control run shows twice.

## Arms

| arm | what it establishes | state |
|---|---|---|
| `native-detect` | the defect reproduces at the pin and the host allocator sees it | **run** 2026-11-10 |
| `host-asan` | the double free named, with the fault in the labelled frame | **run** 2026-11-10 |
| `cheribsd-revocation` | whether the guest's own libc revocation sees it | not written |
| `virtual-malloc` | virtual Capstone, managers stock on mallocng | not written |
| `virtual-pg-pools` | the same with the port's patch 0003 | not written |

Each case declares the three unrun arms `not written` rather than silent: an
arm that has not run is not a MISSED.

## Building and running

    PG_SRC=<a configured PostgreSQL 17 tree> \
    PG_BUILD=<its build dir, if out of tree> \
    PG_PRISTINE=<the pin's ten .c files> \
      bash build-host.sh

`PG_SRC` supplies headers, `PG_PRISTINE` the sources that get compiled; it
defaults to `PG_SRC` and is separate because a tree configured for a
capability target carries port patches -- the memory-context port's inert
`#if SIZEOF_VOID_P > 8` guard in `aset.c` is one, and `check-sources.sh`
catches exactly that.

Each case is built twice. The defect build must reproduce; the
`-DRO_NEGATIVE_CONTROL` build withholds **only the second free**, keeps every
other operation including the error and the second release, and must complete.
A fault in the control would mean a fault in the defect run is not
attributable.

## Results

[`results/2026-11-10-host`](results/2026-11-10-host/README.md) -- result lines
only, as the project requires: a raw capture carries account names and build
paths, and twelve result lines are the better evidence.

## Not here

The sweep that found these two found twelve other temporal defects live at the
pin. [`SCREENED.md`](SCREENED.md) records each one and why it is not a case,
including four whose object is in static shared memory or an unmapped DSM
segment, where no allocator-based mechanism can see anything by construction.
