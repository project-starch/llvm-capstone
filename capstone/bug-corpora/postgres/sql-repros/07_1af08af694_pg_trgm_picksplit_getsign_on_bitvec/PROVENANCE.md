# pg_trgm picksplit applies GETSIGN() to a bare bit vector

Upstream commit `1af08af694`, "Fix pg_trgm's picksplit function with all-true
datums", CVE-2026-14678, backpatched through 14. The 17.5 pin is affected.

Upstream's own account:

> The `CACHESIGN.sign` field is a `BITVECP`, not a `TRGM`, so you should not
> use `GETSIGN()` on it. You don't get a compiler warning because the
> `GETSIGN()` macro includes a cast. It resulted in a bogus read beyond end of
> buffer, which would cause bad split decisions or a crash if you're very
> unlucky.

## Established on the pin by inspection

The pin's own code contains both the correct and the incorrect use of the same
field, which is what makes this one unambiguous.

`contrib/pg_trgm/trgm_gist.c`:

- `:742-746` -- `typedef struct { bool allistrue; BITVECP sign; } CACHESIGN;`
  The field is a bare bit vector. `trgm.h:83` -- `typedef char *BITVECP;`
- `:824` -- `cache_sign = palloc(siglen * (maxoff + 1));`
- `:827` -- `fillcache(&cache[k], GETENTRY(entryvec, k), &cache_sign[siglen * k],
  siglen);` -- each entry's `sign` is a `siglen`-byte slice of that one
  allocation, with no header of any kind.
- `:904`, `:917` -- `hemdistsign(cache[j].sign, GETSIGN(datum_l), siglen)`.
  The field is passed **directly**, and the sibling argument `datum_l` *is*
  wrapped, because that one is a real `TRGM`. This is the correct use.
- `:900`, `:913` -- `GETSIGN(cache[j].sign)`. The same field, now wrapped.

`contrib/pg_trgm/trgm.h`:

- `:74` -- `#define TRGMHDRSIZE (VARHDRSZ + sizeof(uint8))`
- `:106` -- `#define GETSIGN(x) ( (BITVECP)( (char*)x+TRGMHDRSIZE ) )`

So `:900` reads `siglen` bytes starting `TRGMHDRSIZE` bytes into a slice that
is only `siglen` bytes long. The read runs `TRGMHDRSIZE` bytes past that
entry's slice; when `j` is the last entry it runs past the end of the whole
`cache_sign` allocation. The cast in the macro is why the compiler is silent.

## Reachability

The defective expression sits in the branch taken when
`(ISALLTRUE(datum_l) || cache[j].allistrue)` is true and the inner
`cache[j].allistrue` is false -- that is, the split seed is all-true and the
entry being measured against it is not. Hence upstream's title: it needs
all-true datums present in the same split. `trigger.sql` builds an index over a
mixture of long varied strings (which saturate the signature) and short ones
(which do not).

## Why it is nested

`cache_sign` is a `palloc` chunk in the index build's memory context. The
overread is `TRGMHDRSIZE` = `VARHDRSZ + 1` bytes, which is small, so the
expectation is that it stays inside the surrounding aset block and a
`malloc`-granular mechanism cannot see it. That expectation is recorded here
as an expectation and will be replaced by what the three arms actually report.
