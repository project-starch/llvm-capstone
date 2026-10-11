# Conversion-buffer overrun in regexp match/split

Upstream commit `e91dcfccaa`, "Fix potential buffer overrun in regexp
match/split functions". It postdates the 17.5 pin, so the pin is affected.

Upstream's own account:

> `setup_regexp_matches()` sizes the buffer used to convert matched substrings
> back from `pg_wchar` form at the smaller of `maxlen*eml` and the original
> string's byte length, on the assumption that such a conversion cannot produce
> more bytes than the string it came from. That assumption holds only for
> validly encoded input. But `pg_mb2wchar_with_len()` silently accepts bytes
> that are invalid in the database encoding, turning each such byte into one
> `pg_wchar`, and converting that back can take more bytes than the input did.
> A string made of such bytes therefore overruns the conversion buffer by up to
> its own length.

`regexp_match()`, `regexp_matches()`, `regexp_split_to_table()` and
`regexp_split_to_array()` are all affected.

## Established on the pin by inspection

`src/backend/utils/adt/regexp.c`:

- `:1444` -- `wide_len = pg_mb2wchar_with_len(VARDATA_ANY(orig_str), wide_str,
  orig_len);` -- no encoding validation; one invalid byte becomes one
  `pg_wchar`.
- `:1579` -- `int64 maxsiz = eml * (int64) maxlen;` -- the honest worst case.
- `:1592-1593` -- `if (maxsiz > orig_len) conv_bufsiz = orig_len + 1;` -- the
  worst case is discarded in favour of the subject's own byte length. This is
  the defect.
- `:1597` -- `matchctx->conv_buf = palloc(conv_bufsiz);`
- `:1644-1648` (match path) and `:1793-1815` (split path) -- each calls
  `pg_wchar2mb_with_len(..., buf, eo - so)` and then checks the result only
  with `Assert(len < matchctx->conv_bufsiz)`.

That `Assert` is the same shape as case 05's `Assert(*data > 0xC0)`: it
documents the invariant and disappears from a non-assert build, which is what
the pin's server is.

## Why it is nested

`conv_buf` is a `palloc` chunk in the per-call memory context. The overrun is
therefore bounded first by the aset block the chunk sits in, not by a `malloc`
bound -- which is precisely the asymmetry this corpus is built to measure
against `c-repros`.

## Trigger

A subject string made of bytes that are invalid in the database encoding, in a
database whose encoding is multibyte (so that `eml > 1`). `convert_from(...,
'SQL_ASCII')` inserts such bytes without the validation a literal would get.
See `trigger.sql`.
