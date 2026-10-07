# Small overread in SASLprep validation

Upstream commit `5d61bdd114`, "Protect against small overread in SASLprep
validation". The 17.5 pin predates it and has neither of the two checks the
commit adds.

## Why it is in this corpus

`src/common/saslprep.c` is compiled twice. Under the backend it allocates with
`palloc`; under `FRONTEND` -- which is how it enters `libpq` -- `saslprep.c:48`
defines `ALLOC(size)` as `malloc(size)`. The string this case overreads is the
application's password, reached through `libpq`, so it is a libc allocation and
the defect is non-nested.

That dual compilation is worth stating plainly, because it is the cleanest
control this study has: the same source line is nested when the backend runs it
and non-nested when `libpq` runs it, with nothing else changed.

## Established on the pin by inspection

`src/common/saslprep.c:1013-1022`:

```c
while (*p)
{
    l = pg_utf_mblen(p);

    if (!pg_utf8_islegal(p, l))
        return -1;

    p += l;
    num_chars++;
}
```

`pg_utf_mblen` inspects only the lead byte and returns the length that byte
*declares* -- up to 4. `pg_utf8_islegal(p, l)` then reads `l` bytes. If the
string ends before `p + l`, the read crosses the terminating NUL and leaves the
allocation. `while (*p)` cannot stop it: the condition is tested at the lead
byte, which is non-NUL precisely in the failing case.

Upstream's fix tracks the bytes that remain (`size_t len = strlen(source)`) and
tests `len < l || !pg_utf8_islegal(p, l)`.

## Reachability from libpq

`src/interfaces/libpq/fe-auth-scram.c:123`, inside `scram_init`, calls
`pg_saslprep(password, &prep_password)` on the password the application
supplied. The surrounding lines are themselves direct libc calls --
`malloc(sizeof(fe_scram_state))` at `:106`, `strdup` at `:115`, `free` at
`:118` -- which is what makes this path a libc client end to end.

## Trigger

A password whose final character is a truncated UTF-8 sequence: a lead byte
such as `0xE2` (declaring three bytes) as the last byte before the NUL. The
overread is up to three bytes and happens entirely client-side, before any
byte reaches a server.
