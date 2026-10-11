# Case 0 — `ddee3e2`, the `--auth-file` heap buffer overflow

## Upstream

`ddee3e2`, subject **"Fix minor severity heap buffer overflow reading `--auth-file`"**, one file,
`authfile.c`, 14 insertions and 8 deletions.

The allocation, by line, read from the commit itself rather than from the file it touches:

| ref | `authfile.c` | text |
|---|---:|---|
| `ddee3e2^` | 44 | `auth_data = calloc(1, sb.st_size);` |
| `ddee3e2` | 44 | `auth_data = calloc(1, sb.st_size + 1);` |
| `1.6.45` (our pin) | 50 | `auth_data = calloc(1, sb.st_size + 2);` |

## The defect

`authfile_load` sizes the buffer to the file's **exact** length and then parses it with `fgets`:

```c
auth_data = calloc(1, sb.st_size);                        /* :44 */
char *auth_cur = auth_data;
while ((fgets(auth_cur, MAX_ENTRY_LEN, pwfile)) != NULL) { /* :50 */
    for (x = 0; x < MAX_ENTRY_LEN; x++) { ... auth_cur[x] ... }
```

`MAX_ENTRY_LEN` is 256 (`authfile.c:16`, unchanged at the pin). `fgets` stores the line's bytes
**and a terminating NUL**, and nothing relates its size argument to how much of the allocation
remains. A file whose last line runs to EOF and exactly fills the buffer therefore has its
terminator written at offset `sb.st_size` — one byte past a `calloc(1, sb.st_size)`.

That the last line can run to EOF is not hypothetical: the loop has a `// EOF` break at
`authfile.c:83` for exactly that case.

## What is reduced, and what is not

- **The allocator is real.** `calloc` and `free` are the platform's own. The corpus's boundary is
  the system allocator, so there is nothing to port and nothing to model.
- **The arms differ by one term**, the `+ 1`. The line, its length and the write offset are
  identical in both; only the allocation's end moves. A changed reading is therefore attributable
  to the fix and to nothing else.
- **`fgets` is reduced to the write C requires of it** — for a line shorter than the size argument
  and ending at EOF, it stores the line's bytes and then a terminating NUL. The call is reduced
  rather than made so the crossing sits on this corpus's labelled `write_probe` instead of inside
  libc, which is what lets a sanitiser's report be required to land at a known frame. **Nothing
  about where the NUL lands is reduced; that is the defect.**
- **Left out:** the fix's other two hunks — an `auth_end` clamp on the `fgets` length, and a `'\0'`
  break in the entry scan. Both bound *later* lines; this case is reduced to the first crossing,
  which the `+ 1` alone removes. A second case could carry the multi-line shape.
- **No neighbour sentinel.** Two `calloc`s are not adjacent, and asserting that they were would be
  an assumption the allocator never makes. The arm that *sees* the byte is the sanitiser arm; the
  plain arm proves the **address**, by asserting that the terminator's offset equals the
  allocation's end in the buggy arm and is the last byte inside in the fixed one.

## Liveness

`live_in_pin: false`, and the proof is beside it in `case.json`: the fix is an ancestor of the
1.6.45 tag, and the pin carries `+ 2`. This is the tree's normal shape — **27 of its 33 cases** are
fix-reversals — and the convention is stated at `../../allocator-repros/README.md:132-135`: a fix
that precedes the pin is reconstructed by running the pre-fix consumer shape against the shipped
allocator.

That liveness is *recorded* rather than *required* is the correction of 2026-10-06: the not-nested
spatial row of `docs/ref/spatial-and-temporal-bug-inventory.md` stood empty because the hunt had
demanded liveness, which no document asks for.

## Measured

`results/20261006-native-plain-heap/`, both native arms two-sided:

```
plain  fixed  cap=10 touched=9 crossed=0   VERDICT FIXED
plain  buggy  cap=9  touched=9 crossed=1   VERDICT DEFECT-REPRODUCED
asan   buggy  heap-buffer-overflow, WRITE of size 1, 0 bytes after 9-byte region,
              at write_probe (shared/corpus.h:62), allocation at case.c:37
asan   fixed  silent, exit 0
```

**ASan reports this one**, and that is the contrast the corpus exists to draw. The sub-object
corpora record ASan blind for the opposite reason: there the crossing stays inside one allocation,
so no redzone sits where it lands. Here it leaves the `malloc` bound, and a redzone sits exactly
there.
