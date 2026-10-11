# 0261fd7da6 — the HTTP Range cursor is advanced past its file-scope chunk

## The defect

`dissect_http_message` copies the `Range:` header value into **file scope** and then advances the
cursor by a **fixed** amount — 6 to skip `"bytes="`, or 8 at the second site — without ever checking
that the value is that long. A short value puts the cursor past the end of the `wmem` chunk, and
`strtok`/`strtoul` parse on from there.

The read does **not** leave the process allocation. One `g_malloc` hands the wmem file scope a whole
block and every chunk is carved from inside it, so the cursor lands in the *next chunk* of the same
block. That is what makes this a nested-allocator spatial defect rather than an ordinary heap
overread: at `malloc` granularity the access is in bounds and entirely legal.

## Upstream defect

- **Fix:** `0261fd7da6`, 2024-04-19 — *"http: Fix buffer overflow, use after free in HTTP Range"*.
  Upstream's subject names two defects in the same code; **only the buffer overflow is reduced
  here.**
- **CVE:** none assigned.
- **Live in our pin:** **no.** See `case.json`'s `live_proof` — the vulnerable skips have zero
  occurrences in `v4.6.8:epan/dissectors/packet-http.c`, the fix's RFC 9110 range-set commentary is
  present, and the pin parses with bounds-aware `ws_strtou64(pos, &pos, …)` at `:3936` and `:3955`.
  So this is a **fix-reversal** case, as every memcached and FFmpeg corpus case is.
- **First shipped:** the pointer-skip parse predates the 4.6 series; the fix landed on master in
  2024 and reached the 4.6 branch before `v4.6.8`.

## The vulnerable code, quoted from the fix's parent

`0261fd7da6^:epan/dissectors/packet-http.c:3736-3752`:

```c
                guint8  *first_range_num_str = NULL;
                unsigned long first_range_num = 0;

                /* Get the first range number */
                first_range_num_str = wmem_strdup(wmem_file_scope(), value);
                if (first_range_num_str) {
                        first_range_num_str += 6;  /* Move the pointer past "bytes=" */
                        first_range_num_str = strtok(first_range_num_str, "-");
                        first_range_num = strtoul(first_range_num_str, NULL ,10);
                }
                if (first_range_num == 0) {
                        /* The first number of the range is missing or '0'. So we'll
                        * use the second number in the range instead."
                        */
                        char *str = wmem_strdup(wmem_file_scope(), value);
                        str += 8;
                        first_range_num = strtoul(str, NULL ,10);
                }
```

The `if (first_range_num_str)` guard checks the **allocation**, not the **length** — which is
exactly the shape that reads as a bounds check and is not one. Neither site has any length test.

## The fix

The fix replaces both pointer skips with an RFC 9110 `range-set` parse that walks the value with
`ws_strtou64(pos, &pos, …)`, so the cursor can never pass the terminator. It also adds the ABNF it
is implementing as commentary, which is what makes the fix findable in the pinned tree.

## What is real here, and what is reduced

**Real:** the allocator. `wmem`'s core and its `BLOCK` allocator from the pinned 4.6.8 release, so
the chunk geometry the case depends on — a block from one `g_malloc`, chunks carved inside it — is
a property of Wireshark's allocator and not of this driver.

**Reduced:**

- `wmem_strdup(wmem_file_scope(), value)` is written as its own `wmem_alloc` + `memcpy`, because the
  corpus seam includes `wmem_core` only. That is what `wmem_strdup` does.
- The `strtok`/`strtoul` parse is reduced to its **first read**, the byte the cursor lands on. The
  labelled probe is that read.
- The `+= 8` site is the one modelled, not `+= 6`. With a 6-byte value the `+= 6` cursor lands in
  the chunk's alignment padding; `+= 8` lands at or past the neighbour's first byte, so the read has
  a pattern to find. Both sites are the same defect and the same bound.
- The value is `"bytes"` (6 bytes with the NUL). Upstream's trigger is any `Range:` value shorter
  than the skip.
- No dissector, no capture file, no conversation tracking. Reaching the site in place needs all of
  `epan`.

**The triggering condition is created, not assumed.** The case asserts, *before* the marker, that
the neighbour chunk sits above the string's chunk, that the cursor is past the chunk (`8 >= 6`) and
that it is still inside the block (below the neighbour's end). A run on an allocator whose geometry
differs therefore exits 75 as an infrastructure failure rather than quietly measuring nothing.

## What the run establishes, and what it does not

**Would establish**, when run: that the discriminating arm here is `sublet-chunks` and not `sublet`,
and that the fault is a **bounds** fault (cause 5) rather than revoked authority (cause 24). Both
differ from every other row in this corpus.

**Does not establish:**

- **Reachability.** That a capture can deliver a short `Range:` value to this site is not proven
  here; `packet-http.c` is in the port's dissector whitelist, but that is a build fact, not a
  reachability proof.
- **Anything about the pin.** The fix is in `v4.6.8`; this is a fix-reversal.
- **Anything about the `+= 6` site** beyond its being the same defect.
- **Anything on silicon.** Cause 5 under capstone-qemu is a bounds diagnostic; the deployed silicon
  is a separate question (ISSUES Q-11).

## Not yet done

- The run. `case.json`'s `status` records that every arm is a declared oracle and none is a
  measurement, and names the pair to run first: `spatial` vs `sublet-chunks` with `WM_CHUNKS=ON`.
- `native-detect` needs its own positive control before a silent ASan arm means anything — a read
  past the **end of the block**, which ASan must report. Without it, "ASan said nothing" is a
  statement about the instrument.
- `poisoncap-*` and `cheribsd-revocation` are predictions; neither platform is on this host.
