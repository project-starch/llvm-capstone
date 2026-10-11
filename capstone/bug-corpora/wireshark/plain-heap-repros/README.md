# wiretap plain-heap spatial defects — tshark's not-nested row

Upstream Wireshark defects whose access crosses **the `malloc` bound itself**. A sibling of
[`../wmem-repros`](../wmem-repros/README.md), deliberately separate: that corpus's boundary is
wmem — a chunk carved from a block the system allocator handed out — and all eighteen of its cases
cross a bound wmem owns. These cross the system allocator's own bound, **in wiretap, which does not
use wmem at all**: it calls `g_malloc` directly for its page buffers.

> **Looking for more cases?** [`../LIVE-CANDIDATES.md`](../LIVE-CANDIDATES.md) lists **92
> defects live at the 4.6.8 pin**, checked by content rather than by ancestry, with the two
> sharpest verified by hand against the pinned source. The reason they were not found earlier
> is the population: the previous triage searched `v4.6.8..release-4.6`, 83 commits, which by
> construction cannot see a defect whose fix was **never backported**. `v4.6.8..master` is
> 4,401. The checker is committed beside the list as `../check-liveness-at-pin.py`.


The mirror of [`../../memcached/plain-heap-repros`](../../memcached/plain-heap-repros/README.md),
built for the same reason: tshark's not-nested spatial row stood empty because the hunt had required
its candidates to be **live at the pin**, which no document asks for and which most cases in this
tree contradict.

## Shapes

| shape | cases |
|---|---|
| a length read from the file makes a copy run past the end of a `g_malloc`'d page buffer | 0 |
| an error path indexing the second array with the FIRST array's index | 1 |
| a copy sized by a length counting a header the source does not hold | 2 |
| a scan with no case for its terminator | 3 |
| a buffer sized for its payload then consumed as a C string | 4 |
| a fully-written buffer searched by a terminator-seeking function | 5 |
| a two-byte decode entered on the last byte | 6 |
| a decode buffer sized for its output, read as a terminated string | 7 |
| a fixed-length comparison applied to a shorter operand | 8 |
| an unsigned length-minus-one that wraps before the indexing | 9 |
| a copy clamped to the capacity, with the terminator past it | 10 |
| a tail reservation one byte short of its own worst case | 11 |

## The case

| case | upstream | the crossing |
|---|---|---|
| **0** | `19c51d27b9` `wiretap/netscaler.c` | `memcpy(dst, &nstrace_buf[offset], caplen)` with `caplen` a 16-bit field straight from the file — out of a `g_malloc(NSPR_PAGESIZE)` of **8192 bytes**, running **65471 bytes past** it |
| **2** | `381681583b` `wiretap/pcapng.c` | a copy of `size` = `sizeof(uint32_t) + stringlen` = **19** bytes out of a `calloc` of **16** -- the 4-byte PEN is counted in the option length but is not in the string buffer -- running **3 bytes past** it |
| **3** | `c556b648aa` `wsutil/ws_strptime.c` | the timezone scan has no arm for the terminator, so it consumes the NUL of an empty string and reads the byte after it -- **1 byte past** a `calloc` of **1** |
| **4** | `87803328179` `wiretap/blf.c` | `g_try_malloc0(textLength)` is filled completely by `blf_read_bytes`, then walked by `g_strsplit_set` -- **past the end** of a **16-byte** allocation |
| **5** | `140aad08e081` `wiretap/nettrace_3gpp_32_423.c` | `g_malloc(packet_size + 12)` is written in full, then `strstr` searches it -- **past the end** of a **16-byte** allocation |
| **6** | `3aad1ef236e6` `epan/charsets.c` | `(*c & 0xf0) == 0xc0` then `c[1]`, which at `i == length - 1` is `ptr[length]` |
| **7** | `e2ca71beaed2` `epan/uat.c` | `g_malloc(in_len/2)` is filled completely and then read as a C string -- **past the end** of a **16-byte** allocation |
| **8** | `0cae98570ebc` `ui/commandline.c` | `memcmp(elem_data, search_data, strlen(search_data))` reads the PREFIX's length from the shorter stored option |
| **9** | `bf123efe154d` `epan/uat.c` | `strptr[len-1]` with `len == 0` and `len` a `guint` -- the index wraps to **4294967295** |
| **10** | `e9b933473e8f` `epan/to_str.c` | `copy_len` clamped to `buf_len`, so `buf[copy_len] = '\0'` writes **1 byte past** a **16-byte** buffer |
| **11** | `4b15bf76a7f7` `epan/to_str.c` | 15 bytes reserved for a worst case of `1 + 10 + 4 + 1 = 16`, with `g_snprintf`'s would-be length advancing the cursor past the end |

## Measured, 2026-10-06

[`results/20261006-native-plain-heap/`](results/20261006-native-plain-heap/result-lines.txt):

| arm | buggy | fixed |
|---|---|---|
| `native-fix-differential` | `cap=8192 touched=73663 extent=65471 crossed=1` → DEFECT-REPRODUCED | same `cap`/`touched`/`extent`, `crossed=0` → FIXED |
| `native-detect` (ASan) | **`heap-buffer-overflow`, READ of size 1, 0 bytes after 8192-byte region** | silent, exit 0 |

`cap`, `touched` and `extent` are identical on both arms: the guard **refuses the record**, it does
not change its size. That is what makes the single difference attributable.

**`sublet-chunks` is a required arm here, and its prediction is that the port changes nothing.**
This is the one tshark row the inner-allocator port is irrelevant to — the object is not a wmem
chunk but a direct `g_malloc` — which is exactly why the row belongs in the inventory beside the
five wmem cases the port *does* discriminate.

**ASan reports this one**, unlike the wmem and sub-object corpora, for the reason that defines this
corpus: the crossing leaves the allocation, so a redzone sits where it lands.

## Reduction

The case probes **only the first crossing byte**. The unreduced `memcpy` spans 65471 bytes past the
allocation; walking all of it would be an unbounded read through whatever follows, which is a
property of the defect rather than of the reduction. The magnitude is reported as `extent=` on both
arms so it is never lost.

## Running it

```sh
bash runners/run-native.sh [OUT_DIR]
```

Exit 0 means the plain pair reproduced **and** the sanitiser fired on the buggy arm **and** stayed
silent on the fixed one. Exit 75 is an infrastructure failure and is never a verdict.
