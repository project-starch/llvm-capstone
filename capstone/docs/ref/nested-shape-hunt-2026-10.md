# Nested-allocator shape hunt: FFmpeg, tshark, memcached (2026-10-10)

**Why this hunt.** The whole-corpus audit of 2026-10-10 compared our three programs' nested corpora
with the other programs' (`docs/ref/bug-corpora-cross-program-comparison.md`). Six defect shapes
appear in other programs' nested-allocator corpora and in none of ours:

| shape | what it is |
|---|---|
| S1 | a re-entrant callback that mutates storage inside a nested allocator |
| S2 | a double free into a nested allocator |
| S3 | use of a destroyed or foreign allocator instance |
| S4 | a write-after-free that corrupts the allocator's in-band free list |
| S5 | a realloc inside a nested allocator that leaves a stale pointer |
| S6 | a vacated slot reused with no allocator event |

None of the earlier hunts was aimed at these shapes. The temporal hunts filtered on lifetime wording
and the spatial hunts on overflow wording, so an S1–S6 defect was met only by accident.

## Method

**Sources.** Full-history clones under `/tmp/capstone/upstream/{ffmpeg,wireshark,memcached}`. Every
command ran as `git -c protocol.allow=never`. Pins:

| program | pin | commit |
|---|---|---|
| FFmpeg | `n9.0.1` | `bf1b838f2ab8` |
| Wireshark | `v4.6.8` | `e677bf052328` |
| memcached | `1.6.45` | `2d51e3647` |

**Gates, in order:**
1. the commit exists (`cat-file -t`, against a control);
2. the class is read from the diff;
3. the allocation site gives a nested allocator;
4. the shape holds at that allocator;
5. liveness at the pin is checked two-sided and quoted (a fix reversal is admissible if recorded, as
   wmem-repros 16 and 17 already are);
6. the allocator exists at the pin;
7. the caller is named on the executed path;
8. the protection arms can tell the cases apart: no NULL dereference, and no "faults on every arm".

**Instrument limits, stated so the zeros can be weighed:**
- The Wireshark clone is blobless, with 60,782 of about 524,875 blobs present.
  - A `git show` that touches a missing blob blocks for about 39 s before failing, even with
    `protocol.allow=never` and `GIT_NO_LAZY_FETCH=1`.
  - The fast check is `git cat-file --batch-all-objects --batch-check --unordered`, which lists
    every present object in 0.3 s. Compare a commit's `diff-tree -r` blobs against that list before
    reading it.
  - That check was negative-tested: the pin and a known-missing commit read as expected.
  - Fix siblings whose blobs are missing are listed below as UNRESOLVED, not guessed.
- The memcached clone is missing 1,208 objects; there, presence was checked with
  `rev-list --objects --missing=print`.
- **Recall check (memcached).** The four memcached searches were run against the five temporal cases
  already built. Each search alone finds 2 to 4 of them; together they find all five.
- **Filter leak (FFmpeg).** The FFmpeg subject-line exclusion filter hid one commit (`7d38486975`),
  which a second pass caught. Its "nothing found" is bounded by that filter.

## Results

### FFmpeg: no case passes; one structural note

About 60 commits were read across all shapes. Each failed the class gate, was not in pool memory, or
cannot separate the arms (a NULL dereference, or a data race). None reached the liveness gate. This
matches `docs/ref/ffmpeg-pool-consumer-defects.md`: FFmpeg's use-after-free fixes are overwhelmingly
on plain `av_malloc` state.

**S4 is structurally possible in `AVRefStructPool`, contrary to the hunt's premise.**
- `AVBufferPool` keeps its free-list link in a separate entry.
- At the pin, `refstruct.c:230-231` stores the pool's free-list `next` in the object header, just below
  the payload. A write below a pooled object while it rests in the pool therefore corrupts the free
  list. That is the shape of the synthetic fixture 16 (`rs_underflow`). No upstream instance was found.

**A double unref at the pin:**
- On `AVBufferPool` it is a plain-heap double free before the pool is reached.
- On `AVRefStructPool` the second unref of a pooled object wraps its refcount silently (`:131`), and
  the next hand-out resets it (`:263`). After a re-issue, the second unref returns the new owner's
  object to the pool. Under Sublet the stale header access should fault. This is a synthetic
  prediction, not an upstream case.

**Open lead, not read.** VVC's per-CTU `cu_pool` and `tu_pool`. The searches over `vvc/dec.c` and
`vvc/refs.c` covered them; a dedicated read of `vvc/ctu.c` did not happen.

**Rejected.** Hash, shape, allocator and reason:

| commit(s) | shape | allocator | reason |
|---|---|---|---|
| `ef13a29d08`, `083a014746` | S3 | framepool | memory leak only |
| `9d6785d426` | S3 | AVBufferPool, FramePool shared across threads | hardening, no observed defect; needs real threads; code rewritten since |
| `fa77cb258b` | S3 | AVBufferPool, decode_error_flags | data race on an AVFrame field, not memory safety |
| `b645138a34` | S5 | mpegpicture tables, `av_buffer_make_writable` | racy read of a live buffer; no lifetime violation |
| `7ad13e173b` | S6 | mpegpicture `mbskip_table` | stale content in a reused table (determinism), no out-of-lifetime access |
| `3bb00c0a42`, `90bbe1e8e2` | S1 | hwcontext | backend state, not pooled; hwaccel only |
| `49838705a4` | S3 | pthread_frame | cleanup, no observed defect |
| `091341f2ab` | S2 | hwaccel_priv_data | plain-heap double free; hwaccel only |
| `6bbc22dc09` | S2 | swscale frame pool | a feature after the pin, no defect |
| `d9699464c3` | S2 | progress_frame_pool | assertion on an un-unreffed ProgressFrame; no lifetime violation |
| `855463c007` | S6 | h264 frame planes | a feature, not a fix |
| `efff3854f0`, `eaff36c973`, `d216b9debd` | S6 | vp9 segmap | threading logic, NULL retain |
| `4133db39b2`, `dd941af8ac`, `501d8eb62d`, `054dffd133` | S2/S3 | dts2pts node_pool | leaks, an assert, an overflow of a fixed array; all in the pin |
| `e417f939da` | S6 | vvc DPB slot | stale `fc->ref` reaches a vacated slot whose `progress` is NULL: a NULL dereference, and FrameProgress is not pooled |
| `49c3918c1a` | spatial | vvc `tab_dmvr_mvf` | out of scope: an index-geometry mismatch, not a lifetime shape |
| `b593abda6c` | S5 | pngdec last_picture | hardening; no stale alias |
| `1ee3c984b9`, `8732eb124e`, `3ca347900e` | S5 | frame planes | writability, logic, perf; the buffer is live throughout |
| `c37fb99abb`, `2d4d7df10c` | S6 | MPVPicture pool | output order and thread sync |
| `ccd391d6a3` | S2 | hevc DPB | a NULL dereference under frame threading only |
| `8c3b329da2` | S3 | h264 default_ref | a data race; values overwritten before use |
| `4feca2214a` | S6 | h264 ER cur_pic | 2014 code, rewritten; the pin repopulates before every dereference |
| `8d6014dbc6`, `90551b7d80` | S2/S3 | vvc PS, vp3 coeff_vlc | plain refstruct, no pool |
| `c34cb130b6`, `f068ce570f`, `90c6963dae` | S3 | 2011-12 avfilter picture pool | the code does not exist at the pin |
| `e036bb7899` | S2 | decoder pool | a compat path removed long before the pin |
| `7d38486975` | S6 | vvc DPB output | a leak |
| `99501b5015`, `6c12fe0dda`, `8b5ffaea64`, `edf5b777c9`, `265d39e551` | S6 | hevc/vvc DPB | bumping, POC and conformance logic; no stale alias |
| `6705ef6df6` | S6 | h264 hwaccel private | hwaccel only |
| `e70225e0a8`, `b0c77e5a12` | — | vvc PS refs | features |

The mpegts PES AVBufferPool has no lifetime fix in its history.

### memcached: no case passes; one value-oracle lead

About 57 commits were read. None gives an S1–S6 defect that is live or reversible at 1.6.45,
reachable with the page mover off and no proxy, and able to separate the arms.

**Blocked**, recorded rather than dropped:
- **Per-thread cache objects.** At the pin every `io_cache` allocation and both I/O queues belong to
  the proxy or extstore, and both are disabled in the port. This blocks `28f4abf`, `d315f07`,
  `04a8bef`, `1bcbfc0`, `7aa08ac` and `5267f14` (Lua). `28f4abf`'s own message describes a leak and a
  parser desync, not a use-after-free.
- **Page mover (slab rebalancer).** `324975c`, `0240dde`, `221c521` and `f600354` are blocked:
  - `slabs reassign` is gated at `proto_text.c:1250`;
  - the only reader of a free chunk's `prev` link is called from `slabs_mover.c:289`.

  Hooking the rebalancer in the port is what would unblock these, and the seven page-mover defects
  already deferred in `memcached-allocator-defect-triage.md`.

**Rejected:**
- **The response-bundle defect.** `53b4d74` is the only bundle fix in history, and it is a leak.
  Its stale `r[0].free` flag is cleared by a `memset` in `resp_start`, the only caller.
- **`f4983b2` stays rejected.** At the pin, eviction does release through `slabs_free`
  (`items.c:1206` → `:1241` → `:543` → `:360`), so the old "reused in place" reason only applied before
  about 2015. The same race at the pin is the unlocked-refcount family that case 04 already covers.
- **The rest** are NULL dereferences that fault on every arm, leaks, races in which every holder keeps a
  reference, or code that is gone at the pin.

**Lead outside S1–S6: `58d05f1` and a residual in its fix.**
- `58d05f1` fixed incr/decr reading a slab chunk's previous occupant on an empty value.
- At the pin the guard (`memcached.c:2250/2252`) rejects only values of at most two bytes, so a
  one-space value passes. The number parser then reads the previous occupant's digits.
- It is reachable through ASCII incr/decr, meta `ma` and the binary protocol.
- It rests on reading the source and has not been run.
- It would separate arms only by a returned value, never by a fault. That does not fit the corpus
  oracle ("spatial completes, sublet faults at the probe"), so it is a case only if a value oracle
  is admitted.

### Wireshark: two S2 candidates; one built as wmem-repros 22

About 184 commits worded as use-after-free, across all branches, were scanned for wmem free and
realloc calls in their diffs, and about 60 were triaged.

**Facts at the pin** (`wsutil/wmem/`, `epan/wmem_scopes.c`):
- The file scope, the epan scope and `addr_resolv_scope` are BLOCK. The packet pool (`epan.c:618`)
  is BLOCK_FAST.
- BLOCK's `wmem_block_free` writes a `wmem_block_free_t {prev, next}` into the chunk's own data
  (`:390-420`, `:486-555`). So a double free, or a write through a stale chunk pointer, corrupts the
  in-band free list (S2/S4). A normal-sized BLOCK free returns nothing to the system.
- BLOCK_FAST's free is a no-op, so no arm sees a BLOCK_FAST free.

| commit | shape | scope | liveness at the pin | status |
|---|---|---|---|---|
| `c702b44a01` (USB HID, #16818) | S2 double free (+S4 free-list corruption) | `wmem_file_scope()`, BLOCK | fix in the pin, so a **fix reversal**: the pin's `parse_report_descriptor` carries the fix's `field.usages = wmem_array_new(...)` after the OUTPUT append (`packet-usb-hid.c:3790`) | **built as wmem-repros 22** (see below) |
| `9fbd4e6fcd` (addr_resolv async DNS) | S2 + S4 + use-after-free read | `addr_resolv_scope`, BLOCK | fix in the pin; the fix marker `head = wmem_list_head(...)` is at `addr_resolv.c:573` | candidate, not built: c-ares callback synchrony UNRESOLVED |
| `692f9bed05` (RELOAD framing) | S3 | `pinfo->pool` (BLOCK_FAST), freed across allocators | fix in the pin | partial: its fix frees with the file scope what the packet scope allocated, UNRESOLVED; ASan-visible |
| `ec0ffc043f` (RRC strbuf) | S5 | strbuf scope UNRESOLVED, likely the packet pool | fix in the pin | weak unless the strbuf is file or epan scope |
| `615e4731e0` (conversation key) | file-scope key across files | file BLOCK, held by an epan map | fix in the pin | out: a file-scope leave returns wholly unused blocks to the system (ASan-visible), the reason ZigBee `030bf6ad01` was reclassified |

Why 9fbd4e6fcd was not built:
- the reduction would repeat case 22's double-free shape, plus the stale-chunk read that case 12
  already has;
- its second half depends on whether `ares_gethostbyaddr` can call back synchronously, which was not
  settled.

**Rejected:**
- `0defc09e5f` (HTTP): NOT LIVE. `req_list` is still a `GSList` at the pin (`packet-http.c:3972`);
  the `wmem_list` conversion (`6b239cf71aa4`) and the fix both postdate it.
- `1ccf4f3c73`, `e0c563c71e`, `819d392aff` (`foreach_remove`): API additions, no defect.
- `ba1084daac`, `b3142f0e5d`, `512adcb046`, `d1f64e9cbb`, `e375ace05a`: key-length, leak and
  defensive-realloc fixes.
- `22e0f916f5` (TRDP): an `xmlFree` double free, libxml2 rather than wmem.
- `e9343d2da7`: GHashTable on the plain heap.
- `290b43101d` (http3): gated by `HAVE_NGHTTP3`, and freed through gc.
- `29f2177222`, `02085c80ab`: siblings of the S3 rows already vetted and not built.
- `b673bc022a` (kafka): the snappy half frees into BLOCK_FAST, which no arm sees. The lz4 half is an
  S6 with no allocator event.
- `8fd60c6448`, `c8e5887073` (ECMP): a spatial length bug.

**UNRESOLVED, blobs missing:** `36435367d7`, `113a4c3036`, `146721324b`, `1ce6277808`,
`0e67d8b5ee`, `ccb1ac3c8c`, `0b5c71a589`, `1a5e189320`, `2dd000a0ef`, `1c66174ec7`, `9f33105c74`,
`1c5e4b8cf4`, `06e0b0bb09`. The present-side sibling of each was read.

**wmem-repros 22, from `c702b44a01`.** In the pin's `parse_report_descriptor` (`packet-usb-hid.c:3734`),
`wmem_file_scope()` is BLOCK.
- Without the fix line, every OUTPUT item appends the same `usages` array to `fields_out`.
- The `err:` path (`:3949-3957`) then frees `fields_out[j]->usages` once per entry, a double free of
  one BLOCK chunk.
- In a native prototype the buggy sequence made the next three equal-sized allocations return the
  same storage, the free list now holding the chunk twice. The fixed sequence got three distinct
  chunks. It is deterministic and two-sided.
- Its measurements and pre-registration are in the case's own `case.json` and in
  `docs/history/10-10-2026_23-30-00_whole-corpus-audit-cross-program.md`.

## Shape coverage after the hunt

| shape | FFmpeg | tshark | memcached |
|---|---|---|---|
| S1 | none found | none found | blocked: page mover, proxy |
| S2 | none found (AVRefStructPool double unref: synthetic only) | **case 22** | blocked: `5267f14` (Lua) |
| S3 | none found | vetted, not built (ASan-visible) | blocked: proxy and extstore caches |
| S4 | structurally possible in AVRefStructPool; synthetic fixture 16 only | in case 22, as a consequence of S2 | blocked: page mover |
| S5 | none found | `ec0ffc043f`, scope UNRESOLVED | blocked: page mover |
| S6 | none that separates arms | kafka lz4 half: no allocator event | — |

A "none found" is bounded by the recall checks and the filter leak stated under Method.
