# Wireshark: triage for SPATIAL defects whose overflow stays inside a wmem block

The temporal hunt could not have found these. `wireshark-wmem-defect-triage.md:23` makes filter 1
*"a lifetime defect … **and not as an overflow**"*, so every spatial defect in this program was
discarded before triage, and 29 further rows of the 4.6.9 security tracker were rejected en bloc as
*"spatial or availability"* (`:178`). This file is the mirror instrument. See the retraction in
`spatial-vs-temporal-three-programs.md:20` for why the earlier "0 spatial" was a statement about the
search, not about Wireshark.

**Verdict: 15 class-B defects found across the whole history, 2 of them LIVE at the `v4.6.8` pin,
and 4 of them catchable by the port as it stands today.** Every one is a real upstream instance of
the shape that, until now, existed in this project only as the **synthetic** fixture `tsapp` fx12
`wmem_neighbour`.

Read §§1-4 for the three defects found in the post-pin population (the two live ones among them),
then the WIDENED section for the whole-history result, which is where the four immediately
measurable candidates are.

## The instrument

| | |
|---|---|
| clone | `/tmp/capstone/upstream/wireshark`, 118,543 commits |
| pin | tag **`v4.6.8`** |
| population | `v4.6.8..origin/master` = **4,321** commits (the widened population, as `wireshark-wmem-defect-triage.md:112` uses) |
| filter 1 | spatial wording in the subject — **31 survive**. Integer/refcount overflow, leaks, DoS and stack overflow are excluded explicitly: an integer overflow here is a *temporal* driver, which is the trap memcached case 03 sits in |
| filter 2 | **which bound does the overflow cross?** Read from the allocation site in the fix's parent, never inferred from the subsystem |
| filter 3 | liveness **by reading the pinned source**, not by ancestry — see below |
| script | `bug-corpora/tools/spatial-triage.py --self-test` |

### Filter 2 is the whole point: the class decides the value

| class | the overflow crosses | who faults | value |
|---|---|---|---|
| **A** | the `g_malloc` bound | `shrink`, `sublet`, and CHERI alike | a tie row; no contribution |
| **B** | a **wmem chunk** bound, inside the one block `g_malloc` handed wmem | **only a chunk-bounding port** | the contribution |
| **STACK** | a stack array | no allocator involved | out of class |

### Filter 3: ancestry is WRONG here, and it cost a false verdict

`merge-base --is-ancestor <master fix> v4.6.8` called **21 of these candidates live**. Reading the
pinned source shows the fix is already there, backported **under a different hash** — which
`wireshark-wmem-defect-triage.md:22` warns about in as many words (*"master carries the same fixes
under different hashes"*), and whose own rule is the one to follow: *"Liveness proved by reading the
pinned tree, not by the fix date"* (`:50`).

`e8ef9df09d` is the case that exposed it: not an ancestor of `v4.6.8`, yet `v4.6.8` already carries
`PFT_RS_K_MAX` and the widened allocation. So filter 3 now tests whether the **fix's own added lines**
appear in the pinned file, and the self-test keeps that honest two-sided — it must catch
`e8ef9df09d` as `FIX-IN-PIN` (6/6 added lines present) **and** still be able to say `DEFECT-LIVE`.

**But filter 3 has a measured false-live rate of about 20%, so its `DEFECT-LIVE` is a candidate and
never a result.** memcached measures it, because there the pin provably *is* upstream head and no
candidate can be live: filter 3 still said `DEFECT-LIVE` 4 times out of 20
(`memcached-spatial-defect-triage.md`, §5). The mechanism is that a fix's added lines can be
modified again later, so they are absent verbatim from the pin even though the fix is present.

**So the two liveness claims below do not rest on filter 3.** Each was confirmed by reading the
pinned source directly, and that reading is quoted where it is claimed.

The script's other controls, all of which fire: filter 2 separates a hand-adjudicated B / A / STACK triple;
a nonexistent sha reads `UNRESOLVED` rather than either verdict (`--is-ancestor` on a bad object
exits 128, which reads exactly like "live"); a tag resolves; and filter 1 rejects
`"refcount overflow"`. The self-test **failed twice for real reasons** while being written, which is
the only evidence that it can.

## The three class-B defects

All three overflow a chunk obtained from **`pinfo->pool`**, the per-packet wmem scope. One `g_malloc`
hands that scope a block and every chunk carved from it inherits the block's bounds, so on `level0`,
`shrink` and `sublet` the access is **in bounds and legal**. That is the gap, and it is the same one
`tsapp` fx12 demonstrates synthetically.

### 1. `1d8acb21ab` — SolarEdge, 6 bytes PAST a chunk. **LIVE at the pin.**

Quoted from the pinned tree, `v4.6.8:epan/dissectors/packet-solaredge.c:1005-1029`:

```c
int payload_length = length - SOLAREDGE_ENCRYPTION_KEY_LENGTH;
uint8_t *payload = (uint8_t *) wmem_alloc(scratch, payload_length);
uint8_t *intermediate_decrypted_payload = (uint8_t *) wmem_alloc(scratch, payload_length);
...
for (i  = 0; i < payload_length; i++) {
        out[i] = intermediate_decrypted_payload[i + 6] ^ intermediate_decrypted_payload[2+(i&3)];
}
```

`i` runs to `payload_length - 1`, so `[i + 6]` reads indices up to `payload_length + 5`: **six bytes
past a `payload_length`-byte chunk**. `scratch` is `pinfo->pool` at `:1292`. The two `wmem_alloc`
calls are consecutive in the same scope, so the overread crosses into the neighbouring chunk —
**fixture 12's shape, with a real dissector**.

- **Liveness, two-sided:** the `[i + 6]` read is present at the pin (`:1029`); the fix's marker
  `payload_length -= 6` has **0 occurrences** in the pinned file. The fix adds the guard
  `length < SOLAREDGE_ENCRYPTION_KEY_LENGTH + 6` and then `payload_length -= 6`, exactly
  compensating the `+6`.
- **Note** the pin's version has **no guard on `length` at all**, so `payload_length` can also go
  negative; that is a second, separate defect in the same function and is not what this case claims.

### 2. `d24613c461` — opcua, a read BELOW a chunk. **LIVE at the pin.**

This one was **already triaged and rejected on scope**, not on merit —
`wireshark-wmem-defect-triage.md:147`: *"the defect is an **out-of-bounds read inside a
`wmem_alloc`'d buffer**. A bounds bug, and the scope here is allocator lifetime."* It is in class B
and in scope here.

Quoted from the pinned tree, `v4.6.8:plugins/epan/opcua/opcua.c:324-332`:

```c
static int verify_padding(const uint8_t *padding)      /* :324 -- no bound is passed in */
{
    pad_len = *padding;                                /* read from the packet */
    for (i = 0; i < pad_len; ++i) {
        if (padding[-pad_len + i] != pad_len) return -1;   /* :332 -- NEGATIVE index */
```

`padding` points into `plaintext`, allocated at `v4.6.8:642` with
`wmem_alloc(pinfo->pool, plaintext_len)`, and `pad_len` is **attacker-controlled data read from the
packet**. So `padding[-pad_len + i]` reads up to 255 bytes *below* the chunk, into the wmem block's
earlier chunks or its header. The fix passes an `available` count and rejects `pad_len > available`.

- **Liveness, read from the pin, not from filter 3:** the negative-index read is present at `:332`;
  `verify_padding` still has the single-argument signature at `:324`; and the fix's guard
  `pad_len > available` has **0 occurrences** in the pinned file.
- **Class nuance, because the direction matters.** A read *below* a chunk is class B only while the
  chunk has something below it inside the block. If `plaintext` happens to be the **first** chunk
  carved from a fresh block, `pad_len` up to 255 reads below the block itself, i.e. past the
  `g_malloc` bound — which is class A, and which `shrink` and `sublet` would catch. So this defect
  is **class B in general and degrades to class A for the first chunk in a block**. Which case a
  real capture produces is not established here. `1d8acb21ab` has no such ambiguity: it overreads
  *forward* from the second of two consecutive chunks, so the bound crossed is always a chunk bound.

**Its subject says `heap-use-after-free`, and the defect is spatial.** Trust the diff, not the
wording — the existing triage doc reached the same conclusion from the other direction.

### 3. `e8ef9df09d` — ETSI DCP, a write PAST a chunk. Fix already in the pin.

Not live (`FIX-IN-PIN`, 6/6), so this is a **fix-reversal** case, which is what every memcached and
FFmpeg corpus case already is. At the fix's parent, `epan/dissectors/packet-dcp-etsi.c`:

```c
#define PFT_RS_N_MAX 207
#define PFT_RS_K 255
#define PFT_RS_P (PFT_RS_K - PFT_RS_N_MAX)          /* = 48 */
...
uint8_t *output = (uint8_t*) wmem_alloc(pinfo->pool, decoded_size);   /* :376, = fcount*plen */
...
memcpy(output+index_out+PFT_RS_N_MAX, deinterleaved+index_coded, PFT_RS_P);   /* :264 */
index_out += rsk;                                                            /* :270, rsk < 207 */
```

The in-place Reed-Solomon decode writes 48 parity bytes at `+207` while `index_out` advances by
`rsk`, which can be well under 207, and `output` is only `fcount*plen` bytes. The fix says so itself:
*"it does require that output have extra space at the end for the parity bytes (PFT_RS_P) and any
extra zeros (PFT_RS_K_MAX - rsk)"*, and the pin's allocation is now
`wmem_alloc(pinfo->pool, decoded_size + PFT_RS_N - rsk)`. The constants `N` and `K` had been
swapped relative to RS(255,207).

## The finding that matters for the port, and it is a gap

**All three allocate from `pinfo->pool`, the packet scope, which is a wmem `BLOCK_FAST` allocator —
and the `chunks` port deliberately does not narrow `BLOCK_FAST`.** That is measured, not assumed:
`ports/wireshark/app/results/2026-10-03-qemu-wmem-chunks-arm/README.md` shows fixture 10, a
BLOCK_FAST case, still carrying the whole block (`len=1048528`) on `chunks`, while BLOCK chunks come
back at 64 bytes.

So as the port stands today, **these three real defects would not be caught by any arm we have** —
`level0`/`shrink`/`sublet` see one block, and `chunks` leaves this scope alone. The synthetic fx12
uses a **BLOCK** allocator (file scope), which is why it discriminates.

This is the concrete, actionable consequence of the hunt: **extending the chunk adapter to
`BLOCK_FAST` is what converts three real upstream spatial defects into detections.**

**And the gap is wiring, not design.** Three things, each checked:

1. The adapter itself is written in block-generic terms — `wm_block_open`, `wm_chunk_split`,
   `wm_chunk_issue`, `wm_chunk_retire`, `wm_block_reset`
   (`ports/wireshark/wmem/src/allocators/sublet/chunks.c`). Nothing in it is specific to `BLOCK`.
2. The only thing that wires those hooks into a wmem allocator is a single patch,
   `ports/wireshark/wmem/patches/wireshark-4.6.8-0002-wmem-block-chunks-under-sublet.patch`, and it
   patches **`wmem_block.c`**. There is **no patch against `wmem_block_fast.c`** anywhere in the
   tree.
3. `BLOCK_FAST` is the *simpler* allocator: it carves linearly and has no per-chunk free, only
   `free_all`. So its adapter is close to a strict subset of the BLOCK work — open the block, carve
   a region per allocation, one revoke on reset.

**The corpus driver already models the right allocator**, which removes the other obvious obstacle:
`ports/wireshark/wmem/src/shared/scopes.c:39` creates the packet pool as
`wmem_allocator_new(WMEM_ALLOCATOR_BLOCK_FAST)`, faithfully matching upstream's `pinfo->pool`, and
`bug-corpora/wireshark/wmem-repros/shared/driver.c:92` acquires exactly that pool as `wm_packet`.

**The port's 20-dissector whitelist is NOT an obstacle either.** `ports/wireshark/app/dissector-whitelist.txt`
contains none of `solaredge`, `opcua` or `dcp-etsi` — but it does not contain `rpcrdma`, `sip`,
`mysql` or `xml` either, and the corpus's thirteen existing cases are all defects in dissectors
outside it. Corpus cases do not run inside `tshark`: `ports/wireshark/wmem/cmake/Replay.cmake`
builds one standalone program per `case.c` against the real wmem allocator. So a reduction of
`1d8acb21ab` is buildable today; what it cannot yet do is *discriminate*, for want of the
`BLOCK_FAST` adapter.

## WIDENED to the whole history: 15 class-B defects, and 4 of them the port can catch TODAY

The sections above used `v4.6.8..origin/master` (4,321 commits), matching the temporal instrument's
population. Widening to the **whole history to the pin — 96,806 commits** changes the picture, and
fix-reversal cases are perfectly acceptable (every existing memcached and FFmpeg corpus case is one):

| | |
|---|---:|
| population | **96,806** commits |
| filter 1 kept | **303** |
| class **B** | **15** |
| class A | 21 |
| STACK | 16 |
| unresolved | 251 |

Liveness across the 15: **12 `FIX-IN-PIN`, 2 `UNRESOLVED`, 1 `DEFECT-LIVE`** — and that one live
reading is a **false positive**, caught by hand (see below). So the two live class-B defects remain
the solaredge and opcua pair; the other 13 are fix-reversal material.

### The four in `wmem_file_scope()` are the important ones

File scope is a wmem **`BLOCK`** allocator — the one the `chunks` port *does* narrow. So unlike the
three post-pin defects, **these four need no new port work to discriminate**:

| candidate | file | scope |
|---|---|---|
| `0261fd7da6` | `epan/dissectors/packet-http.c` | `wmem_file_scope()` — **and `packet-http.c` is in the port's whitelist** |
| `d7d1686a95` | `epan/dissectors/packet-snmp.c` | `wmem_file_scope()` |
| `1c090e9292` | `epan/dissectors/packet-lbmc.c` | `wmem_file_scope()` |
| `4a4871a831` | `epan/dissectors/packet-ntlmssp.c` | `wmem_file_scope()` |

**`0261fd7da6` IS NOW BUILT AND MEASURED** as case 13 of `bug-corpora/wireshark/wmem-repros`
(results: `results/20261005-qemu-spatial-case13-{ON,OFF}/`). Its mechanism is as reducible as a
synthetic fixture.
Quoted from the fix's parent, `epan/dissectors/packet-http.c:3740-3751`:

```c
first_range_num_str = wmem_strdup(wmem_file_scope(), value);
if (first_range_num_str) {
        first_range_num_str += 6;  /* Move the pointer past "bytes=" */
        first_range_num_str = strtok(first_range_num_str, "-");
        first_range_num = strtoul(first_range_num_str, NULL ,10);
}
...
        char *str = wmem_strdup(wmem_file_scope(), value);
        str += 8;
        first_range_num = strtoul(str, NULL ,10);
```

**Unconditional pointer arithmetic with no length check at all.** A `Range:` header value shorter
than 6 (or 8) characters moves the cursor past the end of the `wmem_strdup`'d chunk, and `strtok` /
`strtoul` then read on into the rest of the file-scope BLOCK. The upstream subject is *"Fix buffer
overflow, use after free in HTTP Range"*, so it is both spatial and temporal — only the spatial half
is claimed here. `FIX-IN-PIN`, so a fix-reversal case.

Three other class-B candidates are in code the port always builds: `ed20250c13` in `epan/proto.c`
and `69dac89280` in `packet-tcp.c` (both `wmem_packet_scope()`, so BLOCK_FAST), plus `0939cf989d`,
which is the **same ETSI DCP defect** as `e8ef9df09d` under a different hash.

### The one `DEFECT-LIVE` reading in the widened run is FALSE, and that is the fifth measured instance

`5a560f3f6a` — *"dns: fix off-by-one buffer overflow (write)"*, 2018 — is a genuine class-B
off-by-one: `g_snprintf(np, maxname + 1, ...)` against a buffer allocated
`wmem_alloc(wmem_packet_scope(), maxname)`, i.e. a one-byte **write** past the chunk. Filter 3
called it live. **Reading the pin refutes that:** `v4.6.8:epan/dissectors/packet-dns.c:1677,1689,1703`
all read `snprintf(np, maxname, ...)` — the fixed form. The fix is present; it is simply unfindable
verbatim because upstream renamed `g_snprintf` to `snprintf`, so none of the fix's added lines
match. Exactly the mechanism §filter-3 predicts, now observed in the Wireshark population too.

## The case that was built from this, and what its run REFUTED

`0261fd7da6` is case **13** of `bug-corpora/wireshark/wmem-repros` — the first upstream **spatial**
defect reduced anywhere in this tree, and the corpus's only row that faults on **bounds (cause 5)**
rather than on revoked authority (cause 24).

**Measured, 4/4 arms on each of two builds, runner exit 0, with the suite's negative control run and
passing 4/4** (so the result is not vacuous):

| build | case 13 `spatial` | case 13 protected arm | case 11 control (temporal) |
|---|---|---|---|
| `WM_CHUNKS=ON` | **FAULT cause 5** `0x10190222c` | **FAULT cause 5** `0x10190222c` | completes / cause 24 |
| `WM_CHUNKS=OFF` | **FAULT cause 5** `0x1019013f8` | **FAULT cause 5** `0x1019013f8` | completes / cause 24 |

**And the run refuted this document's own expectation.** The case was filed — and the section above
was written — predicting that `sublet` would complete and only the chunk arm would fault, i.e. that
it discriminated the chunk port. It does not, and **cannot in that harness**:
`ports/wireshark/wmem/src/shared/wmem-port-hooks.h:11-15` narrows **every** wmem allocation to its
request under `WM_DOMAIN`, unconditionally and whatever `WM_CHUNKS` is set to. So the corpus port
has **no malloc-granular arm at all**, the 6-byte object is bounded to 6 bytes on every arm, and the
read at `+8` faults on all of them.

So the corrected claim, in two halves that must not be merged:

- **Established:** the defect is real, reduces cleanly, and per-allocation bounds catch it.
- **NOT established here:** that a nested allocator hides the *extent* from `malloc`. That contrast
  needs an arm whose bounds genuinely are malloc-granular, and the only place with one is the tshark
  **app** port — the measured fx12 ladder (`level0` 41 908 912, `shrink` 8 388 560, `sublet`
  1 048 528, `chunks` 64). The `BLOCK_FAST` gap below is about that port, not this one.

**Two instrument defects were found and fixed before any of this was believed**, both of which would
have reported a correct reading as a failure, and both negative-tested:

1. `run-defects.py` hardcoded `cause in (24, 25)`. A spatial row faults with cause **5**, so the
   gate would have failed it. The expected causes now come from the case's own arm, and every one of
   the 14 measured rows already in `results/` is still accepted.
2. `classify()` assumed the unprotected `spatial` mode always completes. For a spatial defect it
   faults. Both modes are now judged by their declared arm, with cases 0-12 verified unchanged.

## Rejected, with reasons

| candidate | reason |
|---|---|
| `c8c396cf23` OBEX "off by one going up a path level" | **Not a memory-safety defect.** `wmem_strndup(scope, current_path, i_path - current_path - 1)` — the `-1` is in a *length* argument, so the copy is one byte SHORT and the path is truncated. A logic off-by-one that reads and writes in bounds. Filter 1 matched the wording; the mechanism is not spatial. A matching shape is not a matching mechanism |
| `46c813e7ad` DCT2000, `be813ede9d` etwdump, and 12 others | **`FIX-IN-PIN`** — the fix is already in `v4.6.8`, backported under another hash. Usable only as fix-reversals, and none is class B |
| `be813ede9d` etwdump, `f207d25f4b` / `830cf562a0` wiretap | **class A** — `g_malloc`/`g_strdup`, so `shrink` and `sublet` already fault and CHERI does too. A tie row |
| `76a6c36`-shaped stack cases | **STACK** — no allocator is involved, so the nested-allocator argument does not apply |
| the 29 rows of the 4.6.9 security tracker | not re-triaged here. `wireshark-wmem-defect-triage.md:178` names their subsystems; they are the obvious next population |

## What this instrument cannot see

- **A defect never fixed upstream.** The population is fixes; an unfixed overflow leaves no commit.
- **A defect whose fix does not word itself spatially.** Filter 1's cost, the same cost
  `ffmpeg-live-defect-triage.md:191` records for the temporal side: no wording filter catches a
  subject that says only "clear the ER picture".
- **The allocation site when it is more than one alias hop from the fix's hunks.** 24 of 31
  candidates came back class `?` for this reason. A `?` is *unresolved*, **not** class A — reading it
  as "no nested defects here" would be the clean-zero mistake again.
- **Reachability.** That a dissector path exists in the pin does not prove a capture can reach it.
  None of the three carries a reachability proof yet.
- **Whether `tshark` as our port builds it** compiles the dissector at all; `solaredge`, `opcua` and
  `dcp-etsi` still need checking against the port's dissector set.
