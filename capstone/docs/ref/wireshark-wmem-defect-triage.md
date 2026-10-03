# Wireshark: triage for wmem-scope lifetime defects live in the 4.6.8 pin

The `wmem-repros` corpus had no triage inventory under `docs/ref/`; its `corpus.json` says so in a
`note` and the generated `INDEX.md` lists it as a gap. This file is that inventory for one specific
question, and it states its instrument so a wrong reading can be traced to it.

**The question.** All 13 cases in `bug-corpora/wireshark/wmem-repros` are `live_in_pin: false`: the
defect was fixed upstream before 4.6.8 and each case re-creates it by reversing the fix. Is there a
defect of the same class that is **still present in the tree we compile**?

**Answer: none.** Of 82 commits that landed on `origin/release-4.6` after `v4.6.8`, exactly one is a
lifetime defect at a wmem scope -- and reading its release path showed the storage goes back to the
system allocator, so it is a plain-heap defect and not of this corpus's class. The correction is
recorded below rather than quietly dropped.

## The instrument

| | |
|---|---|
| clone | `gitlab.com/wireshark/wireshark`, full history, 100,298 commits |
| pin | tag **`v4.6.8`** — *not* `wireshark-4.6.8`, which does not exist; the 4.6 series is tagged `v4.6.x` while older series use `wireshark-x.y.z` |
| population | `v4.6.8..origin/release-4.6` = **82 commits**. The release branch, not `master`: master carries the same fixes under different hashes plus work that never reached 4.6 |
| filter 1 | the commit message reads as a lifetime defect — use-after-free, freed, stale, dangling — and not as an overflow, a leak or a denial of service |
| filter 2 | the **diff** touches a wmem scope or removes a file-static pointer. Read from the diff, never inferred from the subsystem |

**Controls, because a liveness check that errors reads exactly like a finding.** The first version of
this check reported "live" for every commit: the tag name was wrong, `git merge-base` failed on an
unresolvable ref, and the shell's `||` branch printed the verdict. The checker now refuses to report
until three controls fire — a commit known to be in the pin must read IN-PIN, a nonexistent sha must
read UNRESOLVED rather than either verdict, and the newest commit must *not* read IN-PIN.

**What this instrument cannot see**: a defect never fixed upstream (nothing to find); one fixed only
on `master` and never backported to 4.6; one whose message does not name the lifetime; and one
introduced after 4.6.8. It is a search along named axes, not a proof of absence.

## The one candidate, and why it turned out to be heap class

### ZigBee ZCL Touchlink: file-scope entries kept in a global map across a redissect

| | |
|---|---|
| upstream fix | `609134fa7c55` (4.6 branch), 2026-08-13 — *"ZigBee ZCL Touchlink: empty the commissioning map on a redissect."* Cherry-picked from `030bf6ad011c` on master |
| first release carrying the fix | `v4.6.9` — **not in `v4.6.8`**, confirmed by ancestry, so the defect is live in our pin |
| advisory | `wnpa-sec-2026-92` |
| CVE | **CVE-2026-95391** |
| tracker | listed in the *"Security issues fixed in 4.6.9 and 4.4.19"* tracker issue |
| consumer | `epan/dissectors/packet-zbee-zcl-general.c` |
| allocator layer | wmem **file scope** |

**Liveness proved by reading the pinned tree**, not by the fix date. In `v4.6.8`:

- `:15899` — `static GHashTable * zcl_touchlink_commissioning_map = NULL;` — a global that outlives
  any scope;
- `:16591` — `commissioning_data = wmem_new0(wmem_file_scope(), struct zcl_touchlink_commissioning_data);`
- `:16593` — `g_hash_table_insert(zcl_touchlink_commissioning_map, &commissioning_data->transaction_id, commissioning_data);`
  — the global keeps both the value **and a key that points inside the same object**;
- `register_init_routine` does not appear in that file at the pin. The fix adds one.

So on a redissect the file scope is left, every entry is freed, and the global map still holds
pointers to all of them — plus keys pointing into freed storage. Upstream's own words: *"All of the
entries in the commissioning map are allocated with `file` scope, which means that, before a
redissect pass, they will all be freed … so that we don't have a bunch of references to freed
memory and thus don't have a use-after-free issue."*

### CORRECTION, same day: this defect is HEAP class, not wmem class

The first version of this file said the defect was in class for `wmem-repros` and would be its first
live-in-pin case and first CVE. **That was wrong, and it is withdrawn.** The storage does not stay
with wmem: it goes back to the system allocator, so this is an ordinary heap use-after-free that
ASan and a libc quarantine both see. It belongs in the plain-heap control half, not in the nested
corpus.

The chain, read at the pin rather than assumed:

1. A redissect runs the registered init routines. `epan/packet.c:271` at `v4.6.8` describes them as
   *"called before we make a pass through a capture file … or run a 'filter packets' or 'colorize
   packets' pass"* — that pass is what the fix hooks.
2. `epan/wmem_scopes.c:62` — `wmem_leave_file_scope()` is `wmem_leave_scope(file_scope)` followed by
   a collection, under the comment *"this seems like a good time to do garbage collection"*.
3. `wsutil/wmem/wmem_allocator_block.c:1032` — `wmem_block_gc` walks the block list and, for a block
   that is entirely unused, *"return it to the OS"*, calling `wmem_free(NULL, cur)`.

After a scope leave the file scope holds nothing, so its blocks are wholly unused and all of them
are returned. Upstream's own commit message is the confirming evidence: the crash is *"a 'that
address is not valid' trap"*, because *"memory allocators are free to release regions of the address
space that no longer contain any allocated data"* — that is an unmapped page, which only happens
once the storage has left the allocator.

**This is the rule that already disqualified two candidates**, recorded in
[`port-candidate-survey.md`](port-candidate-survey.md): issues 19399 and 19265 were excluded because
their freeing frame is `wmem_block_gc`, *"the return-to-OS call"*. A scope reset that **retains** the
blocks is in class; a path that returns them is not. `wmem_free_all` on the packet scope retains —
which is why cases 0–11 are in class — and `wmem_leave_file_scope` does not.

**What it is instead.** A real, live, CVE-carrying plain-heap use-after-free: a global container
holding pointers, and keys pointing into the same objects, across a release that reaches the system
allocator. That makes it a **control** case — valuable, because it is a genuine upstream defect
rather than a synthetic fixture, but it supports no claim that this project's mechanism is uniquely
able to catch it.

**What this costs the corpus:** `wmem-repros` still has **13 cases, 0 live in the pin and 0
advisories**, and no live-in-pin candidate for its class has been found.

## WIDENED 2026-10-03: the master-only population this instrument could not see

The section above names the gap itself — *"a defect … fixed only on `master` and never backported to
4.6"*. That gap was the whole answer: searching the release branch saw **82** commits where `master`
carries **4321**. The widened search, and what it found:

| | |
|---|---|
| population | `v4.6.8..origin/master` = **4321** commits, against 82 on the release branch |
| lifetime wording | **24** (one of them the ZigBee commit already triaged) |
| **LIVE at the pin, two-sided** | **5** |
| NOT-LIVE | 18 — of which **7 were backported** under a different sha, 5 have code that postdates the pin (3 of those are files absent from v4.6.8), 3 are funnel commits against a list that does not exist at the pin, 1 is a different implementation, and 2 are the traps below |

**The decisive fact, and it is a methodology one: `v4.6.8` is dated 2026-08-12, newer than 20 of the
24 commits.** So "absent by ancestry" says almost nothing here, and 7 of 23 were in fact present by
content. Liveness was therefore decided by the test that replaced the retracted method (see
[`ffmpeg-live-defect-triage.md`](ffmpeg-live-defect-triage.md)): a cherry-pick probe over the tag,
**plus** reading the enclosing code to confirm the free precedes the read — with controls proving the
probe fires on a known backport, errors on a missing path, and does not read an empty result as
"not live".

### The two one-sided-grep traps, which are the real finding here

Both would have been scored **LIVE** by grepping the pinned tree for the pre-fix line, and both are
NOT-LIVE because *the free that turns the stale read into a use-after-free postdates the pin*:

- **`ac4aa6510e` PPI.** The pre-fix order IS at `v4.6.8:epan/dissectors/packet-ppi.c:417-418`. But
  `v4.6.8:epan/proto.c:1324` is `/*g_free(ptvc);*/` — it frees nothing. The real free arrived in
  `5a8efd7836`, proven **not** an ancestor of `v4.6.8`.
- **`e5dc76c30c` zlib.** Pre-fix `g_free(strmbuf)` then reuse; at the pin the code accumulates into a
  separate buffer and `strmbuf` is never read after any free.

This is the same shape as the `ops_dispatch` retraction on the same day, in a second program. A
pre-fix line matching is **evidence of a line, not of a defect**.

### The 5 live candidates, and which were taken

| sha | subsystem | class | taken? |
|---|---|---|---|
| **`6e61bca421`** http2 | dissector | **heap** — a `static GRegex *` unref'd to a zero refcount but left non-NULL, so its own `== NULL` guard passes and `g_regex_match` reads freed storage | **YES — fixture 15** |
| `030bf6ad01` ZigBee | dissector | heap (see the correction above) | **YES — fixture 14** |
| `2a6db056e1` ngap, `81e76cd4a8` e2ap | dissector | **wmem-nested** — a global `proto_tree *top_tree` moved into `pinfo->pool` private data | **no: LATENT.** Hardening, not observed defects. The exported-handle sets were intersected against the `set_message_label` callers and the intersection is **empty**, so no stale window was reachable; both authors say so in their own messages (*"I haven't found any use-after-free in this dissector"*). Recorded as a pattern, not counted as a bug. One route left open and UNRESOLVED: whether `tshark -d`/Decode-As can bind a `*.proc.*` entry directly and skip the top-level dissector |
| `4a1ae0b63c` dfilter | epan-core | heap | no: latent for tshark — `dfilter_init`/`dfilter_cleanup` run once each per process, so a single invocation never does cleanup→init |
| `d24613c461` opcua | dissector | **neither** | no: **out of class.** Despite a subject saying "heap-use-after-free", the diff adds only length guards and touches no allocator call; the defect is an out-of-bounds read inside a `wmem_alloc`'d buffer. A bounds bug, and the scope here is allocator lifetime |

So the widened search yields **one** new usable case, `6e61bca421`, which became fixture 15. That is
the honest yield: 4321 commits → 24 worded → 5 live → 1 in class, reachable and not latent.

## What was rejected, and why

| candidate | reason |
|---|---|
| the other 29 rows of the 4.6.9 security tracker | spatial or availability: integer overflow into a heap overflow (SCTP, SPDY, LBMC, LoRaWAN, RF4CE, CSN.1, MBIM, SMB, AVCTP), out-of-bounds read (DFVM, X11, FP Hint), infinite loop or memory exhaustion (TIFF, netmon, SYNCHROPHASOR, DICOM, pcapng). Denial of service by memory consumption is out of class by the same rule that rejected two httpd bucket candidates |
| `#21457`, `#21458` CMS — *"reuses freed capability-tree pointer"*, *"leaves freed top-tree pointer"* | **in class but not live.** The release-4.6 cherry-pick `e6320512dc` is an ancestor of `v4.6.8`, so our tree already has the fix. Their issues closed six days *after* the 4.6.8 release date, which is exactly why liveness is decided by ancestry and not by dates. They remain good *historical* candidates, and are siblings of existing case 01 |
| `#21495` Procmon parser state not freed | a leak, not a lifetime defect |
| `#21541` Toshiba stale **stack** memory | the storage is not wmem's, so no allocator event ends its lifetime |

## Historical candidates, if more cases are wanted

The corpus already covers 13 historical defects in four shapes, so a 14th adds a row and little
information unless it opens a new shape. Named here so they are not re-triaged: the CMS pair above;
`#21393` Livewire (`4ba90bea89`); and the recurring upstream shape *"move the global into private
data"* (`81e76cd4a889` E2AP, `8574da1dc1d7` RTSE, `0defc09e5fd6` HTTP), which is shape 1 again.
