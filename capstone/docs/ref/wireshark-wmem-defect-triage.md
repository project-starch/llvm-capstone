# Wireshark: triage for wmem-scope lifetime defects live in the 4.6.8 pin

The `wmem-repros` corpus had no triage inventory under `docs/ref/`; its `corpus.json` says so in a
`note` and the generated `INDEX.md` lists it as a gap. This file is that inventory for one specific
question, and it states its instrument so a wrong reading can be traced to it.

**The question.** All 13 cases in `bug-corpora/wireshark/wmem-repros` are `live_in_pin: false`: the
defect was fixed upstream before 4.6.8 and each case re-creates it by reversing the fix. Is there a
defect of the same class that is **still present in the tree we compile**?

**Answer: one.** Of 82 commits that landed on `origin/release-4.6` after `v4.6.8`, exactly one is a
lifetime defect at a wmem scope.

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

## The one candidate

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

**Why it is in class.** The lifetime ends at a wmem scope reset. The backing block stays with wmem
and nothing reaches `free()`, so a free-keyed mechanism — ASan, a libc quarantine and sweep — has no
event to act on. That is the corpus's whole premise, and this case is the first instance of it that
is *live in the version we build*.

**Its sibling, and why it is not a duplicate.** Shape 1 of the corpus — "stale pointer held by a
global across packets" — holds cases 0, 1 and 6, whose lifetime ends at the *packet* scope between
packets. Here the lifetime ends at the *file* scope on a redissect, a coarser event that frees every
entry at once, and the stale key is an extra. If it is built, `distinguishing` must say that.

**What it would be, if built.** The corpus today has **13 cases, 0 live in the pin and 0
advisories**. This case would be its first live-in-pin case and its first with a CVE.

**Not yet established:** whether the redissect path is reachable in the reduced tshark build, and
whether the reduction is better keyed on `wmem_leave_file_scope` directly, as cases 7–11 key on the
packet scope. The corpus runs `-r <file>` once per process, so a redissect may have to be modelled
rather than driven — which is the same reduction the existing cases already make and declare.

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
