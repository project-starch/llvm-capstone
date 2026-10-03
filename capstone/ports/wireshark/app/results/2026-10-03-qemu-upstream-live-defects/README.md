# Two upstream Wireshark defects live in the v4.6.8 SOURCE, on four heap arms (2026-10-03)

**Question.** The widened triage in
[`docs/ref/wireshark-wmem-defect-triage.md`](../../../../../docs/ref/wireshark-wmem-defect-triage.md)
found two lifetime defects still present in the **source** of the release this port pins. Does the
revoking heap catch reductions of them, and do the unprotected arms let them through?

**Pre-registration.** Both fixtures and all eight predictions were pushed in `319c19730377` **before**
either fixture was built: the push is timestamped 15:08:06 and the earliest fx14/fx15 image anywhere
on disk is 15:08:42, 36 s later. `git diff 319c19730377 -- host/safety-expect.txt` is empty and that
commit is still the most recent one to touch the file.

## Scope — read this before the table

**Neither defect's code is in the binary this port builds.** Checked on the port's *real* tshark image
(`tshark_m1.dom`), not the fixture image, with controls that fire:

- **fixture 15's defect is compiled out.** The whole block sits inside `#ifdef HAVE_NGHTTP2`
  (`packet-http2.c:1693`), and the port's own configs undefine it —
  `/tmp/capstone/tshark-app/cap-cfg/config.h:118` and `xbuild/config.h:119` both read
  `/* #undef HAVE_NGHTTP2 */`. In the image: **0** symbols for `populate_http_header_tracking` or
  `dissect_http2`, and **0** occurrences of the `imsi-` regex string (control: the literal `HTTP`
  appears 42 times; the only http2-ish symbols, `dissect_http2_settings_ext` and
  `http2_get_stream_id`, come from `packet-http.c`'s own HTTP2-Settings handling).
- **fixture 14's dissector is not linked.** `packet-zbee-zcl-general.c` is in the port's source set but
  **0** `zbee` symbols are in the image, against **353** `dissect_*`/`proto_register_*` symbols
  overall. Not a stripping artifact: `g_regex_new` is linked.
- **fixture 15's defect is additionally preference-gated upstream**:
  `v4.6.8:epan/dissectors/packet-http2.c:88` is `static bool http2_3gpp_session = false;` and the
  defect block is inside `if (http2_3gpp_session)` at `:2143`.

So these are **source-level upstream defects reduced to synthetic fixtures**, not defects reachable in
this port's own build. The reductions never execute the upstream consumer in any case — that is what a
reduction is — but the distinction matters and an earlier version of this file got it wrong, claiming
they were "live in the version we compile". **The triage doc's "reachable and not latent" for
`6e61bca421` is withdrawn with it.**

## Verdict

**Eight of eight cells exactly as pre-registered.**

| arm | what the arm does | fixture 14 — ZigBee Touchlink | fixture 15 — http2 GRegex |
|---|---|---|---|
| `level0` | arena-wide bounds, `g_free` only marks | RETURN `e00163` | RETURN `f0015b` |
| `shrink` | per-object bounds, no revocation | RETURN `e00163` | RETURN `f0015b` |
| `sublet` | bounds **+ revoke on free** | **FAULT**, `val==target` | **FAULT**, `val==target` |
| `chunks` | `sublet` + wmem's chunk port | **FAULT**, `val==target` | **FAULT**, `val==target` |

**The registered hypothesis held.** The expect file predicted that 14 and 15 *must agree on every arm*,
because they differ only in the lifetime-ender and in the alias used, and revocation is per object
rather than per alias or per ender. There is no disagreement. `chunks` matching `sublet` was predicted
for the same reason.

## The two defects, at the pin

- **14 — ZigBee ZCL Touchlink, CVE-2026-95391 / `wnpa-sec-2026-92`** (upstream `030bf6ad011c`, fix
  `609134fa7c55`; `git tag --contains` the fix gives `v4.6.9` and `v4.6.10rc0` only). A file-scope
  global `GHashTable` keeps commissioning records across a redissect that frees them, **and its keys
  point inside those same records**. Verified at the pin: the global at `:15899`, the
  `wmem_new0(wmem_file_scope(), …)` at `:16591`, the insert at `:16593` with
  `&commissioning_data->transaction_id` as the key — so the key really is interior to the value — and
  the single `g_hash_table_new` at `:16917`, with **zero** `register_init_routine` and **zero**
  `g_hash_table_remove_all` anywhere in the file. The fix's *content* is absent, not merely its sha.
- **15 — http2 3GPP header decoding** (upstream `6e61bca421`, in `v4.7.0`+ and `master` only). Two
  file-static `GRegex *` are created lazily behind an `if (regex == NULL)` guard and released with
  `g_regex_unref`, which frees at refcount zero **without nulling the static**; the guard then passes
  on freed storage and `g_regex_match` reads through it. Verified at `:2148-2149`, `:2160`, `:2171`,
  `:2185-2186`, repeating at `:2196`, `:2213`, `:2262-2263`. `g_regex_new` returns refcount 1 and
  nothing else references it, so the unref drops the last reference.

**Both fixes are absent by content, and the probe that says so is a tested negative.** The cherry-pick
probe over `v4.6.8` returns 0 for both shas, and the same probe over the tag finds **685** cherry-pick
messages, three of which were re-probed by their source sha and each returned its backport. So the
negatives are negatives, not a silent miss — which matters because `v4.6.8` is dated 2026-08-12, newer
than 20 of the 24 candidates triaged, and 7 of 23 proved to be backported.

**Class: heap.** For 14 that is a correction already recorded (`06a775b2e1b6`), and the chain was
re-verified in the pinned tree: `wmem_leave_file_scope()` → `wmem_gc` (`wmem_scopes.c:62-71`) →
`allocator->gc` (`wmem_core.c:114-117`) → `wmem_block_free_all` re-initialises every non-jumbo block
(`wmem_allocator_block.c:1004-1028`) → `wmem_block_gc` then `wmem_free(NULL, cur)`s every block that
is unused and last (`:1031-1071`), i.e. all of them. So the storage really does return to GLib's heap.

## Why these results are evidence rather than crashes

**The reuse was measured.** Each fixture printed, before touching anything:

    fx14  rec cursor=a5002600  key-interior cursor=a5002608  new-owner cursor=a5002600  same-address=1
    fx15  regex cursor=a5002600  new-owner cursor=a5002600  same-address=1

`same-address=1` means the released storage really came back to a new owner, so on the unprotected
arms the read returns the **new owner's** byte: `0x63` for 14, which is `0x5b + 8` because `fill()`
writes `v0 + i` and 14 reads through an interior key at offset 8, and `0x5b` for 15 at offset 0. Both
values were derived in the prediction, not copied from another row.

**Two of the printed flags are NOT evidence, and were previously listed as if they were.**
`map-still-holds=1` is `tl_map_value != NULL` and `guard-passes=1` is `h2_regex != NULL`; the fixtures
never null either static, so **neither can read 0**. They describe the modelled defect; they measure
nothing. Only `same-address=1` is a measurement.

**The faults are attributed, by a load base established independently of the symbol:**

1. each fixture printed `target=` *before* the touch, and QEMU's diagnostic names the same value —
   `a5002608` for 14, `a5002600` for 15, on both arms;
2. **the load base is `0xa01f0000`, fixed by the entry point, not by the symbol.** Both images are
   `ET_EXEC` with first LOAD `p_vaddr = 0x10000` and entry `0x10000`; the first QEMU trace line of
   every boot is `[CAPSTONE] DELIN gp (#1): pc = a0200000`, and `0x10000 + 0xa01f0000 = 0xa0200000`.
   Under that base, `pc - base` is `0x25044` (sublet) and `0x273ec` (chunks) — `tsapp_fix_touch+0x14`
   in each, and the function spans `[0x25030, 0x2505c)` so the offset is strictly inside it. Both
   disassemble to `cincoffset a0, a0, a1`;
3. **the alternative base is refuted:** under `0xa0200000` the same pcs land on `shrink s1, a0, a1`
   and `ldc s6, 0x30(sp)` — neither a `cincoffset`, neither with `rd=a0 rs1=a0`;
4. **the pc is exact, not translation-block-start:** `qemu-12`'s `target/riscv/op_helper.c` calls
   `cpu_restore_state` immediately before the diagnostic, and a TB-start pc would have read
   `a0215030` (the function entry, a call target) rather than `a0215044`;
5. the diagnostic says `rd=x10 rs1=x10`, and `a0` **is** `x10`;
6. **neither section contains a `mark=` or `returned` line at all**, which is the rule
   `safety-verdict.py`'s `classify()` enforces (imported by `check-safety.py`), stricter than
   "nothing after the touch line";
7. the codegen shows the mechanism: for 14, `0x253e8: ldc a0, 0x10(s8)` reloads the static **after**
   the `g_free`, then calls `tsapp_fix_touch`; for 15, `0x253f8: ldc a0, 0x10(s7)` likewise.

**A previous version of this file argued the attribution from a symbol-delta coincidence** — that the
`0x23a8` delta between the two images' `tsapp_fix_touch` equals the pc delta. **That argument is
vacuous:** for any common base `B`, `(pc₂−B)−(pc₁−B) ≡ pc₂−pc₁`, so the equality is an algebraic
identity that holds for any pair of symbols `0x23a8` apart. It has been dropped. The same version
stated the base as `0xa01f0014`, which is `pc − symbol` and self-inconsistent with the `+0x14` offset
it also claimed.

## Controls: what ran where, precisely

**Eight boots, two of them aborted.** The first attempt put fixtures 1, 5, 14 and 15 in one boot on
each fault arm; fixture 5 faulted as predicted and 14 and 15 never ran — both serial logs end at
`=== FIXTURE 5 BEGIN` with no EXIT. The file's own header says a faulting fixture must be last; the
first attempt did not honour it. The four decisive FAULT boots were then re-run with one faulting
fixture each.

- **Fixture 1 ran first in all eight boots** and returned `100001` every time.
- **Fixture 5 — the same allocate/free/reuse shape without the global or the refcount — ran on
  `level0`, `shrink` and the two aborted fault boots.** It returned `50015b` on the unprotected arms,
  identical to its 2026-09-25 recorded value, and faulted on the aborted `sublet` and `chunks` boots,
  which is its own registered prediction. **It is NOT an in-boot control for the four decisive FAULT
  boots**, where the only in-boot control is fixture 1 — which never frees. An earlier version of this
  file said "both controls fired, in every boot"; that was false.

## The mechanism claim is PLAUSIBLE-BUT-UNPROVEN

The arms' behaviour is consistent with revocation on free, and that is the documented mechanism, but
**this bundle does not measure a revoke**:

- `err-14.txt` and `err-15.txt` are **0 bytes** on all four fault arms — the heap report never printed,
  so there is no `free=`/`revoke=` counter reading for the faulting runs. (`err-1.txt` on `sublet`
  reads `free=0 … revoke=0`, but that is fixture 1.)
- **The surviving alternative** is that a sublet-heap capability could be untagged by a `.bss`
  round-trip independently of the free. It is only partly closed: `stdout` is loaded as a capability
  from a global with `ldc` and the subsequent `printf` chain works on every arm, so that path
  preserves tags in-boot — but `stdout` is a *runtime* capability, not a sublet-heap one. **No in-boot
  arm stores a sublet-heap pointer in a file-scope static, leaves it unfreed, reloads it and touches
  it.** That is the missing control, and it is cheap: such a fixture must RETURN. Alternatively print
  `tsapp_heap_report()` before the touch so `free=1 revoke>=1` is on the record.
- **`val==target` discriminates less than it appears.** `a5002600` is the first 64-byte allocation on
  the sublet pool and is the same address for fixture 1's object, fixture 5's target and fixture 15's
  target — the aborted `sublet` boot's fault reads `val=0xa5002600` too, and that one is fixture 5's.
  Inside a one-faulting-fixture boot there is no ambiguity, so the cells stand; but the check excludes
  a null dereference and little else.

## Oracle coverage

- **The four RETURN cells carry the oracle's corroboration**: `{"version":1,"kind":"exit","value":99}`
  for 14 and `value:91` for 15, and `99 = 0x63 = e00163 & 255`, `91 = 0x5b = f0015b & 255`.
- **The four FAULT cells cannot be judged by it on this platform.** `check-safety.py:40` is
  `elif result.get('kind') == 'signal' and result.get('value') == 11 and result.get('fault'):`, and
  this build of `capstone-job` contains **0** occurrences of `fault` (control: `version` appears 3
  times); its format string is exactly `{"version":1,"kind":"%s","value":%d}`. The signal/11 branch is
  structurally unreachable here.
- **No fault record exists, and that is structural rather than surprising:** `capstone-job` contains
  **0** occurrences of `CAPSTONE` at all, so it never reads `CAPSTONE_FAULT_RECORD`. An earlier version
  of this file called the absence surprising "despite `CAPSTONE_FAULT_RECORD` being set".

So the four faults were attributed by hand. **A hand check is not the gate, and this bundle does not
contain a run of the gate.** The driver also reports `rc=1` on every FAULT boot: the fault wedges the
guest shell so the prompt pattern times out. That is the fault's consequence, not a fixture failure.

## What this does and does not say

- **These are CONTROL results, not a new detection claim.** Both defects are plain `g_malloc`/`g_free`
  use-after-free, which ASan reports. **No native/ASan arm was run for these two** — the comparator
  claim is inherited from the class, not measured here.
- **Fidelity, stated as a limitation.** Fixture 15 models `g_regex_unref` with a hand-rolled refcount
  that has one holder; the decrement, the zero test and the guard are all real machine code (not folded
  away), so it exercises the shape — free at zero, static left non-NULL, guard passes — and not GLib's
  internals. Fixture 14 frees a **64-byte object** where upstream frees a **≥1 MiB block**, and its
  `same-address=1` reuse is guaranteed by construction, whereas a 1 MiB block returning to the same
  address is plausible but unmeasured.
- **The nested-allocator point, stated correctly.** Defect 14's object *is*
  `wmem_new0(wmem_file_scope(), …)`, so "neither defect is at a wmem scope" — an earlier wording here —
  is wrong. The accurate statement: **the lifetime-ending free bottoms out in `g_free` of the
  containing block, not in a wmem reset**, which is why the class is heap and why `chunks` has nothing
  wmem-specific to see. That last part is an **inference**: the wmem counters were not read in the
  faulting runs, and fixture 1's `err-1.txt` on `chunks` showing `wmem opens=0 … revokes=0` is a
  different program.
- **QEMU only.** The temporal faults are the emulator untagging a revoked capability on reload (Q-11);
  the deployed silicon lets such an access retire.
- **N = 1 per cell.**

## Platform

The same assembled platform as
[`ports/ffmpeg/app/results/2026-10-03-qemu-upstream-defects-heap-arms/`](../../../../ffmpeg/app/results/2026-10-03-qemu-upstream-defects-heap-arms/README.md):
application-SDK images need the delegated runtime's process ABI (SBI `0x21`–`0x2b`), so the monitor is
`deleg-gate2/opensbi-T`'s `fw_jump.elf` and QEMU is `qemu-12`, a checkout at exactly this branch's pin.
Each boot's own serial log names its `-virtfs` directory, so the `.dom` disassembled above is the file
that was served; `cma=1536M`; all six `capstone-job` copies are byte-identical.

**The cached cross-build relinked against the intcap SDK without triggering a recompile** — 15 s per
arm, rc=0 on all four — so the feared ~90-minute rebuild was not needed. The relinked images carry
`.capstone_application`; the negative half of that check could not be re-run because no 2026-09-25-era
image survives on disk, and it is not load-bearing.

Files: `result-lines.txt` (every line above).
