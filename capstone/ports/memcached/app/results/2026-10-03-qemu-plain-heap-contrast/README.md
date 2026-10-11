# The plain-malloc contrast: two memcached defects where the free DOES reach the allocator (2026-10-03)

**Question, and it is not "do more defects get caught".** Fixtures 12–16 end their objects **inside**
memcached's own allocators — `cache_free` pushes onto a STAILQ, `do_slabs_free` onto a size class's
slots list — so the runtime's `free` is never called. The expect file therefore registers `sublet` as
the **negative control that must RETURN** on all five, and that silence is the nested-allocator
blindness claim itself.

**But a silence is only attributable if the same arm is shown to speak.** Fixtures 17 and 18 are two
upstream memcached defects whose objects are released by the *platform's* `free`/`realloc`, so the
release does reach the runtime heap. If `sublet` is working, it must fault on these.

**Pre-registration.** Both fixtures and all ten predictions were pushed in `e611f0679fa8` before
either was built. `git diff e611f0679fa8 -- host/safety-expect.txt` is empty, and the control for that
command — the same diff against a file I did edit after committing — reports 54 insertions, so the
check can report a change.

## Verdict

**Ten of ten cells exactly as pre-registered, and `sublet` speaks.**

| arm | what it does | fixture 17 — `realloc` cursor | fixture 18 — write-after-free |
|---|---|---|---|
| `level0` | arena-wide bounds, free only marks | RETURN `1100163` | RETURN `12001ee` |
| `shrink` | per-object bounds, no revocation | RETURN `1100163` | RETURN `12001ee` |
| `sublet` | bounds **+ revoke on free** | **FAULT** cause 24, `val==target` | **FAULT** cause 24, `val==target` |
| `slabsublet0` | sublet + per-unit bounds, renews nothing | **FAULT** cause 24, `val==target` | **FAULT** cause 24, `val==target` |
| `slabsublet1` | sublet + the slab adapter's revokes | **FAULT** cause 24, `val==target` | **FAULT** cause 24, `val==target` |

**Read this against fixtures 12–16 in the same expect file.** There, `sublet` and `slabsublet0` must
RETURN and only `slabsublet1` faults. Here, every revoking arm faults. The difference is not the
protection — it is the same heap, the same images' runtime — it is **where the lifetime ends**. That
makes the 12–16 result a statement about the nested allocator rather than about an arm that never
worked.

## The two defects

- **17 — commit `204019d` (a 2006 contributed patch).** `try_read_network` grows a malloc'd connection
  read buffer with `realloc`, which may move the block and release the old one. The fix survives
  verbatim at the pin, `memcached.c:2467`, and updates **both** pointers:
  `c->rcurr = c->rbuf = new_rbuf;`. Reversed to the base alone, the parser's interior cursor still
  points into the released block, and `proto_text.c:243-244` reads straight through it. Plain
  `malloc`/`realloc` — the path a conn reaches via `rbuf_switch_to_malloc`, **not** cache.c's rbuf
  cache. Same *object* as fixture 12, different allocator seam: 12's ender is `cache_free` onto a
  STAILQ, this one's is `realloc` releasing the block.
- **18 — `e779381` logger.** `logger_thread_close_watcher` clears the global slot and frees the
  watcher (`logger.c:734,738`); the fix present at the pin is the caller's recheck at
  `logger.c:691-693`. Reversed, the caller writes `w->failed_flush` through its own stale pointer.
  The corpus's **only write-after-free** — every other case reads — on a `calloc`'d struct that is
  neither an item nor a slab chunk.

Both are **historical**: the upstream fix is reversed, as in cases 0, 1, 3 and 4. That is not a
shortcut. memcached cannot supply a live-in-pin case, re-verified two-sided on 2026-10-03:
`origin/master` **is** the pin with 0 commits after it, and `origin/next` — the branch memcached
actually develops on, and which the earlier single-branch check had missed — is one commit ahead with
`slabs: minor cleanup`, nothing lifetime-related.

## Why these are evidence rather than crashes

**The triggering condition was created, and each fixture refuses to report a mark otherwise.** 17
exits `0xE0017` if `realloc` grew in place or the released block was not reissued; 18 exits `0xE0018`
if the storage was not reissued. Both printed the positive form instead:

    fx17  realloc-moved=1  reissued-to-next-owner=1
    fx18  same-address=1   slot-cleared=1

So on the unprotected arms the read returns the **new owner's** byte — `0x63` for 17, which is
`0x5b + 8` because `mcapp_fill` writes `v0 + i` and 17 reads through an interior cursor at offset 8;
and for 18, the `0xEE` it poked through the dead pointer, read back through the **live** pointer,
which is what makes it a demonstrated write-after-free rather than a silent one. Both values were
derived in the prediction, not copied.

**The arms differ in the one thing they should.** The capability the fixture holds:

    level0  fx17  bounds=[c02e2820,c42e2820)  len-from-cursor=62132144   the whole arena
    sublet  fx17  bounds=[c8003600,c8003800)  len-from-cursor=512        the object alone
    sublet  fx17  rcurr: same bounds, len-from-cursor=504 — the interior cursor, 8 bytes in

**The faults are attributed, not merely observed:**

1. each fixture printed `target=` before its touch, and every diagnostic names the same value —
   `c8003608` for 17, `c8003600` for 18, on all three revoking arms;
2. the implied load base `(pc − symbol)` is `0xc01f0014` for **every** fault, and under it each pc
   lands on the helper whose access type is correct: fixture 17 **reads** and resolves to
   `mcapp_fix_touch+0x14`, whose `cincoffset` is followed by `lbu a0,0x0(a0)`; fixture 18 **writes**
   and resolves to `mcapp_fix_poke+0x14`, followed by `sb a2,0x0(a0)`. A coincidence would not place
   the write fixture on the store helper;
3. the fx17→fx18 pc delta is `0x2c` in every image, exactly `mcapp_fix_poke − mcapp_fix_touch`
   (`0x578dc − 0x578b0`);
4. **the image hash, not the arm label.** Every sublet fault record carries
   `sha256=1e3df2b1edefeb31`, every slab one `sha256=ed4fe4380886feea`, and each matches the hash the
   build itself reported for that image. The two slab arms share one image and are selected by
   `MC_SLAB_SUBLET_MODE`, so the hash is also what distinguishes them from `sublet`.
5. exit status corroborates both directions: RETURN cells exit `mark & 255` (99 = `0x63`,
   238 = `0xEE`), FAULT cells exit `139` = 128+11 and wrote a fault record.

## What this does and does not say

- **These are CONTROLS, and that is their whole purpose.** Both are plain `malloc`/`free`
  use-after-free, which ASan reports. They are not evidence for the mechanism this project claims;
  they are what makes the *nested* rows' silence attributable instead of merely silent.
- **Neither adds a nested shape.** No unused member of memcached's lifetime-worded history is a slab
  defect; the only nested shape among them (`5267f14`, a double return into the proxy's rctx cache)
  needs a Lua state. memcached's nested column stays where it was.
- **No native/ASan arm was run here.** The comparator claim is inherited from the class, not measured
  in this bundle.
- **QEMU only**, and the temporal faults are the emulator untagging a revoked capability on reload
  (Q-11); the deployed silicon lets such an access retire.
- **N = 1 per cell**, eight boots: one each for `level0` and `shrink` (two fixtures), and one per
  faulting fixture on `sublet`, `slabsublet0` and `slabsublet1`.
- **The port's own oracle did not run.** `host/run-safety.py` drives each fixture through
  `capstone-vm exec`, and this rootfs has no dropbear — nor `mc-harness`, nor any capstone tooling.
  The guest command here is the one that runner issues, replicated over the serial/9p path, with
  `mc-harness` cross-built for the guest. Its predictions-vs-outcome comparison was therefore done by
  hand against the same expect file. **A hand check is not the gate.**

## Platform

The assembled pinned platform documented in
[`ports/ffmpeg/app/results/2026-10-03-qemu-upstream-defects-heap-arms/`](../../../../ffmpeg/app/results/2026-10-03-qemu-upstream-defects-heap-arms/README.md):
application-SDK images need the delegated runtime's process ABI (SBI `0x21`–`0x2b`), so the monitor is
`deleg-gate2/opensbi-T`'s `fw_jump.elf` and QEMU is `qemu-12`, a checkout at exactly this branch's
pin. Kernel command line `cma=1024M`.

Build cost, measured rather than estimated: libevent 199 s (its own `make verify` flagged one
timing-sensitive test, `util/monotonic_prc_fallback`, which the script re-ran 6 of 6 clean and
accepted — the script is built for that), arm SDKs 14 s, safety images 10 s, the slab image 21 s.

Files: `result-lines.txt` (every line above).
