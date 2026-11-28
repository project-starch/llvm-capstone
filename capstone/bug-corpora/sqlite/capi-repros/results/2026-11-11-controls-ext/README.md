# SQLite 3.22.0 lookaside: parent, sibling and successor controls on the one-operand lifetime instructions (2026-11-11)

The controls of [`../2026-10-12-controls-ext/`](../2026-10-12-controls-ext/README.md), built and run again after the
lifetime instructions changed. In the lookaside half of `sublet-3220000-memsys5.patch`, `lookasideRelease` now ends the
slot's lifetime with `capstone_cap_release(p)`, and the lookaside's generation `pLife` is a direct child of the buffer:
the intermediate `lookaside.pBlock` is gone. The cases, probes, builder and runner are in
[`../../controls/lookaside/`](../../controls/lookaside/); none of them changed. The full record is `results.json`.

**Outcome: every prediction held, on both arms, in three boots per image (r1, r2, r3).** Every reading equals the
2026-10-12 record's.

| control | release (lookaside as shipped) | protected (`sublet-3220000-memsys5.patch`, lookaside half) | prediction |
|---|---|---|---|
| parent | completed. A's former range read through `lookaside.pStart + 129600`. The typed link read returned `pNext` = D's address (tag 1). Write and read-back ok. A, then D reissued | completed. Same, through `lookaside.pStart + 118800`, `pNext` = D (tag 1) | held, both |
| sibling | completed, B intact, write and read-back ok | completed, B intact, write and read-back ok | held, both |
| successor | completed. C = A | completed. C = A | held, both |
| successor, stale read | completed. The read through old A returned C's bytes | **fault, cause 25**, at `stale-A-read` (`control-successor/control.c:71`). pc = image offset `0x629dc` in `run_case`. Fault address = A's start | held, both |

Every A, B, D and C passed `CHECK <name>-is-lookaside-slot` on every arm and run. On the protected arm the block's
capability bounds are the slot, `[A, A+1200)`; the lookaside has 100 slots there and 109 on the release arm, as on
2026-10-12.

**The slot check can fire.** `slot-check-negative`, one boot per arm: the 128-byte request is `ok`, the 2000-byte
request and the request with the lookaside disabled are `FAILED`, on both arms. Images `05cea972537f...` (release) and
`d8d3029b4fc2...` (protected).

## What differs from 2026-10-12

- The patch, and so the protected staged engine and the protected images. The release staged engine is the same bytes
  (19ceadb5d103...).
- The platform, the SDK and the compiler (below). The stale fault's image offset moved from `0x673b0` to `0x629dc`
  with the compiler; it resolves to the same source line.

## Images and platform

| image | release sha256 | protected sha256 |
|---|---|---|
| la-control-parent.dom | `0fc1ff5efc7ab48d1d78400f1c225d210878101a8251c8cf052fe8ffd5fd53e5` | `73d3442126e434e3edef55799d3d39a22a548e7373e1ba041974cb22b2a8a47e` |
| la-control-sibling.dom | `7cde4d2c50f7a01a0539893860dbfb55b77c8866401eee56cc2877686537c529` | `ae5839545784068ed9e2a064ac0467c5e9c60d2505f87b4cf533e28896447396` |
| la-control-successor.dom | `1628cd29b7888403ad1276b6bda4cd71e5f0be636fe549b69b7632770b81426c` | `72115d52250f722e51dda8fc9398fdf0f135871dadaf347364a0ae8e8af2f290` |
| la-control-successor-stale.dom | `e35fe39cd41627ca99bcddf88c5d336dbff4f76db4ef0257c37b6e5db46da78f` | `bc8aef1a2f29aaf9eebd9842ffd77e2d8dffe524e57f33caa4e98e6f1691fa56` |
| staged sqlite3-capstone.c | `19ceadb5d103cab190dc4a76b3a54f457495fa1cf28ec5befcd0d5c1fa0c4d66` | `49812bb92722eb4acab6d662508070ab81faf10e4c07ab6cd3dd9fa867954fec` |

The staged engines are the memsys5 controls' engines of [`../2026-11-11-controls/`](../2026-11-11-controls/README.md).
The protected images carry 8 `CDERIVE` and 4 `CREVOKE` encodings, the release images the C library's 6 and 3.

Tree, SDK and platform are those of the memsys5 controls of the same day: branch `one-operand-crevoke-crenew` at
59201bdf2, QEMU 2d5a71078904..., launcher 0f8e0f903725..., module e47bca62addf... loaded without parameters, guest CPU
with `x-capstone-test-csrs=true`. The hashes are in `results.json`.

## Wall times

One boot per image, 8 VMs at once: 26.3 s, 25.3 s and 25.7 s for the three sets.

## Notes

- A realloc that keeps its lookaside slot (`lookasideRenew`, the block port's `CRENEW`) is not reached by these
  controls: they use the request-bounds port, as on 2026-10-12.
- No run was discarded.
