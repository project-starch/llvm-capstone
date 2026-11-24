# SQLite 3.22.0 memsys5: parent, sibling and successor controls on the one-operand lifetime instructions (2026-11-11)

The controls of [`../2026-10-11-controls/`](../2026-10-11-controls/README.md), built and run again after the lifetime
instructions changed: `CREVOKE` takes the object alone, and `sublet-3220000-memsys5.patch` was converted to it
(`sqlite3_free` ends the block's lifetime with `capstone_cap_release(pOld)`, and the pool is a direct child of the
configured heap, without the intermediate `mem5.pHeap`). The cases, probes, builder and runner are in
[`../../controls/`](../../controls/); none of them changed. The full record is `results.json`.

**Outcome: every prediction held, on both arms, in three boots per image (r1, r2, r3).** Every reading equals the
2026-10-11 record's.

| control | release (memsys5 as shipped) | protected (`sublet-3220000-memsys5.patch`) | prediction |
|---|---|---|---|
| parent | completed. Read of A's former range through `mem5.zPool + 1032192` = memsys5's link `{next 15874, prev -1}`. Write and read-back ok | completed. Through `mem5.zPool + 972288` = `{next -1, prev -1}`. Write and read-back ok | held, both |
| sibling | completed, B intact, write and read-back ok | completed, B intact, write and read-back ok | held, both |
| successor | completed. C = A | completed. C = A | held, both |
| successor, stale read | completed. The read through old A returned C's bytes | **fault, cause 25**, at `stale-A-read` (`control-successor/control.c:73`). pc = image offset `0x628d8` in `run_case`. Fault address = A's start | held, both |

The pool offsets and block indices are those of 2026-10-11: A at index 16128 on the release arm and 15192 on the
protected arm, where A and B are buddies. On the protected arm A's capability is bounded to its 128-byte request, and on
the release arm it carries the whole 1 MiB heap array's bounds.

## What differs from 2026-10-11

- The patch, and so the protected staged engine and the four protected images. The release staged engine is the same
  bytes (19ceadb5d103...).
- The platform, the SDK and the compiler (below). The stale fault's image offset moved from `0x672a4` to `0x628d8` with
  the compiler; it resolves to the same source line.
- Three boots per image where 2026-10-11 had two.

## Images and platform

| image | release sha256 | protected sha256 |
|---|---|---|
| control-parent.dom | `9b7a8e2ae7a234525eb575d6c10bfbd2a1fc9e3d25e0d0ab38cca3d72b193a75` | `918f88429149e176971763ad5ba82e19d41be03c90fab8852836182596e70c3e` |
| control-sibling.dom | `5a2a91f4c3bea9fc5398f3ba0bb78574f93878bdb7d102828b2b41330da6025e` | `f10840718cbb2351ddf5709474f7c684a56c3ffa23448dd7c7f1e9ef1a4ba06e` |
| control-successor.dom | `c222ae06d55ff56bfe15557a6d3ca30921c5261c246c62fe114d61cf807bbc5a` | `5c29d82fdbf6b1add41c2fe84288e90574346165fbca2560709fb682fc071ac5` |
| control-successor-stale.dom | `84815e9f6c24178c9983ca6eba43b889488d85470192edbaba5ccc07a11fdef6` | `b838e012fe3f7e7a6f3ac8e7a3c28792dae60c8bf8156308d2184bbe4f03eef2` |
| staged sqlite3-capstone.c | `19ceadb5d103cab190dc4a76b3a54f457495fa1cf28ec5befcd0d5c1fa0c4d66` | `49812bb92722eb4acab6d662508070ab81faf10e4c07ab6cd3dd9fa867954fec` |

The protected images carry 8 `CDERIVE` and 4 `CREVOKE` encodings, the release images 6 and 3: the C library's
mallocng has its own (6 and 3), so the patch adds 2 and 1 at `-O0`. No image has a `CRENEW`.

- Tree: branch `one-operand-crevoke-crenew` at 59201bdf2, QEMU at ea57497908. `sublet-3220000-memsys5.patch` is
  f96ee0329a08....
- SDK: a corpus SDK (`CAPSTONE_SDK_CORPUS=1`, effective `-fno-builtin`) from `runtime/virtual/build-sdk.sh`.
- Platform: QEMU 2d5a71078904..., launcher 0f8e0f903725..., module e47bca62addf..., loaded without parameters (pinned,
  one context). The kernel, firmware, rootfs, compiler and SDK libraries are in `results.json`. The guest CPU adds
  `x-capstone-test-csrs=true`, which a pinned boot of this kernel needs. `run-controls.py` boots through
  `runtime/virtual/run-staged.py` with its default bounds mode, one hart and 2 GiB.

## Wall times

One boot per image, 8 VMs at once: 24.7 s, 25.0 s and 25.1 s for the three sets, each boot 22.2 to 24.8 s. Other
boots of the same campaign ran on the host at the same time.

## Notes

- `run-controls.py` gives every boot a lock file of its own. The three sets ran inside one exclusive hold of the
  host's shared QEMU lock.
- No run was discarded.
