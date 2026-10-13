# Nested-lifetime controls for ffmpeg/pool-repros

The FFmpeg row of the pre-registered parent, sibling and successor controls
([`docs/plans/2026-10-11-nested-lifetime-controls.md`](../../../../../docs/plans/2026-10-11-nested-lifetime-controls.md)).
The corpus cases show that patch 0003 rejects stale accesses to `AVBufferPool` storage; these
controls ask whether it leaves the permitted ones alone. Each drives FFmpeg 9.0.1's own
`libavutil/buffer.c` through its API, linked the way the corpus links it (the buffer-pool port's
`capstone-application` library, memory from the process's virtual mallocng).

| control | sequence | release arm (`FFPOOL_SUBLET=OFF`) | protected arm (`FFPOOL_SUBLET=ON`, patch 0003 on 0001) |
|---|---|---|---|
| `91_ctl_parent_entry_reference` | get A, fill 0x5A, `av_buffer_unref(&A)`, then read, write and read back through `BufferPoolEntry.data` | completes, reads 0x5A | completes, reads 0x5A; also through `sublet_life` |
| `92_ctl_sibling_entry` | get A and B, unref A, read, write and read back B through `B->data` | completes, B intact | completes, B intact |
| `93_ctl_successor_entry` `run` | unref A, get until C is at A's address, write and read C through `C->data` | completes, C == A | completes, C == A |
| `93_ctl_successor_entry` `stale` | the same, then one read through A's old pointer at `stale_read_probe` | completes, reads C's 0x77 | fault, cause 25, at `stale_read_probe` |

The release operation is `av_buffer_unref`, which reaches `pool_release_buffer`: the point where
patch 0003 revokes the user's child (`CREVOKE`). The pool is `av_buffer_pool_init(64, NULL)`, so each
entry's storage is `av_buffer_alloc(64)`.

## Which reference is the parent (control 91)

`BufferPoolEntry.data` (`libavutil/buffer_internal.h`) is the pool's own pointer to an entry's
storage. `pool_alloc_buffer` stores there what the pool's allocator returned
(`buf->data = ret->buffer->data`), and `buffer_pool_flush` frees the storage through it when the pool
ends. Patch 0003 leaves that field as it is ("the entry keeps its original pointer"). From it the
patch derives the entry's lifetime, `sublet_life = CDERIVE(entry->data)`, and from that, at every
hand-out, the user's child, `sublet_issued = CDERIVE(sublet_life)`. That child becomes
`AVBufferRef.data`, and `pool_release_buffer` revokes it. So `entry->data` is the root of the subtree
that the release cuts a leaf off. It is the reference the pool manages the storage through. On the
protected arm, control 91 also reads and writes through `sublet_life`, the direct parent named by
`CREVOKE`.

The control reaches the entry through `A->buffer->opaque`, which it reads before the release. That
is the `opaque` that `pool_release_buffer` receives. After the release it never touches A's
`AVBufferRef` or `AVBuffer`: the first hand-out's `AVBuffer` is freed at the unref.

## Layout and how to run

    shared/ctl.h               probes' conventions, LCC metadata, CTL-* output lines
    shared/driver.c            main(), the pool; `ctl-NN run NN` / `ctl-93 stale 93`
    shared/build-controls.sh   a copy of controls/virtual/shared/build-cases.sh that takes the arm
    shared/run-controls.py     one boot per arm (virtualvm.Batch); `combine` judges the predictions

    CAPSTONE_SDK=<virtual SDK> bash shared/build-controls.sh capstone-application <out> protected|release \
      -DCAPSTONE_LLVM_BUILD_DIR=<toolchain> -DFFPOOL_ARCHIVE=<ffmpeg-9.0.1.tar.xz>
    CAPSTONE_QEMU_LOCK=<own lock> python3 shared/run-controls.py boot --arm protected|release \
      --build <out> --sdk <SDK> --kit <kit> --llvm-bin <toolchain>/bin --raw <raw> --out <boot.json>
    python3 shared/run-controls.py combine --manifest <SNAPSHOT.json> --out results.json <boot.json>...

`build-controls.sh` refuses a library whose `buffer_internal.h` does not match the arm: patch 0003
must be present exactly when the arm is `protected`. The control includes that header, and
`BufferPoolEntry`'s layout differs between the two arms. `run-controls.py` refuses a build whose
`CMakeCache` names another `FFPOOL_SUBLET` or another SDK. It also refuses a run whose `case=` line
names another control, arm or mode.

Each boot runs the virtual heap's own controls first: `bounds-malloc` and `uaf-malloc` from
`postgres/c-repros/shared/controls.c`. Both must fault on both arms, so that a column of completions
cannot come from a dead instrument. Then it runs 91, 92, 93 `run`, and 93 `stale` last. The `combine`
judge was negative-tested before the run. With the arm labels swapped, a mismatch count injected
into control 92, and a platform control set to `complete`, each affected prediction read NOT HELD.

Results: [`../../results/2026-10-11-controls/`](../../results/2026-10-11-controls/README.md).
