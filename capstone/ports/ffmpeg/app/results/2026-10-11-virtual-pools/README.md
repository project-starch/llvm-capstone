# FFmpeg's own pools protected in the application, virtual profile, 2026-10-11

The decoder application with the buffer-pool port's patch 0003 compiled into its libavutil
(`FFAPP_SUBLET=1`), against the same application with FFmpeg's pools as released. Both images
come from one source snapshot of llvm-capstone `265e4eb47677` (manifest `runtime_revision`
18ab68e6499f is that snapshot's local commit, `runtime_dirty` false), built by
`common/application/build-virtual.sh ffmpeg` with `FFAPP_EXTRA_FIXTURES="11 12 13 14 15"`.
Predictions for both arms were registered in `host/safety-expect.txt` in that commit, before
either image was built.

| arm | image (`ffmpeg.dom`) | nested |
|---|---|---|
| `vstock` | 385f958631d8 | none |
| `vpools` | 6993865e22c4 | ffmpeg |

Platform, one boot per check, exact bounds: QEMU af37cc32 (Sublet-lifetime build), launcher
3975314a, module a1c6cb6b, Image e833dfb12bcc (sha256 prefixes; the same platform as
`bug-corpora/ffmpeg/pool-repros` and `plain-temporal-repros` 2026-10-11). Compiler: clang from
llvm-capstone cda7e335ff9f.

The protected libavutil carries the lifetime instructions and the stock one does not: opcode
0x5b appears 8 times in `buffer.o` and 5 times in `refstruct.o` of `vpools`, and in neither
object of `vstock`.

## Decode

`decode-result.json`. Both images decode the run-of-record clip (input.mkv 7c0964440ad1) and
its one-byte-flipped control (27e4bed2f712), exit 0, 30 frames each, every frame hash equal to
the native reference (native.out 3eebf311c9fa, native.flip.out a7854e605fb8). The flipped
input changes 10 of the 30 frames on both images, so the comparison can fail. This shows the
patch does not change the decoder's output on this workload; it does not count how many pool
entries the decode handed out and revoked.

## Pool fixtures 11-15

`safety-vstock.json` and `safety-vpools.json`, judged by `runtime/virtual/run-safety.py`
against the registered predictions: PASS on both arms, 5 of 5.

| fixture | access | vstock | vpools |
|---|---|---|---|
| 11 | in bounds of a live buffer | RETURN b00001 | RETURN b00001 |
| 12 | stale buffer after it returned to its pool | RETURN c000a0, its old byte | FAULT temporal, cause 25 at the target |
| 13 | stale buffer after the pool reissued it | RETURN d0015b, the new owner's byte | FAULT temporal, cause 25 at the target |
| 14 | one byte past a 64-byte buffer | FAULT oob | FAULT oob, cause 28 |
| 15 | stale AVRefStructPool object after its return | RETURN f000a0, its old byte | FAULT temporal, cause 25 at the target |

The vpools fault line of fixture 12, as the launcher printed it:
`capstone-exec: domain fault cause=25 pc=0x3f8e291024 address=0x3f8e399440`, after
`FFAPP-FIX 12 target=3f8e399440`; QEMU reports "Cap mem access on revoked capability".

Not run, as registered: 16 (a refstruct object outside any pool, untouched by 0003) and 17
(its stale access reads the RefCount header, which the outcomes cannot express). One run per
arm (N = 1).
