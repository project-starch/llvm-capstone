# 20261029-hybrid-ab -- is the allocator's abort capability-enabled?

Fourteen of the 56 corpus cases end at the platform default in `SIGABRT` raised by the process on
itself, with nothing on stderr -- CheriBSD's allocator consistency check rather than a capability
fault ([`../20261029-allocator-probe`](../20261029-allocator-probe)). That alone does not say
whether the check belongs to CHERI or to jemalloc, and the corpus counts those aborts as catches,
so the question has to be settled rather than argued.

`controls/abort-probe.c`, built twice from the same source and run in ONE boot of the same guest
against the same jemalloc. The ABI is the only variable.

| stage | purecap | hybrid |
|---|---|---|
| malloc then free | 0 | 0 |
| **free the same pointer twice** | **SIGABRT** | **0, survived** |
| read a byte after free | 0, reads the old byte | 0, reads the old byte |
| write a byte after free | 0 | 0 |
| **free an interior pointer** | **SIGABRT** | **0, survived** |

So the check is **capability-enabled**: hybrid survives the same double free that aborts under
purecap. The reading rests on three internal controls in the same boot -- stage 0 returns 0 under
both ABIs, so the probe can report a survival at all, and stages 2 and 3 return 0 under both, so
the abort is specific to an invalid free rather than something any use-after-free provokes.

Note what stages 2 and 3 also say: a plain read or write after free does **not** trap at the
platform default under either ABI. The chunk is quarantined, not revoked, and the stale capability
stays valid until a sweep runs.

`matrix.tsv` is the per-stage reading and `inputs.json` carries both binaries' hashes and ELF
flags. The transcript is not committed.
