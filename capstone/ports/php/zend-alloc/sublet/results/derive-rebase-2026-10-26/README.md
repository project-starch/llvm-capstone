# The seam on CDERIVE/CREVOKE, 2026-10-26

The 2026-10-25 merge deleted `__capstone_sublet_malloc_linear` / `_free_linear` from
`runtime/virtual/heap.c` -- the linear loan every PHP block came through, and which this port
was the only consumer of. Linear authority is no longer exported at all: `__capstone_malloc_map`
keeps its grant in the heap's private slot and returns a bare base. So the block PROVIDER was
re-based onto `capability.h`'s CDERIVE/CREVOKE. The seam's SHAPE is unchanged: split at the
48-byte header, retire the payload per `_efree`, never revoke the header.

These are the two probe runs that were taken BEFORE `zend-cap-blocks.c` was touched and after,
in that order.

## probe-derive-split.json -- the primitives, before any seam code changed

| stage | question | result |
|---|---|---|
| 0 | control: carve, write both children, free | exit 0 |
| 1 | geometry | `object_off=48`, header `[obj,obj+48)`, payload `[obj+48,obj+256)` |
| 2 | **is the header still writable after the payload is revoked?** | exit 0 -- yes |
| 3 | does a revoked payload deny? | exit 139, cause 25 |
| 4 | re-derive over the same bytes (the `AG(cache)` hit) | fresh child works, old alias exit 139 |
| 5 | cursor read of a revoked capability | exit 0 -- does NOT trap, returns a bogus value |

Stage 1 is why `pay_off` is computed and never assumed: the object sits 48 bytes INTO the
parent, because mallocng's slot prefix lies below it. An offset of zero would have bounded the
prefix instead of the header.

Stage 5 settles the plan's open risk 2 favourably but not for free: `ZEND_CAP_HEADER(ptr)` on a
double-`_efree` will NOT fault inside the allocator, so the harness will not refuse it as a port
artefact -- but the value it reads is wrong, so the lookup misses and the seam must make that
miss loud. It does (`ZEND_CAP_DOUBLE_FREE`, by name, at a labelled site).

## probe-seam.json -- the REAL seam code, after the re-base

`probe-seam.dom` links the actual `zend-cap-blocks.c` and `zend-cap-backing.c`, because "a probe
with its own copy of the logic would pass while the engine broke". All ten stages match the
expectations documented at the head of `probe-seam.c`:

| stage | expectation | exit |
|---|---|---|
| 0 | positive control, must COMPLETE | 0 |
| 1 | write through a retired payload alias | 139 |
| 2 | read through a retired payload alias | 139 |
| 3 | `AG(cache)`: retire, re-issue to a new tenant, use the OLD alias | 139 |
| 4 | the intrusive list: retire the middle block, walk the neighbours' headers | 0 |
| 5 | realloc moves the payload: contents preserved, old alias dead | 139 |
| 6 | the CONTROL arm, revoke off: the same stale use must COMPLETE | 0 |
| 7 | the bound: one byte past the request | 139 |
| 8 | the platform column on a cache-ABSORBED free: must not fault | 0 |
| 9 | the platform column on a REAL free: must fault | 139 |

Stage 6 is what makes 1, 2, 3 and 5 mean anything: the same access on an arm whose only
difference is that `_efree` does not revoke. Stages 8 and 9 keep a temporal catch attributable
to this seam rather than to mallocng underneath it.

Stage 0's own report:

    S0 header still readable and writable after the payload was retired
    S0 RESULT: PASS -- split, retire, header use, re-issue and block free all behave
    PHP_VIRTUAL SEAM blocks=1 frees=1 issues=2 retires=2 revokes=3 splits=1

Three revokes for two retires and one block: both payload retires, plus the header at block
death. The payload was already dead by then, which is why it is three and not four.

## One judgement that is not forced by the primitives

The children are revoked at block death in BOTH arms, and only the COUNT is arm-dependent. The
loan's return used to rotate the heap's own authority and so ended them whichever arm ran;
`free()` does not. Issuing the revokes unconditionally keeps the level below held constant --
which is what lets the temporal pair differ only in whether `_efree` revokes -- and stops a
child outliving a block mallocng may hand to someone else.
