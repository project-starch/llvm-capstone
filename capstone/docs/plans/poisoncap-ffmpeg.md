# PoisonCap FFmpeg allocator integration

Base: `ports/8-cheribsd-allocators`, commit `46d380be9c6a`.
Work branch: `ports/9-poisoncap-ffmpeg`.

## Objective

Run the extracted FFmpeg buffer/refstruct pools with PoisonCap protection of
individual lease lifetimes. Reuse existing native recordings and the shared
CheriBSD runner. Preserve the existing spatial configuration.

## Platform gate

Reconstruct the published artifact at
`731c5720d1a6321d436a8d7b85bf3aa2e4971fe0` with complete upstream dependencies.
Keep source, SDK, build, rootfs and disk image under a separate
`$CAPSTONE_TMP_ROOT/poisoncap-work` directory. Record exact revisions and any
reconstruction patches; a development branch is not automatically equivalent
to the artifact. Run ABI, poison-read, nested-revocation and live-parent
controls before calling an allocator execution protected.

## First adapter

Use the existing separated payload and manager metadata. On lease return,
poison the bounded payload using retained manager authority. Before issuing
that storage again, require successful synchronous revocation, then detox.
This conservative first implementation measures a sweep-before-reuse policy;
it is not an optimized quarantine implementation or a performance baseline.

Preserve RefStruct state across idle periods: its initialization callback may
run only once. The initial adapter snapshots each rounded payload before
poisoning and restores it after safe reuse. Account for retained snapshots and
both copy directions. This uniform policy is deliberately simple; selectively
preserving only required state is a separate optimization, so its storage cost
must not be presented as inherent to PoisonCap.

Keep user authority narrower than manager backing authority. Verify padding,
compressed bounds and 16-byte poison granularity. Parent pool teardown and
deferred destruction must preserve outstanding live leases until their
declared lifetime ends. A failed or unavailable revocation call must stop
execution before reuse.

## Acceptance

1. Standard CheriBSD/native regressions continue to pass.
2. PoisonCap controls demonstrate live access, stale read rejection, and
   successful new access after same-address reuse; retain failures.
3. Old aliases remain invalid after reuse; unrelated live payload survives.
4. Native-recording replays match logical event and payload oracles.
5. Report poison/clear bytes and sweep counts separately from allocation
   payload and metadata. No QEMU wall-time or cache-performance claims.

If the platform cannot satisfy its control tests, record the exact boundary
reached and the reproducible failure. Do not label a spatial-only run as a
successful PoisonCap port.

## Verified pilot and open gate

The [pilot result](../../ports/ffmpeg/buffer-pool/results/measurements/20260919-poisoncap-pilot/README.md)
records a complete 29-process suite with automatic guest libc revocation
disabled before SSH starts, while explicit per-lease revocation remains active.
Three short replays match exactly. Preserving the guest default instead has a
captured kernel VM-map locking panic, also in a spatial-only arm. Keep those
attempts visible. Extend the explicit configuration with memory accounting;
default-policy stability remains a separate open gate.
