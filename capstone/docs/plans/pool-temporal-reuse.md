# Protected pool reuse in QEMU

Status: completed on the experiment branch based on PR #61.
[Measured results](../../ports/ffmpeg/buffer-pool/results/measurements/20260919-temporal-reuse/README.md)
include the original campaign, the separately labeled extension, and failures.

## Question and hypothesis

Does repeated return/reissue at a fixed payload address accumulate protection
state or trigger global reclamation, while a sibling allocation stays live?
The hypothesis is that local Sublet revocation avoids PICASSO color recycling
pressure. Both systems must reject retained stale leases before comparing
their protected behavior. A null result or a Capstone limitation is retained.

## Arms and controls

Use the same FFmpeg pool implementation and 64-byte allocation API. Two blocks
stay allocated: one sibling and one repeatedly returned/reissued block. Check
the same address on every reuse and both current payload values. Retain one
old pointer throughout. Case 13 completes with valid accesses; case 12 performs
the stale access after the identical churn. Also run valid alias/callback/
deferred-close and short stale-return/reuse controls.

Capstone uses Sublet mode 2 and the existing pinned QEMU. PICASSO uses an
explicit adapter: mmap supplies the trusted arena authority; a real 64-byte
libc allocation supplies each fresh color, copied onto a bounded payload
capability with recoloring permission removed. Freeing the token revokes that
lease. Tokens are an implementation choice and their storage must be reported
separately, never presented as an unavoidable PICASSO cost. No kernel, libc or
emulator modifications are required. The same native recordings must still
replay event-for-event through this adapter.

This adapter is not hierarchical allocator protection. It assumes a trusted
pool manager with recoloring authority. A child receives a flat, independent
color; parent invalidation does not automatically invalidate that child color.
The PICASSO paper's sections 3 and 6.2 describe a trusted malloc/free allocator,
and section 7 does not separately evaluate nested parent/child lifetimes.
Testing PostgreSQL is not itself evidence of protected Memory Context returns.
A separate experiment must test parent revocation with independently live
children and count the additional bookkeeping needed to preserve that property.

## Measurement protocol

Bring-up: 1,000 reuse rounds and all companion controls. Main campaign: 300,000
rounds, three fresh guests per arm. Checkpoints
after round 1, every 10,000 rounds, and the final round. If an execution fails,
retain it and its reason; do not silently retry or change the threshold.

Protocol correction after the first 300,000-round result: the initial
expectation that this crossed PICASSO's threshold was wrong. The installed
machine header defines 21 color bits, not 18; the compiled threshold is 2,095,148.
Keep all 300,000-round results. Add an explicitly separate 2,200,000-round
extension, one fresh guest per valid/stale Capstone case and one PICASSO guest
running both cases in separate processes. Increase Capstone's declared node
capacity to 4,194,304 for this extension. It tests a mechanism beyond the actual
threshold; it is a single repetition, not part of the three-repeat campaign.
The new PICASSO binary prints its compiled color width and threshold.
The first Capstone extension attempt hit its inherited 90-second guest-command
timeout after the 860,000 checkpoint. It is retained as incomplete, not an
allocator failure. The separately named replacement uses a 900-second command
budget, with the binary, node capacity and workload unchanged.

Record payload carving, current/peak live tokens, cumulative token issues and
returns, runtime busy color IDs (`malloc2(0)`), and the runtime's exit-time sweep
count (`CC_DEBUG`). Capstone snapshots report node high-water and free-list
length; their difference is allocated nodes not on the free list, not the live
object count. The pinned emulator allocates unused IDs before consuming its
free list, so high-water alone is not minimum capacity or physical cost.

The main outcome is successful reuse and protection-state behavior, not QEMU
wall-clock performance. Do not equate a color ID with a Capstone node in bytes.
Neither counter is a complete protection-memory ledger. Fixed arena/harness
reservations, token payload and runtime metadata remain distinct. Synthetic
churn isolates a mechanism; it does not establish its frequency in applications.

Source: [official PICASSO artifact](https://github.com/coloredcapabilities/colored-artifact),
installed libc `mrs.c` color allocation, free and recycling paths. Exact build
and input hashes accompany the result bundle.

## Next semantic experiment: revoke a parent with live descendants

Keep a sibling region alive while allocating 1, 16, 256 or 4,096 independently
leased children under a second parent. Verify every live child, revoke only
that parent, then probe each retained child in an isolated execution and verify
the sibling remains usable. Include an independently returned/reissued child
before parent revocation. Preserve the parent/child event sequence across arms.

For PICASSO compare the flat-color negative control with an explicitly trusted
adapter maintaining the required child-color bookkeeping. For Capstone use
actual derived authority and ancestor revocation. Count child tracking storage,
color/node state, per-child software invalidations, and reusable payload bytes.
First establish the same lifetime property; do not count failure of an
unadapted baseline as the cost of a protected design. Delegation to an untrusted
child manager is an additional, separately stated requirement, not implied by
the trusted-pool experiment above.
