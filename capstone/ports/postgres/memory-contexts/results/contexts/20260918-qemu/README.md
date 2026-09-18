# PostgreSQL 17.0 context allocator checks

Allocator-level QEMU validation, not a protected PostgreSQL server, a consumer
defect reproduction or a performance measurement. The source pin is the parent
component's `upstream.json`; the new versioned patches are 0005–0007.

`matrix.tsv` records 36 paired cases: **72/72 arms passed**, including 25 Sublet
faults at the declared read/write instruction. All spatial arms complete. Live
Sublet controls, policy/payload workloads and exhaustion/recovery arms complete.
Cases unavailable upstream are excluded: Bump has no individual free/realloc;
Slab only accepts its fixed allocation size.

The policy workload hashes PostgreSQL's logical block/space counters at the
same points in the spatial and Sublet arms. Native ABI geometry differs and is
not compared with capability-ABI hashes.

| Manager | Spatial and Sublet policy hash |
|---|---|
| Generation | `5568948294794541063` |
| Slab | `13966694474297062898` |
| Bump | `76013758635295445` |

Both mixed-manager A11 replay arms passed payload and runtime-accounting checks.
All eight native CTests passed. All seven QEMU CTests have passing runs: the
initial invocation passed six, while the original AllocSet Sublet replay
stalled after printing its domain result and before host cleanup completed.
That attempt failed; an explicit rerun of the unchanged binary passed with
60-second login/command timeouts. The stall's cause is not established here.
The native mixed fixture's backing oracle is 49/48/0/7 (taken/returned/grown/peak);
independent mutations of the block and realloc counts are rejected. Native
verdict controls reject missing markers, wrong PCs/causes, duplicate faults,
failed controls and missing policy results. Selecting no applicable matrix arms
is also rejected. All six attempts to compile the new managers with either
`MEMORY_CONTEXT_CHECKING` or `CLOBBER_FREED_MEMORY` hit the intended guard.

The full final matrix completed in one invocation without retries. Earlier
development campaigns retained failures before login and after loader segment
output; those attempts were not counted as allocator passes. Their raw
logs remain local; only result lines and input identities are committed.

Reproduce with the component README's CMake/CTest commands. These domain builds
use CMake `Debug` (no optimization flag) and PostgreSQL release-layout headers.
No silicon or optimized-build result is claimed. `inputs.json` fingerprints the
tested domains, loader, compiler, emulator, region sizes and revocation budget.
