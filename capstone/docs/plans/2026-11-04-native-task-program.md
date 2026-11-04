# Native-task program: removing the boundary cost that is not inherent

Status: PLAN, 2026-11-04. It organises the work toward the north star (2026-11-03-boundary-north-star.md). It authorises no
ISA or Linux change by itself; each lane lands separately, default off, after independent review and the nine runtime gates.

## Where we are (census 7, full, gcc, M5P on, dev a476237c079b)
Overhead against plain RISC-V Linux: program +25.1 points, boundary +36.7 points (M5P off: +44.9). One point is about 318,000
attempts. Boundary by location: extra pages ~14 points (about 11 are the 16-byte pointer footprint of gcc's own data and are
inherent under the 128-bit ABI), fault path/pin/module/locks ~10, exits/syscalls/copy/launcher/firmware ~8, kernel tag work
~2.6, rest ~2. The buckets classify location, not cause; they are not a ledger of removable work.

## Target
Domain = ordinary Linux task: no launcher, no exchange round trip, faults and syscalls handled in the kernel, tags managed by
the kernel's memory manager, capability-checked syscall arguments, kernel-held signal frames, lifetime namespace per address space.
Honest range for gcc boundary after all lanes: 20-30 points (a first estimate of 13 is a stretch target that needs another 7-17
points of demonstrated removal). Done means every remaining excess cost is measured and justified as necessary.

## Interfaces (versioned; each lane supplies an executable test double, unsupported operations fail closed)
1. Context/event ABI: lossless capability tags and payloads, integer/FP state, PCC, namespace identity, translation state, fault
   classification, restart PC; contexts belong to threads, namespaces to address spaces; slot eviction/restoration mandatory.
   Two backends behind one ABI: (phase 1) trampoline + context ioctl handled in the kernel; (phase 2) the arch return-to-user hook.
2. Grant transactions: mmap mints a grant, munmap retires before release, rollback, MAP_FIXED, permission attenuation, remap;
   whole-grant-only subset first.
3. Copy: authenticated capability reference {protected-frame-slot, context-epoch}; copy(ctx, capref, offset, kernel-buffer,
   length, direction) -> {bytes-done, cause}; never reconstruct authority from scalar addresses. Interim: the held-exchange
   path (lane AB V1); target: a single-hart supervisor check executed before each non-faulting chunk, fault resolution that
   never writes user memory.
4. Syscall descriptors: nr -> argument kinds; unknown calls ENOSYS; fd/path policy and security hooks explicit.
5. Signal frames: kernel-held, nested, authenticated sigreturn, restart rules, capability-aware ptrace.
6. Image relocation/initialisation format.

## Lanes
| Lane | Scope | Exit criterion | Owner model |
|---|---|---|---|
| Q | Contract, evidence ledger, integration, census accounting | three matched full census pairs; P+K+H reconcile; no instrumentation in scored builds | integration owner |
| T | Bulk tag clear on page reuse, raw COW/migration, CPTAG | tagged clear <= zeroing + 128 attempts/page; zero per-granule sweep | Codex |
| F | Fault fast path, grant-aware mapping transactions | fault <= 4.4K attempts, >= 1.2 gcc points; zero grant/mapping mismatch under injection | Codex |
| N | Native task, context ABI, slot virtualisation | exec <= 1.2M (target 0.4-0.8M); two tasks share one slot; 14/14 census PASS through execve | Codex |
| U | Checked kernel access (single-hart check, restartable copy) | 1-byte pipe read <= 4K, 64 KiB <= 48K; zero unauthorised bytes | Gemini |
| S | Descriptor dispatch, SDK stubs, fd/path policy | scalar/map services plain +500 attempts; required calls pass original oracles | Codex |
| L | fork/exec/exit, signals, sigreturn, ptrace | fork < 300K for a 444-page process; zero per-arena exit loop | Claude |
| P | Avoidable pages, image layout | 30-100 fewer faults, 0.7-2.5 gcc points | Codex |
| PB | M5P on makes perlbench's boundary worse (+7.2 points) | ON <= OFF on perlbench | Codex |
Critical path: contract freeze -> N and U -> S and L -> release qualification. T, F, P and PB do not wait.

## Risks and the first experiments
1. Check-then-copy stays safe across every retry: force a thread switch between chunks and during COW resolution while
   revoking and reallocating the object and the mapping; the unsafe control must corrupt the sentinel.
2. Protected state becomes ordinary schedulable task state: alternate two namespaces through one slot with nested signals, then
   fork and revoke in the child only; a deliberately skipped namespace restore must fail.
3. The budget reaches the target: combine native x fault-fast on/off with M5P fixed on, count actual events, price context work
   in cycles; this exposes overlapping savings.

## Rules for every lane
Default off until qualified; a negative control for each invariant touched (no capability usable after its lifetime ends; tags
never survive a non-capability write, DMA included, or a page's reuse; a domain reaches nothing outside its grants; the kernel
acts on a buffer only through a validated capability and never into recycled memory; one process cannot affect another's
lifetimes); attempts per event against plain Linux and the seven-benchmark census; nine gates on both hosts through the gate
queues; independent cross-model review.
