# Split roadmap tradeoffs after the architecture review
Timestamp: 2026-05-08T14:31:30+08:00
## Summary
This note records the strategic conclusion reached after reviewing an alternative
architecture proposal.
The main conclusion was:
- a full native `capstone64-unknown-linux-gnu` userspace remains a valid long-term
  goal, but it is too expensive and risky as the near-term milestone for Paper I;
- the more practical near-term direction is a split host/enclave roadmap;
- that split roadmap still needs a careful ABI design and should not be described
  as an automatic path to large hosted software.
## What the review agreed with
1. The repository already has a working host-side control path:
   host userspace -> `/dev/capstone` -> kernel module -> SBI/runtime -> domain.
2. That existing control path makes a split host/enclave architecture a natural
   paper-oriented milestone.
3. A small domain-local runtime or libc is a more realistic fit for the early
   domain environment than a full hosted Linux libc.
4. `newlib` or a similarly small libc may become reasonable later if the target is
   a statically linked domain runtime with proxied host services.
## What still required caution
1. The split model does not make the hosted-software problem disappear.
   A `puts()`-level host call is only the first step; larger workloads would still
   need buffering rules, result propagation, error handling, and likely a domain-
   specific file or service contract.
2. Passing raw pointers between the isolated domain and the host side is unsafe as
   a default assumption. Shared-memory or copy-based protocols are more realistic.
3. A resumable trap/resume syscall-proxy ABI had not yet been validated. At that
   point it was still only a hypothesis about a future architecture direction.
## Resulting near-term recommendation
For Paper I, the recommended near-term milestone was:
> Build a split host/enclave runtime with a minimal host-call ABI and a small
> domain-local runtime, instead of attempting a full native hosted Capstone Linux
> userspace first.
That recommendation did **not** invalidate the earlier hosted-userspace analysis.
It changed the chosen milestone.
## Practical next-step ladder from that review
1. Do not rely on the Buildroot glibc sysroot for domain code.
2. Build a domain-local runtime/sysroot.
3. Define a minimal host-call ABI.
4. Prove one tiny request such as `puts()` or `write()`.
5. Extend only after the ABI and buffering model are validated.
## Long-term caveat
If the project later returns to the full hosted-Linux goal, the earlier loader,
libc, syscall-ABI, and process-model issues will still need to be solved.
