# Shared-region RPC roadmap for the split host-enclave path
Timestamp: 2026-05-08T15:13:28+08:00
## Executive summary
After checking the repository sources, the recommended split architecture was
refined into a more conservative and source-backed form:
> Use synchronous multi-round RPC over shared regions as the first host-call ABI,
> rather than assuming a resumable yield/resume syscall-proxy mechanism.
## Source-backed findings
### 1. The repository already contains the core split substrate
The existing `miniweb` path already demonstrates:
- host-side region creation and mapping,
- region sharing into a domain,
- repeated host <-> domain rounds,
- synchronous domain returns via the current runtime path.
That means the split direction is not starting from zero.
### 2. The first ABI should reuse the existing region API
The existing region operations already provide the right shape for an initial
request/response channel:
- `create_region(len)`
- `map_region(region_id, len)`
- `shared_region_annotated(dom_id, region_id, perm, rev)`
- `call_dom(dom_id)`
This makes an annotated shared-region RPC model more credible than inventing a
new ad hoc memory-grant mechanism.
### 3. A general-purpose host-visible resumable trap ABI was not yet proven
The audited user/kernel ABI showed calls such as `IOCTL_DOM_CALL` and region
operations, but no validated host-visible `DOM_RESUME`-style mechanism that would
support "domain yields, host inspects registers, host resumes at the same PC".
So the safe conclusion at that point was:
- synchronous return paths were real,
- a general resumable syscall-proxy ABI was not yet established.
## Recommended ABI shape for v0
The recommended first prototype was a memory-based request/response ABI using two
shared regions:
1. a metadata region,
2. a bounce-buffer region.
Example metadata shape:
```c
struct hostcall_v0 {
    uint64_t phase;
    uint64_t opcode;
    uint64_t offset;
    uint64_t length;
    int64_t  result;
    int64_t  error;
};
```
Suggested control flow:
1. host creates the domain,
2. host creates and shares the two regions,
3. host calls the domain,
4. the domain writes a request and returns `HC_PENDING`,
5. the host performs the service and writes back the response,
6. the host calls the domain again,
7. the domain consumes the response and continues.
## Why this was the preferred first milestone
- It avoided new kernel ABI work.
- It reused already validated primitives.
- It tested the architectural contract directly.
- It provided a safer path toward later libc work.
## Recommended next milestone from that review
The smallest meaningful next milestone was:
> a tiny hosted-output proof of concept using shared metadata, a bounce buffer,
> and two `call_dom()` rounds.
At that stage the recommendation was explicitly **not** to jump straight to:
- a full hosted Capstone Linux userspace,
- a `glibc`/`musl`/`picolibc` port,
- a resumable host-visible syscall ABI,
- or `sqlite`-class software.
