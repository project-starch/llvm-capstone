# Trusted-Linux M1 boundary model

This executable, finite model tests the transitions proposed in
[the M1 execution boundary](../../docs/design/trusted-linux-execution-boundary.md).
Its central contract is **reuse the address, never revive the old object**:

```c
p = malloc(64);
free(p);          // retire, invalidate and drain before permitting reuse
q = malloc(64);   // the test deliberately reuses p's address
*p;              // capability fault
*q;              // successful access
```

These are access-level requirements, not a claim about compiling a C program
with undefined behavior. The model retains both returned capabilities and
attempts both accesses after reuse; it does not reconstruct `p` from `q`.

| Retained pointer | Virtual address | Object generation | Dereference after reuse |
|---|---|---|---|
| `p` | `0x1000` | 0 | Capability fault, before translation |
| `q` | `0x1000` | 1 | Issued and completed |

The same distinction must survive `munmap` followed by `mmap` at the same
address. Linux controls mappings and page permissions; `malloc` controls the
object lifetime within an existing arena. Object reuse cannot create a VMA
or upgrade a read-only PTE. Exhausting the modeled generation space refuses
allocation without changing state; generations never wrap.

## Run and evidence

Run from the repository root:

```sh
source capstone/tests/capstone-test-env.sh
python3 capstone/models/trusted-linux-m1/model.py \
  --record capstone/models/trusted-linux-m1/result.json
```

The deterministic schema-2 [record](result.json) includes the model source
SHA-256, the `malloc64_contract`, per-action outcomes and counterexample traces.
It contains **19 families, 90,735 named-event schedules and 18 distinct injected
faults**, exercised in 21 family/variant combinations. All correct families
pass; each injected fault must fail at its intended property.

| Schedule family | Enumerated schedules | Schedules completing every event, including fresh access |
|---|---:|---:|
| One pending access across `free` | 5,040 | 20 |
| One pending access across `unmap` | 5,040 | 20 |
| Two harts accessing the same context across `free` | 40,320 | 80 |
| Two harts accessing the same context across `unmap` | 40,320 | 80 |
| Directed contracts | 15 | Includes expected refusals |

The permutation families include attempts in invalid orders, such as a drain
before completion or allocation before retirement finishes. Those operations
must refuse without changing state. Such a schedule is still checked but is
not counted as completing every event. Every permutation has a suffix that
attempts the old and new pointers, including demand-fault resolution for a
remap. Coverage requires full successful lifecycles, rejected old pointers,
completed new accesses and refused remote drains with real pending accesses.

The directed contracts cover context/ASID switching, exact-address `malloc`
and remap, retained read-only permissions, live and retired fault retries,
PTE write checks, syscall bounds, unrelated-context progress and private
cloning. Cloning must preserve both stale and fresh inherited pointers and
independent writes. A previously used context remains unavailable to clone
even after full unmap; a dead object is not a fresh process namespace.

## State and transitions

The model has two address-space instances and two harts. Both spaces reuse
the same numerical ASID, virtual page, object node number and initial
generation; their physical frames and lifetime records are distinct. A hart
has separate user, lifetime and translation selectors. Its one-entry
authorization cache is indexed by the reusable ASID and generation. Correct
switching and retirement clear that cache. Each hart can hold one checked
access that retains its target frame after translation.

`Capability` describes semantic tag, bounds, rights, cursor and generation
information; it does **not** specify a 128-bit encoding or where each field is
stored in hardware. Its `namespace` and `birth` fields are ghost provenance,
unavailable to access guards. `birth` changes independently on every allocation,
so an observer detects stale authority even if a faulty transition forgets to
advance the architectural generation. The backing `epoch` is also ghost state:
it detects access completion after reuse and does not authorize operations.

| Transition | Effect |
|---|---|
| `mmap_one_page` | Linux/ABI reserves a VMA and supplies arena authority plus the sole modeled object. It advances the identity even at an old address; the PTE stays absent until a fault is resolved. |
| `reuse` | Models `malloc(64)` within an existing arena: allocate a fresh identity, preserve VMA and PTE rights, publish `p` or `q`. |
| `issue` | Check tag, bounds, capability rights, liveness/generation and PTE permissions, then create a pending access. An absent PTE records the capability and operation for retry. |
| `complete` | Finish the checked access; a store updates the retained frame. |
| `retire` | Break new authority. For `unmap`, also remove the VMA and PTE; previously checked accesses still target the old frame. |
| `invalidate(c,h)` | Clear hart `h`'s cached authorization and acknowledge invalidation for `c`. |
| `drain(c,h)` | Acknowledge only if hart `h` has no pending access to `c`. |
| `finish(c)` | Permit reuse only after both harts have invalidated and drained. |
| `resolve_fault`, `retry` | Install a PTE with the VMA's rights; recheck the original capability before issuing the retried access. A live retry must succeed; a retired pointer must fail. |
| `read_into_object` | Validate the requested writable span before consuming input; an invalid span leaves input untouched. |
| `clone_private` | Claim a virgin namespace instance, copy lifetime state and pointer meanings, and copy page contents to private backing. Dead identities remain dead. |

The retirement break is modeled as globally visible for its context before
`retire` returns. It blocks new accesses to that context while per-hart cache
invalidation is under way; an unrelated context may issue. This remains a
hardware/ABI contract to implement and price. `context_used` never resets:
reclaiming a process-context slot would need a separate instance-generation
protocol, which this model does not implement.

## Failure controls and limits

Controls omit selector/cache updates or drains, bypass PTE/liveness checks,
clamp an invalid syscall span, share private backing or revive a dead node.
Additional controls retain an allocation generation across `malloc` or remap,
reuse a used context, upgrade PTE rights in `malloc`, retain the invalidated
cache, ignore hart 1's pending access, or reject all fresh accesses/live retries.
The last two denial controls require positive behavior: refusing everything
cannot pass. Safety observers report violations after authorization; they
never grant or deny authority on behalf of a transition.

Review corrections close three defects in the original model: remap could
revive a generation, clone could reuse an old namespace, and allocator reuse
could add PTE write permission. The original schedules also stopped at reuse
and never placed a retiring context's pending access on hart 1. Their earlier
10,089-schedule result did not establish these properties. The revised gate
also rejects all four independently applied source mutations from that review:
no generation advance, an uncleared authorization cache, unconditional denial
of fresh generations and a drain that ignores hart 1. These mutation checks
are supplemental to the reproducible named controls in the record.

This is a pre-implementation counterexample search, not a proof of QEMU, RTL
or Linux. The state space has one node and one page per context, two issued
generations, one pending access per hart and fixed event multisets. Schedule
permutations are exhaustive **within those families**, not over all programs.
Page contents are integers. Encodings, tagged register save/restore, bounds
compression, multi-page accesses, linear moves, arbitrary capability transfer
between processes, kernel fault recovery, asynchronous I/O, TLB internals and
RTL timing are outside the model. `read_into_object` covers preflight and
input consumption, not concurrent kernel copy recovery. The next prototype
must exercise the omitted paths and the same old/fresh-pointer contract.
