> **ARCHIVED — pre-implementation note (superseded).** This is an early
> planning interpretation written *before* the C1 granularity work landed. Its
> central claims are now stale: it says there is "no normal automatic `SHRINK`
> on object materialization" and that "intra-domain object spatial safety is not
> achieved yet," but the compiler now narrows common global materializations
> (default on), two benchmark allocators, and stack objects (opt-in). The
> canonical current discussion is
> `design/granularity-provenance-discussion.md`; the measured/limited reality is
> in the `2026-06-29` granularity-provenance audit
> (`history/29-06-2026_15-08-22_granularity-provenance-audit.md`). Kept only for
> historical provenance of the reasoning.

# Capability Granularity, Provenance, and Next Steps

Below is my interpretation of `discussion-capability-granularity-provenance.md`.

## 1. Main conclusion

The key point is not that the current compiler is already fully safe. The more accurate conclusion is:

**Capstone provides the right architectural primitives, but the current compiler does not yet turn them into strong intra-domain C-level memory-safety guarantees.**

What seems to work well already:

1. **Integer-to-capability forging is blocked.**
   `ptr -> int -> ptr` goes through integer instructions, loses the tag, and should fault on dereference. This is good for the provenance story.

2. **Integers and capabilities are disjoint via hardware tags, not MACs.**
   The out-of-band tag is stronger than a checksum/MAC for in-memory safety. A MAC becomes relevant only for untagged storage: disk, network, persistence, swap, or untagged shared memory.

3. **Pointer arithmetic preserves provenance.**
   `p + i` is implemented as cursor movement from an existing capability, not as capability creation from an integer.

The major problem:

4. **Bounds are too wide.**
   Capabilities are derived from root capabilities such as `gp` or `sp`, but are not automatically narrowed to the object. There is no normal automatic `SHRINK` on object materialization.

5. **Therefore, intra-domain object spatial safety is not achieved yet.**
   For example, `char a[16]; p=&a[10]; *(p+25)` should trap if `a` has bounds `[a, a+16)`. Today it may succeed because the capability inherits broad `gp` bounds.

6. **Spilled capabilities are a real concern.**
   Spills preserve usable tagged capabilities. This is correct mechanically, but dangerous if in-domain code can read the spill slot using an overly broad capability.

So the current state is roughly:

```text
Provenance discipline: partially good already.
Capability granularity / bounds: main unresolved gap.
```

## 2. What to work on first

I would split the work into two tracks: paper evidence and compiler implementation.

### Priority 0: Make the current facts reproducible

Turn the existing probes into a permanent test/evidence suite:

```text
tests/capstone-authority/
  forge-inttoptr.c
  ptr-int-ptr-roundtrip.c
  pointer-diff.c
  global-oob.c
  heap-oob.c
  stack-oob.c
  spill-leak.c
  memcpy-capability.c
  union-punning.c
  va_arg-capability.c
  many-pointer-args.c
```

For each test, record:

```text
source code
generated assembly
expected runtime behavior
expected trap / no trap
```

This becomes both a regression suite and a paper artifact.

### Priority 1: Provenance audit and negative tests

This is probably the cheapest high-value step.

Define all legal authority constructors:

```text
root capabilities
ldc from tagged memory
cincoffset from valid capability
SHRINK from valid capability
delin / linearity-related operations
reviewed intrinsics
domain-entry mechanisms
```

Then verify that no other lowering path can produce a tagged capability.

This supports the second paper contribution:

> Every pointer capability must be derived from an existing pointer capability; integers cannot become authority.

### Priority 2: Implement the first granularity slice: globals

Start with globals because they are easier than stack and heap.

Example:

```c
char a[16];
char b[16];

char test(void) {
    char *p = &a[0];
    return p[20];   // currently may read adjacent global memory
}
```

Desired lowering:

```text
materialize &a
  cap  = cincoffset(gp, address(a))
  cap' = SHRINK(cap, address(a), address(a) + sizeof(a))
```

Expected result:

```text
Current compiler:
  global out-of-bounds access may succeed.

After global SHRINK:
  global out-of-bounds access traps.
```

This is the cleanest before/after demo for the paper.

### Priority 3: Add heap bounds via malloc

The current bump allocator returns un-narrowed pointers. A proper allocator should return a capability narrowed to the allocation:

```c
void *malloc(size_t n) {
    size_t rounded = round_to_representable(n);
    char *raw = arena + bump;
    bump += rounded;

    return cap_shrink(raw, raw, raw + rounded);
}
```

Then:

```c
char *a = malloc(16);
char *b = malloc(16);
a[20] = 'X';
```

should trap instead of writing into `b`.

### Priority 4: Stack bounds

Stack is harder because of ABI, spills, `alloca`, varargs, stack slots, and register allocation.

But stack bounds are important for the reviewer’s question about spilled capabilities.

Example:

```c
void f(char *input) {
    char buf[32];
    void *sensitive = get_sensitive_pointer();

    unsafe_copy(buf, input);
}
```

If `buf` has bounds only `[buf, buf+32)`, an overflow through `buf` cannot reach nearby stack slots or spilled capabilities.

## 3. What capability splitting means

The bad design is:

```text
Create one broad capability for the whole domain/global/stack memory,
then only move the cursor.
```

A capability is not just an address. It is an authority token:

```text
cap = { tag, base, cursor, end, permissions, ... }
```

So a pointer may have:

```text
cursor = &a[0]
base   = start_of_domain_memory
end    = end_of_domain_memory
```

It looks like a pointer to `a`, but it grants access to much more than `a`.

Example:

```c
char a[16];
char secret[16];

char leak(int i) {
    return a[i];
}
```

If `a`’s capability covers the whole global region, then `leak(20)` may read outside `a`.

Capability splitting means deriving narrower capabilities from the broad root:

```text
root/global cap:
  [global_start, global_end)

split into:
  cap(a):      [a, a + sizeof(a))
  cap(secret): [secret, secret + sizeof(secret))
  cap(heap1):  [heap1, heap1 + size1)
  cap(buf):    [buf, buf + sizeof(buf))
```

In practice, for the compiler this mostly means:

```text
When materializing a pointer to an object, emit SHRINK to narrow the bounds to that object.
```

## 4. Why “maximum / near-ideal granularity” matters

This is the principle of least authority applied to pointers.

Bad:

```text
pointer to a[0] grants access to the whole domain
```

Good:

```text
pointer to a[0] grants access only to a
```

“Maximum granularity” means each capability should be as narrow as possible while still allowing correct C programs to run.

It is “near-ideal” rather than “perfect” because of:

1. **Representable bounds constraints.**
   The architecture may not represent every exact byte range.

2. **C compatibility.**
   C has pointer arithmetic, casts, allocators, flexible array members, `memcpy`, and custom memory idioms.

3. **Subobject ambiguity.**
   For example:

```c
struct S {
    char buf[16];
    int len;
};
```

Should `&s.buf[0]` have bounds only for `buf`, or for the whole struct? The former is safer, the latter is more compatible.

For the first paper, I would not promise perfect subobject bounds. A better scoped claim is:

> We provide near-ideal object-level granularity for globals, heap allocations, and stack objects, modulo representability constraints and explicitly stated subobject limitations.

## 5. Suggested paper contributions

### Contribution 1: Capability granularity for C object safety

Possible formulation:

> We design and implement compiler-driven capability materialization that assigns near-minimal bounds to C objects — globals, heap allocations, and stack objects — using Capstone’s monotonic derivation and bounds-narrowing primitives. We prove/check that compiler-generated capabilities are no wider than the object they name, modulo representability rounding.

### Contribution 2: Capability provenance preservation

Possible formulation:

> We define and enforce a lowering invariant: a tagged capability can only be produced by derivation from an existing tagged capability or by trusted root/domain entry mechanisms; scalar data and integer computations cannot create authority.

These two contributions complement each other:

```text
Provenance answers: where did the authority come from?
Granularity answers: how much authority does it carry?
```

## 6. Most important takeaway

Without capability splitting, the security story is:

```text
An attacker cannot forge a capability from an integer.
But once the attacker has any in-domain capability,
that capability may be broad enough to reach too much memory.
```

With capability splitting, the story becomes:

```text
An attacker cannot forge a capability from an integer.
And normal C pointers grant access only to the object/allocation/stack slot
they are supposed to name.
```

This is the transition from “the Capstone backend can compile benchmarks” to “the Capstone compiler provides meaningful C-level memory-safety guarantees.”

My recommended immediate plan:

```text
1. Build the provenance negative-test suite.
2. Implement global-object SHRINK as the first end-to-end granularity slice.
3. Add heap SHRINK in malloc.
4. Add stack bounds.
5. Add a checker/verifier for legal capability-authority construction.
```
