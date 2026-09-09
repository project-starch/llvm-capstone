# Patches that are NOT applied

`fetch-mruby.sh` applies `patches/*.patch` and does not descend here. A patch in
this folder is one that is believed correct and is deliberately not in the build,
with the reason written down. Moving one up a level is the whole cost of adopting
it.

## `0002-envadjust-rederive-even-when-the-address-is-unchanged.patch`

`src/vm.c`'s `envadjust` opens with

```c
if (newbase == oldbase) return;
```

On a capability target the same ADDRESS is not the same CAPABILITY. When `realloc`
grows a block in place it returns the old address carrying the NEW, wider bounds,
while every `ci->stack` still holds the OLD, narrower one. The fast path leaves
those stale. The slow path is already correct -- it re-derives each pointer from
`newbase` -- so dropping the fast path is the whole fix, and it is a no-op when the
base really did not move.

**Held because it is unproven and this is not the bug being chased.** The
stage-3 fault reproduces identically with narrowing turned OFF
(`MRUBY_NO_NARROW=1`), where every capability carries the full arena and stale
bounds cannot matter. So `envadjust` is not the cause of that fault, and adding it
to the instrumented build would have meant two variables instead of one.

Also worth knowing: the CheriBSD purecap port in `xlang/cheri/mruby-port` runs
mruby without this patch. There `realloc` moves the block when it grows, so the
correct branch is taken and the defect stays latent. It is a real capability
portability defect that a different allocator happens to hide.
