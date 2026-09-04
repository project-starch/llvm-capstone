# S-12: the narrowed-fence experiment cannot work on this core, and what the fence result does mean

## The experiment, and why it is VOID as a discriminator

The plan was to narrow the barrier: `fence w, w` orders stores only and drains the store buffer,
`fence r, r` orders loads only and does not. If the store-only fence cured S-12 and the load-only
one did not, the store path — the mechanism's own stated requirement — would be implicated.

Both cured it. `fence r, r` completed 3/3, `fence w, w` completed 1/1, against a baseline that
traps 3/3.

**That inference is invalid, because the two arms are the same instruction on this core.**
`core/decoder.sv:464-466`:

    // FENCE
    // Currently implemented as a whole DCache flush boldly ignoring other things
    3'b000: instruction_o.op = ariane_pkg::FENCE;

The pred/succ fields are not read. `fence w,w` and `fence r,r` both decode to `ariane_pkg::FENCE`
and both perform a whole D-cache flush. The arms differ in four bits of an immediate the hardware
discards. This is a check that fires correctly and still under-determines: it would have supported
a conclusion about store-versus-load ordering from an experiment that never varied it.

## What the fence result DOES establish

Combining tonight's draws with the folder's existing NOP control:

| inserted at the identical point | semantics | result |
|---|---|---|
| `addi x0, x0, 0` (NOP) | inert | **wedged 4 / 4** |
| `fence rw,rw` | whole D-cache flush | 0 wedges / 7, and 4/4 completing tonight |
| `fence w,w` | same op after decode | completed 1/1 |
| `fence r,r` | same op after decode | completed 3/3 |

The NOP build has a **byte-identical symbol table** to the fence build, so layout, displacement and
all addresses are identical. Therefore:

* the cure is **semantic, not positional** — an inert instruction at the same point does not cure;
* the cure is **not mere pipeline delay** — the NOP supplies delay and does not cure;
* on this core the fence's semantics are **a whole D-cache flush**, so the operative effect is in
  the MEMORY SYSTEM, not in instruction scheduling.

That is a real constraint on the mechanism and it survives the invalidated arm.

## Where that leaves the root cause

Assembled account, each leg independently supported:

1. **Necessary condition** — the store's source, the load's destination and the faulting
   instruction's operand must be the same architectural register. One byte, 2x2 dissociation,
   two independent matched pairs.
2. **The cure is a memory-system flush**, not layout and not delay (above).
3. **The RTL structurally permits it** — `decoder.sv:1313` decodes STC's `rd := rs2`, making the
   store a real scoreboard producer; `create_cnull()` is cursor 0 AND `cap_type` 0, which is
   `mcause 25` with `tval 0` exactly and is what separates this from S-07/A-1's real-cursor
   signature.

**Still not isolated.** A D-cache flush drains the store buffer, empties the pipeline and changes
timing all at once; "semantic rather than inert" narrows the cause to the memory system but does not
pick out the forwarding path specifically. Nothing measured tonight distinguishes the
scoreboard-producer account from any other memory-system-dependent one.

## The cheap discriminators are now exhausted

* **The synthesised S-07 detector cannot be read.** On a live core after a returned fault, `sw=204`
  and `sw=208` returned the SAME byte (`0xbe`) — cross-aperture contamination — and the driver's own
  guard rejected it: *"count>0 with ldc_seen clear cannot be produced by the design; the readout is
  wrong, NOT the core."* Displacement is therefore neither confirmed nor excluded by measurement.
* **The narrowed fence cannot discriminate**, per the decode above.
* **The NOP-vs-fence contrast is already spent** — it is what established "semantic, not layout".

What remains is a simulation that actually fires, which nothing has achieved in ~53,000 in-domain
executions across eight eliminated variables, or a bitstream carrying a recorder on the operand mux
— roughly 90 minutes plus a reflash, ask-first, and worth doing only with the predicted reading
written down first.

## A resource limit worth knowing, because it fails as a false negative

`SPLB` caps the region budget at about **two large domains per boot, and the 1.6 MB control is one
of them** — so exactly ONE large SQLite arm fits per boot. A third large domain silently receives no
regions, reports "no return", and reads exactly like a fault. A four-arm boot lost `fence r,r` this
way and a three-arm boot lost the baseline; neither was a result. Small ladder rungs (~10 KB) batch
four at a time without trouble, which is what made the mistake easy.
