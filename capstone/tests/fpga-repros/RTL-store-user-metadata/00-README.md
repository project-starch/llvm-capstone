# Every store routes capability metadata into the dcache write-user sideband

**Status: code-level RTL observation, NOT a demonstrated software-visible defect.**
The repro here tests the invariant most obviously at risk and finds the silicon **correct**.
Committed as a regression test and as the evidence trail behind the observation.

## The RTL observation

In `capstone-ariane` (branch `fpga-testing`), the capability-metadata shadow of a store's
**data** register is routed into the dcache write-user sideband for **every** store, not only
for capability stores (`stc`):

| file | what it shows |
|---|---|
| `core/load_store_unit.sv:1003-1020` | builds `lsu_ctrl.user` from `fu_data_i.cap_data.cap_metadata_b` unconditionally for every LOAD/STORE-fu instruction — not gated on `operation == STC` |
| `core/store_unit.sv:344-346` | `st_user_n = lsu_ctrl.user;` — ungated |
| `core/store_buffer.sv:172-176` | `req_port_o.data_wuser = commit_queue_q[...].user;` — ungated |
| `core/cache_subsystem/wt_dcache.sv:70-79` | the write buffer tracks `data` **per byte** (`dirty`/`valid` bitmasks) but `user` as **one flat field with no per-byte mask** |

`core/ex_stage.sv:771` confirms `cap_metadata_b` is the shadow-register-file entry for
`operand_b`, i.e. the store's data register.

Bit-position note: in `cap_metadata_t` (`core/include/ariane_pkg.sv:609-637`) the packed layout
puts `bounds` at bits [27:0], so `bounds.cursorless` is bit 27 — which is why `0x08000000`
looked suggestive. That turned out to be unrelated (see "What this is not").

## What the repro tests, and the result

If a plain integer store's `user` field can carry a foreign capability metadata word into a
location, the invariant most likely to break is: **an integer store over a stored capability
must destroy that capability.** The pair tests exactly that.

    tagr   store a capability to a 16-byte slot, read its type with lcc, then overwrite the
           low half with a plain 64-bit integer store -- and RETURN without re-reading.
           CONTROL: proves the domain reaches the overwrite and returns.

    tagf   identical, but ALSO reloads the slot as a capability and lcc's it afterwards.

Architectural reference (QEMU): `tagr` returns **1017**; `tagf` **aborts** in
`helper_cslcc` on `rs1_v->tag` — the integer store clears the tag, so the second `lcc` is an
error. Correct silicon behaviour is therefore a **trap (no result)**; a returned value would
mean the tag survived an integer overwrite.

**Board result** (`caplifive_fixed_forward.bit`, instrumentation mode 0, `tagr` first):

    tagr    1017    OK      (control returned -- the overwrite path executes)
    tagf    none            (the lcc faulted -- tag correctly cleared)

**Silicon is correct.** No tag survival, no metadata forgery through this path.

## What this is not

* Not a demonstration that the `user`-sideband routing is exploitable or even observable.
  The RTL analysis explicitly could **not** trace a path from `data_wuser` into a plain
  `lw`'s returned data — `core/load_unit.sv:773-808` builds `result_o` for `LW`/`LWU` purely
  from `shifted_data` (i.e. `data_rdata`), never from `data_ruser`. That remains UNRESOLVED
  and would need the `wt_dcache_mem.sv` fill/writeback merge path.
* Not the cause of any blocker we have. It was suspected of explaining a `0x08000000`
  corruption in a ladder rung; that corruption was subsequently traced to the **ladder
  measurement harness** (`ladder_perf_domain.h`, instrumentation mode 4), which the SQLite
  domain does not use at all.

## Reproducing

    source capstone/tests/capstone-test-env.sh
    bash capstone/tests/fpga-repros/RTL-store-user-metadata/run.sh

Build the rungs at instrumentation **mode 0** (`DOMAIN_EXTRA_CFLAGS=-DLADDER_NO_MINSTRET=1`).
Mode 4 perturbs codegen enough to produce deterministic silicon miscomputes of its own, so a
defect repro must not carry it. Run `tagr` FIRST: without it, `tagf`'s no-result is
indistinguishable from the domain wedging earlier.

Note both kernels carry a dead `..._pad[512]`: without the instrumentation the image comes to
exactly 0x1000 bytes, which makes the monitor's `create_domain` SPLIT degenerate (QEMU asserts
in `helper_cssplit`). The padding keeps the image above 0x1000.
