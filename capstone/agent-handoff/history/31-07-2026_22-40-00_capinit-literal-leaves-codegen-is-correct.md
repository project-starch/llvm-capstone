# cap-init literal leaves: the codegen is correct (static proof), and two mechanisms are refuted

Date: 2026-07-31 (late)
Context: localising the stage-51 LIVELOCK (`rc=0xB1`, the domain RETURNED, so the core was
never hung). Stage 52 = `0xC1`: `lit[1]` is the first literal whose `strlen` walk overruns.
Stage 53 = `0xDF`: `lit[0]`'s first 8 bytes are `l t r i m \0 r t` — correct for a MERGED
container, so `lit[0]` is fine and its data is intact.

The open question was whether `lit[1]`'s POINTER or its DATA is wrong. The board went
unreachable before the probes could run, so this was answered **statically instead**, from
the linked domain ELF and the RTL. That is a stronger answer than the board would have
given for the codegen half, because it is a proof rather than a measurement.

## What was examined

`capstone/caplifive-system/sw/buildroot/build/target/test-domains/wd55.dom`, function
`__capstone_cap_init` (2718 instructions), disassembled with `llvm-objdump`.

## Result 1 — the merged container and the expected deltas

`"ltrim\0rtrim\0trim\0..."` lives in `.rodata` at vaddr `0x16e52e`. Laying the 16 literals
out end-to-end gives `&"ltrim"=0x16e52e`, `&"rtrim"=0x16e534`, `&"trim"=0x16e53a`, so the
correct deltas are **`lit[1]-lit[0] = 6`** and **`lit[2]-lit[1] = 6`**.

## Result 2 — the emitted code computes exactly those deltas

The 16 literal capabilities are derived at `0x14d6d8`+:

    14d6d8  cincoffsetimm s1, a0, 0x6da     <- lit[0]
    14d6e0  cincoffsetimm s0, a0, 0x6e0     <- lit[1]
    14d6e8  cincoffsetimm t6, a0, 0x6e6     <- lit[2]

`0x6e0-0x6da = 6` and `0x6e6-0x6e0 = 6`, all three from the same base `a0`. **The offsets
are right.**

## Result 3 — the same 16 registers are stored into THREE arrays, and stay live

There are three `lit` arrays in the TU (stage 51's, stage 52/53's `run_sqlite_staged.lit`,
stage 54-56's `run_sqlite_staged.lit.863`; both 256 B = 16 x 16-byte capabilities). They are
written at `0x14d6dc` (interleaved with the derivations), `0x14eef8` and `0x14ef40` — all
three storing the SAME 16 registers.

Between the derivation and the last store there are **1544 instructions, zero calls, zero
branches**. Only `a0` is redefined (539 times), and the compiler handles it correctly:
`lit[15]` is spilled to `0x260(sp)` at `0x14d758` and reloaded at `0x14e800`. The other 15
capabilities are never touched.

**So all three arrays receive correct pointers. The codegen is not the bug.**

## Result 4 — two plausible mechanisms REFUTED against the RTL

Both were checked in `capstone-ariane/core/anvil_build/`, not assumed:

* **`cincoffset` does NOT consume its source.** `capstone_flu_unit.anvil:43` and `:62` both
  return `create_result_pack(..., rs1, rd)` with `rs1 = data.cap_rs1` unchanged. So deriving
  12 pointers from one live `a0` is safe, and the "C-14 shape but for cincoffset" theory —
  that `lit[0]` survives and everything after it comes from `cnull` — is **wrong**, even
  though it fits the observed `lit[0]`-good/`lit[1]`-bad signature perfectly.
* **`STC` does NOT clear its source register** for LINEAR/NONLIN. `capstone_dyn_unit.anvil:427`
  returns `rs2_v` unchanged. Only the UNINIT path nulls it (`rcnull`, ~`:410`). So storing
  the 16 capabilities three times does not destroy them. (The documented "linear clearing"
  is on **LDC**, clearing the *memory* source — it is not symmetric on STC.)

Recording the refutations explicitly because both were good-looking theories that matched
the symptom, and both would have been asserted as the root cause without the RTL check.

## What this means for the pending probes

`wd54/55/56` are built and staged. Their expected values are now **proved, not assumed**:
`55 -> 6`, `56 -> 6`, `54 -> 0xDF`. That makes them a clean discriminator:

* deltas come back **6** => pointers correct on silicon too; the fault is in the WALK
  (the `strlen` loop / its `lcc` epilogue), not in cap-init.
* a delta comes back **wrong** => silicon EXECUTION diverges from provably-correct codegen,
  which is a hardware finding, not a compiler one.

Either outcome is informative, and neither can any longer be blamed on the emitted offsets.

## Incidental: an asymmetry worth reporting (NOT this bug)

`CINCOFFSET`'s operand type-check is commented out behind a `FIXME`
(`capstone_flu_unit.anvil:31-33`), while `CINCOFFSETIMM` keeps it (`:49-51`). So a
register-form `cincoffset` with a `NOT_CAP` rs1 proceeds silently where the immediate form
raises `UNEXPECTED_OPERAND`. `lit[3]`/`lit[4]` do use the register form. Not implicated
here, since `lit[1]` uses the immediate form and its base is shared with the working
`lit[0]`.

## Board status at time of writing

Unreachable. DNS resolves and TCP :443 connects instantly, but the **TLS handshake times
out** (15 s) — the console tunnel is up with its backend not answering. Three consecutive
runner attempts failed at `connect()`. Nothing was flashed; the firmware built and passed
its freshness check (initramfs 10,495,488 bytes, verified by decompressed content).
