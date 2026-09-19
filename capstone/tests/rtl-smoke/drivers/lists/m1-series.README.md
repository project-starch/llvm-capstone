# The M1 reclaiming lists, and which image each one needs

An M1 list is only half of a run: the other half is the image, because `M1_MAXRET` is a build-time
define and three of the four Design arms are bounded by it rather than by `--cap`. A list run against
the wrong image measures the buffer and reports a clean-looking `stop=buffer`.

| list | arms | image it needs | why |
|---|---|---|---|
| `m1-pilot.txt` | `drop` ×3, rising `C` | any | the platform pilot; `C` small enough that the buffer never binds |
| `m1-series.txt` | all four at `C` = 65,532 | none that exists | see below — `release` is unreachable at this `C` |
| `m1-series-nopressure.txt` | `drop`, `ring`, `release` at `C` = 65,532 | `1b7a04fe237e1580` (`M1_MAXRET`=4096) | drove boot 2 on 2026-09-18; `drop`/`ring` complete, `release` truncates |
| `m1-order9-ctl.txt` | `drop`, `ring` at `C` = 65,532 | `c685b0d7a95492ef` (`M1_MAXRET`=4096, `R1_STACK`=1 MiB) | the allocation-order control — see below |
| `m1-maxret43296.txt` | `pressure`, then `release` at `C` = 4,329 | `249cfda958f22f16` (`M1_MAXRET`=43,296) | the largest buffer that stays at allocation order 8 |
| `m1-relbuf43296.txt` | `release` at `C` = 65,532 | `29d9099326304ee5` (same, `M1_RELEASE_AT_BUFFER=1`) | the re-specified trigger — see below |

## Which arms are bounded by `M1_MAXRET`, measured rather than assumed

In `run_m1`, an arm that is neither `drop` nor `ring` takes the retain-every-alias branch, and the
`release` arm's phase 2 is gated on `alloc >= target`. So:

- `drop` retains nothing and `ring` retains 16 — **neither needs the buffer**, at any `C`;
- `pressure` always stops at `alloc = M1_MAXRET + 1`, whatever `C` is. Its result is the **fraction of
  distinct indices covered**, `M1_MAXRET / 65,532`, not an allocation count;
- `release` needs `M1_MAXRET >= 10C` to reach phase 2 at all. At `C` = 65,532 that is 655,320
  capabilities — **10.5 MB**, since a `void *` here is a 16-byte capability — against a ~2 MB region.
  **`m1-series.txt` therefore cannot be satisfied by any image on this platform**, and is kept only
  because it is what boot 1 was scheduled from. Use `--cap M1_MAXRET/10` for a release arm.

A smoke run at `C` = 64 hides all of this: 4,096 covers 10C with margin there and every arm reads
`stop=target`. That is why the emulator pass did not catch it.

## The allocation-order control

`domdata-budget.py`'s model, verified against two built images, is `declared = 278,064 + 16 x M1_MAXRET`
for this harness, and the module rounds `code_len + 8,192 + declared` up to a power-of-two page count.
So `M1_MAXRET` moves two things at once: the **storage carve** the entry glue walks, and the
**allocation order** of the region the module hands the domain. 4,096 lands at order 7 and 65,532 at
order 9, with order 8 never built.

`m1-order9-ctl.txt` separates them. Its image keeps `M1_MAXRET` = 4,096 — the pilot's storage carve,
76,832 bytes, byte-identical arithmetic — and reaches order 9 by padding `R1_STACK` to 1 MiB alone.
Boot 1 wedged at `mcause` 28 (`RISCV_EXCP_CAP_OOB`) with `mepc` at `_start+0xec`, the entry glue's
byte-wise zeroing store, so the region and the carve are both live suspects and this image tells them
apart: wedging indicts the region size, completing indicts the carve.

## The two release arms, and why there are two

The lead ruled on 2026-09-18 that both be run, because they answer different questions.

`m1-maxret43296.txt` keeps the **approved algorithm** and reduces the capacity until phase 2 is
reachable: `C` = 4,329, so 10C = 43,290 fits inside the 43,296 buffer with six to spare. It is a real
release arm at **6.6 % of the protocol's production capacity**, and that reduction is the thing to state
whenever the number is quoted.

`m1-relbuf43296.txt` keeps `C` at the **production capacity** and moves the trigger instead:
`M1_RELEASE_AT_BUFFER=1` starts phase 2 when the retained buffer fills rather than at 10C. It is *not*
the protocol's release arm and a transcript from it must never be reported as one — which is why the
define lowers `target` rather than hiding the change, so the start line reports the trigger it used and
carries `rel_at_buffer=1`.

The two images differ by **exactly** that define: same `M1_MAXRET`, same allocation order 8, storage
704,032 against 704,048 (the sixteen bytes are the extra start-line field). And the define is opt-in and
free: with it present but off, `M1_MAXRET=4096` rebuilds to `1b7a04fe237e1580` byte for byte — the image
currently on the board — and `M1_MAXRET=43296` is unchanged. That was checked rather than assumed, and
the check earned itself: the first version of the change put the extra term in the loop's phase-2
condition, where it folds away at compile time, and the default image still moved (`598dce77`).

## The stale-take probe: which arms can carry it, and how to read it

`--stale-take` is M1's instrument for **condition 3** — that a stale reference never regains authority.
It runs last in `run_m1` and performs a capability operation through `oldest`, the oldest retained alias.

**It is guarded by `if (stale_take && oldest)`, and `oldest` is null in two of the four arms.** Read from
the source rather than assumed:

| arm | `oldest` | probe |
|---|---|---|
| `drop` | never assigned — stays 0 | **silently does nothing** |
| `ring` | the ring entry about to be overwritten, ~16 allocations old | fires |
| `pressure` | `m1_ret_alias[0]`, the **first** alias retained — maximally stale | fires |
| `release` | set during phase 1, then **cleared to 0 when phase 2 begins**, and the arm ends there | **silently does nothing** |

So `--stale-take` on `drop` or on `release` emits no `go` line, no `returned` line, and nothing else:
**the transcript is indistinguishable from a normal completion**, and condition 3 would be recorded as
exercised while nothing was exercised. Put the probe on `pressure`, where the alias is the oldest one
the run ever held.

**Read it three ways, not two.** The probe is expected to fault, and a faulting domain wedges, so
"it hung" is the *success* path — which means a wedge from any other cause reads as a pass unless the
cause is checked:

- **no `R1 m1 stale-take go` line** → the probe never armed. An instrument failure, not a result.
- **`go`, then a wedge whose latched `mcause` is 25** (`RISCV_EXCP_INVALID_CAP`,
  `capstone-qemu/target/riscv/cpu_bits.h:694`) → authority was refused. **Condition 3 holds**, and the
  cause says so rather than leaving it to inference.
- **`go`, then `R1 m1 stale-take returned nonzero=1`** → the operation succeeded through a stale
  reference. **Condition 3 is violated**, and that is the finding the arm exists to look for.

A wedge with `go` present but some *other* `mcause` is none of the three: it is a different fault that
happened to land at the probe, and it settles nothing.

**Cost.** The probe wedges the core, so it takes the rest of its boot with it — including the closing
`k800` control. It therefore goes in the LAST invocation of a boot, and that boot gives up its closing
control by design. That is a deliberate trade, not an accident to be explained afterwards.

## CORRECTION (2026-09-19): the `--stale-take` probe as written does NOT test condition 3

The section above described how to read `--stale-take`. It was run on silicon on 2026-09-19 and the
result was **uninterpretable**, for a reason the section did not anticipate. Recorded here rather than
edited away, because the boot was spent.

**What happened.** The probe faulted with `mcause` 26 (`UNEXP_CAP_TYPE`), not the 25 predicted, and the
domain emitted **no R1 output at all** — not even its start line. The absent output is not evidence the
probe never armed: **a wedged domain never has its output buffer read back, so a wedge destroys the
whole transcript.** The `mepc` is what showed the probe had in fact run — it resolves to `mrev` inside
`run_m1`, the mint in `sublet_take`.

**Why it proves nothing.** `sublet_take` mints a revocation with `mrev`, and `mrev` requires
`CAP_TYPE_LIN`. An alias is `NONLIN`. So the probe asks for an operation that is a **type error on any
alias**, stale or live. Confirmed on the emulator with the `M1_STALE_TAKE_LIVE=1` control: the stale
alias fails `Assertion rs1_v->tag` and a **live** alias fails `Assertion type == CAP_TYPE_LIN`, one line
apart in `op_helper.c`. Both fault. The mint separates nothing, and a detector that fires on both
hypotheses is not a detector.

## The probe that DOES test condition 3, and its control

**Dereference the alias.** A live alias performs an ordinary load through an ordinary `NONLIN` alias, so
a fault can only be the stale reference being denied. Build with `-DM1_STALE_DEREF=1`; the probe then
dereferences first and prints `R1 m1 stale-deref ok`, and only afterwards attempts the (known useless)
mint, so the mint can never pre-empt it.

**Its control is mandatory, not optional.** `-DM1_STALE_TAKE_LIVE=1` points the same probe at
`alias[0]`, a live alias, through the identical path. Without it, a fault is uninterpretable — which is
exactly what the 2026-09-19 boot paid to learn.

Measured on the emulator, 2026-09-19:

| operand | image | reading |
|---|---|---|
| **live** (control) | `aed492ab985653f3` | dereference **succeeds**; execution continues to the mint, which then fails the `CAP_TYPE_LIN` assertion |
| **stale** (probe) | `feacd8a621cce681` | `Cap mem access requires capability` → `domain halted by capability fault: cause = 24` (`UNEXP_OP_TYPE`); the operand is a raw scalar carrying no tag |

So a stale reference is **disarmed at use**, as the approved specification says, and a live one is not.
**This is an emulator result and not a silicon one** — the board has so far run only the void mint
probe. Both images carry emulator pass records, earned on an invocation that returns, since a
deliberately-faulting one never can.

**Reading a board run of the deref probe.** A wedge destroys the transcript, so read `mepc` and
`mcause`, not the absent prints. `mepc` inside the dereference with a cap-access fault = condition 3
holds; `mepc` at the later `mrev` means the dereference SUCCEEDED, which is condition 3 **violated** and
is the finding to look for. The control boot must show the live alias reaching the `mrev`.

**Neither probe image may be used for a measurement arm.** Both differ from the arm images by the probe
body, which `run_m1` links unconditionally, so their hashes differ. `M1_STALE_DEREF` is off by default
precisely so the measured arms keep the binaries their records cite: with it off, `M1_MAXRET=4096`
rebuilds to `1b7a04fe237e1580` and `M1_MAXRET=43296` to `249cfda958f22f16`, both byte for byte.
