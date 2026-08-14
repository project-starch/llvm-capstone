# TOCTOU / double-fetch at a sharing boundary

*Class 6 of `agent-handoff/design/sharing-bug-taxonomy-and-novelty.md` — previously the one
class with zero evidence. Triaged from a web sweep 2026-08-14. Nothing built yet.*

## The framing result: callback-expressibility is a property of the BOUNDARY

This is the finding that makes the class tractable for us, and it is worth stating in the
paper:

> **Kernel↔user and hypervisor↔guest boundaries are crossed by *traps*, so the privileged side
> never calls the attacker's code — every one of those bugs needs a second core. Sandbox↔host
> boundaries are crossed by *calls*, so the privileged side routinely re-enters attacker code
> between check and use.**

So the cheap, single-hart, deterministic reproducers all come from **JS engines, wasm
embedders and Node/Deno** — not from the kernel corpus. Our domain model has no concurrency,
and it does not need any.

Corollary for the mechanism argument: **Capstone does not detect the race, it removes the
precondition.** Linearity enforces *"the lender gave up write authority for the duration of the
loan"*, which is violated identically whether the second write comes from a racing core or from
a callback. A race detector needs concurrency; a structural prevention does not.

## Tier 1 — verified, callback-expressible, no threads

| # | Specimen | Mechanism | Severity |
|---|---|---|---|
| **1** | **wasmtime GHSA-2hw9-mc66-jc2q** · RUSTSEC-2026-0223 · *no CVE* | Bulk `memory.copy` pointers **not recomputed between preemption points**; the embedder grows linear memory during an epoch callback, the base moves, the copy resumes through stale pointers. Siblings in the same advisory: cancelled `table.grow` leaves non-nullable slots uninitialised; `array.copy` holds raw GC-heap pointers across preemption → *"corruption of the GC heap"* | 46.0.0–46.0.1, 47.0.0–47.0.2 |
| **2** | **CVE-2026-25641** `@nyariv/sandboxjs` | Purest form: keys typed as strings, never coerced. `hasOwnProperty(key)` calls `toString()` → `"x"` (passes denylist); the property access calls it **again** → `"__proto__"`. No memory model involved at all | **CVSS 10.0** |
| **3** | **CVE-2023-28445** Deno | *"Resizable ArrayBuffers passed to asynchronous native functions that are shrunk during the asynchronous operation could result in an out-of-bound read/write."* `V8Slice::open` sized from the length at call time, never revalidated. **The `await` point is the callback** | **CVSS 9.9** |
| **4** | **Ladybird GHSA-w89h-j2xg-c457** · *no CVE* | `m_data` *"is cached at TypedArray construction and is never recomputed during the object's lifetime"*; `WebAssembly.Memory.grow()` on a **shared** memory reallocates the backing buffer, and the spec requires skipping `detach_buffer()` for shared buffers, so the cached-view walk never runs | **CVSS 9.6** |
| **5** | **CVE-2024-21896** Node.js | Permission model calls `path.resolve()`, then `Buffer.from()` — and `Buffer.prototype.utf8Write` is monkey-patchable. Fetch #1 = the string that passed the check; fetch #2 = whatever the attacker's encoder emitted. **Patch and PoC in one commit** | High |

**Build #1 as the flagship.** Production runtime, two domains, one shared buffer,
single-threaded, deterministic, 2026 — and the failure mode (a cached base surviving a
relocation) is precisely what a capability catches: a copy carrying a capability faults on
resume instead of writing through a stale base. **Pair it with #2** as a ten-line conceptual
version that makes the idea unambiguous before the memory-safety one.

## Prior art that says "we could not solve this" — quote these

* **V8 `IterateElements`: one defect, three CVEs, five years.** CVE-2016-1646 (prototype
  getter — in CISA KEV), CVE-2017-5030 (Proxy `defineProperty` via `Symbol.species`),
  CVE-2021-21225 (`valueOf` during `ToNumber`). `int fast_length` cached, loop calls back into
  JS, callback shrinks `length` and GCs the backing store. **The durable fix is the quotable
  part** — after years of enumerating vectors V8 gave up and forbade re-entry outright:
  `DisallowJavascriptExecution no_js(isolate); // Disallow execution so the cached elements
  won't change mid execution.`
* **RLBox, "Retrofitting Fine Grain Isolation in the Firefox Renderer", USENIX Security 2020**
  (arxiv.org/abs/2003.00572): *"We could address this by copying `output_scanline` to a local
  variable… But, it's not always this easy — in our port of Firefox we found numerous instances
  of multiple reads, interspersed with writes, spread across different functions. Using local
  variables quickly became intractable."* Their answer is a `freeze()`/`unfreeze()` API — **the
  thing a capability gives you for free.**
* **wasmtime's own `Memory` docs** state it as a design hazard: *"even holding a raw pointer to
  memory over a wasm function call is also incorrect."* **WAMR ships an unfixed, documented one**
  in `wasm_export.h`: *"The validation result, especially the NUL termination check, is not
  reliable for a module instance with multiple threads."*

## Kernel specimens with no race at all

* **CVE-2014-0206** (aio ring `head`) and **CVE-2024-41009** (BPF ringbuf `consumer_pos`) — the
  kernel simply trusts a user-writable field in a `MAP_SHARED` page. Write the value, then make
  the syscall. ~30 lines each, zero concurrency.
* **io_uring has a real SQE double-fetch history and *zero* CVEs.** Verified verbatim:
  `56080b02ed6e` *"don't re-read sqe->off in timeout_prep() — SQEs are user writable"* (8 lines,
  the cleanest minimal specimen anywhere); `9c280f908711` *"Don't re-read userspace-shared
  sqe->flags, **it can be exploited**"*; `6f7a644eb7db` on ring heads read twice. NVD keyword
  search returns 0 results, with `"double fetch"` → 21 as a positive control. **The commonly
  cited io_uring CVEs are all something else** (CVE-2023-2598, CVE-2022-29582, CVE-2021-3491,
  CVE-2023-21400).

## Build the ARCHITECTURAL shape, not the compiler-induced one

Xen's XSA-155/166/197/478 are **compiler-induced**: the source reads once and the codegen reads
twice, so they manifest only at certain `-O` levels (XSA-166's bitfield case probably never
miscompiles). The JS-engine and wasm items are **architectural**: the source itself reads twice,
so they reproduce at any optimisation level.

**A compiler-shape reproducer is a codegen lottery** — a clean run would tell you nothing about
the subject and something about your build flags. Build architectural only.

If one true-concurrency specimen is wanted for honesty, **XSA-478 `unserialize_data`** is a
15-line function that reimplements 1:1 over `shm_open` and two processes.

## ⚠ Verification cautions — read before citing anything

* **Four CVE IDs that surfaced in search summaries are FABRICATED**, all claimed as QEMU virtio
  double fetches: **`CVE-2026-50624`, `CVE-2026-50626`, `CVE-2026-61476`, `CVE-2026-63321`**.
  MITRE returns `CVE_RECORD_DNE` for all four, with positive controls confirming the API does
  serve 2026-range records. **Do not let these reach a citation list.**
* **CVE-2025-14325** (SpiderMonkey resizable-SAB): NVD says only *"JIT miscompilation."* The
  specific `valueOf`→`sab.grow()` mechanism is third-party analysis. Mapping **UNVERIFIED**.
* **CVE-2016-4622** (JSC `Array.prototype.slice`): NVD gives no mechanism; the `valueOf`
  account is from Phrack 70. Well known, almost certainly right, **not primary**.
* **WAMR's WASI `iovec` loop** (`libc_wasi_wrapper.c`) loads `buf_offset`/`buf_len` for
  `validate_app_addr()` and **again** for the native conversion. Source-confirmed at two release
  tags, **but unreported — no CVE, no build, no race attempted.** Do not cite as a known bug; it
  is fine as a from-source specimen.
* **Refuted, so they do not contaminate the set:** vDSO/`vvar` (no `VM_MAYWRITE`); the eBPF
  verifier (operates on bytecode, never map contents); AF_XDP `xskq_cons_read_desc` (copies to a
  kernel local *first* — correct by construction); the perf mmap ring (`READ_ONCE`);
  `BPF_MAP_TYPE_USER_RINGBUF` (**cite as the reference-correct implementation**); all 508 Xen
  XSAs for hypercall arguments (`copy_from_guest()` *is* the snapshot pattern); and Hyper-V,
  where the real finding is inverted — Linux guests hardening *against* an untrusted host
  (`adae1e931acd8b`: *"return a copy… In this way, the packet can no longer be modified by the
  host"*).

## Next

1. Build specimen #1 (wasmtime bulk-copy shape) as a ~40-line two-domain reproducer: privileged
   domain caches `base`, copies in chunks, calls a hook every K bytes; the hook re-enters the
   guest which grows/reallocs; the copy resumes through the stale base.
2. Matched control differing by exactly one thing: the hook does not grow.
3. Capstone column — the copy carries a capability; expect a fault on resume.
4. CHERI column, three configs — expect 0/N in all three, since nothing is freed.
