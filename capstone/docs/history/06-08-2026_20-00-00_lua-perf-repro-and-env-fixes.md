# Capstone perf reproduced + two environment bugs fixed; real-Lua path characterised

**One line.** The report's Capstone temporal-safety perf number reproduces on the
current QEMU (`revoke − norevoke = +10 instr/op, O(1)`); getting there uncovered two
environment bugs and a submodule regression. The *real-Lua* perf/conformance upgrade
stays blocked on a gp-captable-init QEMU-strictness issue.

## Reproduced result (cjalr path, the report's method)

`revoke_cost_tree` (2000-key BST, 2 rounds = 4000 node lifecycles), `-icount shift=0`:

| config | per-op | report (`15-07-2026_..._cheri-capstone-perf-comparison.md`) |
|---|---:|---:|
| norevoke | 96,129 instr | 96,202 |
| revoke   | 96,139 instr | 96,212 |
| **revoke − norevoke** | **+10.00 instr/op, O(1)** | **+10** |

Exact match on the O(1) revoke-at-free cost. (Alloc-side inflation is the Phase-0
allocator's O(n) `rof_find`, not the mechanism — the delta is what matters.)

## How to reproduce

```bash
# The ONLY real prerequisite is a correct CAPSTONE_REPO_ROOT (see bug 1).
CAPSTONE_REPO_ROOT=<repo> bash capstone/tests/runtime-qemu/run-tree-cost-probe.sh
```
This builds bump/norevoke/revoke domains (cjalr, `my_first_domain/start.S`) and the
host, runs each under `-icount`, and prints the per-op counts + the revoke delta.
No module swap and no gp-captable are needed — this is the working path.

## The two environment bugs (these caused ~all failed runs this session)

1. **`CAPSTONE_REPO_ROOT` was pre-set to `/home/capstone`** in the shell (wrong for a
   checkout at `<repo-root>/llvm-capstone`). `capstone-test-env.sh` derives *every* path
   from it via `:-`, so QEMU/buildroot/LLVM binaries all resolved to nonexistent paths.
   **Fix:** set `CAPSTONE_REPO_ROOT` to the actual repo (or unset it — the env script
   then computes it from its own location).

2. **Host↔module ioctl-ABI must match.** `create_dom` fails (`observed=-1`) when the
   host's `libcapstone.c` and the loaded kernel module disagree on the
   `ioctl_dom_create_args` struct. Commit `8c7b973` added a `gp_offset` field, so a host
   built from that source **must** run against a module built from it (via the
   `capstone_new.ko` swap), and a host built from `6912474` runs against the baked-in
   `6912474` module (no swap). Keep the buildroot submodule at the parent's committed
   pointer (`6912474`) unless you deliberately swap a freshly-built module.

## Submodule regression (the CLAUDE.md hazard, in the wild)

`caplifive-buildroot` had been reverted from **`8c7b973`** ("deliver the gp-captable
init descriptor into dom_data") back to **`6912474`** — losing the descriptor-delivery
fix from the working tree. The commit is preserved on branch
**`xlang-gp-captable-delivery`**. This session it was restored and **verified working**
(the module's `pr_info` fires: `copied 10664-byte globals template … to dom_data +52fb0`,
landing exactly where the monitor carves `dom_data`, `DOMAIN_DATA_N=96 → 1536`). It was
then reverted to keep this branch clean, because delivery alone does **not** unblock the
real-Lua path (next section). Whoever tackles gp-captable should re-checkout `8c7b973`.

## Why real-Lua perf/conformance is still blocked (deeper than delivery)

Even with delivery working, the **gp-captable init glue** performs capability ops the
current QEMU strictly rejects: `csdelin` on a NONLIN cap (Jul-28 QEMU asserts;
`f4d416c265` made Aug-5 tolerate it) and then `cscincoffset` on an untagged non-gp cap
(Aug-5 asserts at `op_helper.c:598`). No present QEMU binary tolerates all the init ops,
so **no gp-captable Lua domain runs to completion** — including a re-verification of the
published CDP 13/15. This is the "QEMU-permissive vs RTL-enforces" tension noted in
`state/current-state.md` (§ around the `f4d416c265` discussion); making QEMU tolerant
diverges from silicon, so it is a capability-model decision for the QEMU/compiler lane,
not a blind patch.

**Consequence:** the full reference-Lua interpreter needs gp-captable (too many globals
for cjalr's `.capstone_cap_init`), so the *real-Lua* perf (GC-tree revoke cost) and
conformance (`testes/`) measurements cannot run yet. A complete perf + conformance
harness for them was built this session and then **removed from `xlang-lua-conformance`**
(it targets the blocked path); the design is recorded here so it can be rebuilt when the
substrate works:

- **perf:** a `LUA_GC_PERF` domain — a bounded binary tree of Lua tables built/traversed/
  `collectgarbage()`-freed, bracketed by `rdcycle` under `-icount`, built revoke vs
  `LUA_CDP_NO_REVOKE` (the exact cjalr method, on real Lua). Needs a **reclaiming**
  allocator, not the one-way `rof` bump arena (which `cssplit`s under GC churn — a
  separate, real limit surfaced this session).
- **conformance:** open the OS-free stdlib (`string/table/math/utf8/debug` — the amalgam
  excludes them; `math` needs `tan/asin/atan2/log2/log10` in the softfloat libm) and run
  the OS-independent `testes/` files (`closure/coroutine/gc/sort/…`) embedded as C
  strings; the OS-integration tail (`main.lua` subprocess, `api.lua` dlopen) stays
  CHERI-only.

## Retractions (reasoning that ran past the evidence, for the record)

The real cause was found only after retracting: domain-size, big-arena-host,
arena-depletion, and allocator-vs-GC — all wrong; and "the whole environment is broken"
was wrong too (the cjalr path works). The truths: the two env bugs above, the submodule
regression, and the gp-captable-init QEMU strictness.
