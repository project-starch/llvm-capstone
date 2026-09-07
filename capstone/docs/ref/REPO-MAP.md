# Repository and branch map

**What this is for.** This tree is a parent repo plus eight submodules, two of which have
submodules of their own, three levels deep. Several things in this project have gone wrong purely
because someone (including agents) did not know which repo or which branch a file lived in:

* a `git status` reported the live OpenSBI monitor "clean" while not covering the file at all,
  because it sits three submodule levels down;
* the parent's gitlink pointed at a commit that existed on no remote, so a fresh clone plus
  `git submodule update` failed for everyone;
* a test was committed to the wrong branch and had to be moved.

Keep it current. When you move a branch, push one, or change a gitlink, update the table in the
same commit. A stale map is worse than none, because it will be trusted.

**Last verified: 2026-08-12.** Every value below was read from the working tree, not remembered.

---

## The one-line version

Work happens on the parent's `capstone-bootstrap`. The RTL lives in `capstone-ariane`, and the
branch that now collects **all** of our RTL and verification work is **`fpga-testing-dev`**. The
emulator is `capstone-qemu` on `s06-lcc-total-query`. The live monitor is **not** in
`caplifive-system` directly — it is three levels down, in the nested `capstone-sbi` repo.

---

## Parent repo

| | |
|---|---|
| path | the repo root — `git rev-parse --show-toplevel` |
| remote | `github.com/project-starch/llvm-capstone` |
| working branch | `capstone-bootstrap` |
| what it holds | the LLVM fork (`llvm/lib/Target/Capstone`), `capstone/docs/`, `capstone/tests/`, `capstone/benchmarks/`, and the submodule gitlinks |

`main` is the default branch but is not what we develop on.

---

## Submodules

Declared in `.gitmodules`. **`branch` is what is checked out locally; `gitlink` is what the parent
records.** When they differ the submodule has moved and the parent has not been told — that is
normal mid-work, and must be resolved before pushing the parent.

| path | purpose | branch (local) | pushed? |
|---|---|---|---|
| `capstone/capstone-ariane` | **the RTL.** CVA6 + Capstone, Anvil sources, directed tests | `fpga-testing-dev` | yes — `7e4dc440f`, gitlink in sync |
| `capstone/capstone-qemu` | the emulator; functional oracle for the board | `capstone-bootstrap` | yes — pushed `62bf0f1d61` |
| `capstone/capstone-c` | Capstone C runtime / cap-table reference | detached `8cda52c` | n/a |
| `capstone/caplifive-system` | board bring-up: buildroot, OpenSBI, device tree (remote: `caplifive-system-dev`) | `capstone-bootstrap` | pushed `28f9436` (2026-09-07) |
| `capstone/capstone-academic-spec` | **the spec we cite.** Exception codes, instruction semantics | `caplifive-s06` | branch local; tracks `origin/caplifive-release` |
| `capstone/capstone-spec` | older spec checkout, branch `caplifive` | `caplifive` | — |
| `capstone/paper` | the paper. **Overleaf owns the remote — never push** | `main` | — |
| `capstone/caplifive-buildroot` | second buildroot copy: **the QEMU stand-in** (its `components/opensbi` monitor is what QEMU's `DOM_CREATE` ecall reaches); not the board's | `capstone-bootstrap` | pushed `4d97ecf` (2026-09-07) |

### Two traps in that table

**There are two spec checkouts.** `capstone-academic-spec` (branch `caplifive-s06`) is the one
carrying our amendments and the one to cite. `capstone-spec` is a separate older checkout on branch
`caplifive`. Check which you are in before quoting a line number.

**There are two buildroots.** The **live** one, whose overlay the board image is built from, is
`caplifive-system/sw/buildroot`. `caplifive-buildroot` is a separate copy and staging a domain
there does nothing. Its `sqlite_silicon.dom` dates from 2026-08-02 and is not what runs.

---

## Nested submodules — where the live monitor actually is

`caplifive-system` contains submodules of its own, and the OpenSBI monitor is three levels down.
A `git status` run at `caplifive-system` **does not see it**.

```
capstone/caplifive-system            (project-starch/caplifive-system-dev)   branch capstone-bootstrap
└── sw/buildroot                     (project-starch/caplifive-buildroot)    branch capstone-bootstrap-unified
    ├── components/opensbi           (project-starch/caplifive-opensbi)      branch capstone-bootstrap-unified
    │   └── lib/sbi/capstone-sbi     (project-starch/capstone-sbi)           branch capstone-bootstrap-unified
    │       sbi_capstone.c           <-- THE MONITOR, one source for both targets
    └── package/capstone-sbi-domain/capstone-sbi (project-starch/caplifive-sbi)  977af95 (the sbi.dom copy)
```

**Since 2026-09-07 the QEMU stand-in is the SAME tree** (`docs/plans/monitor-unification.md`): the
second checkout under `capstone/caplifive-buildroot` tracks the same `capstone-bootstrap-unified` on
every level and differs only in which target it builds (`make TARGET=qemu …`, output `build-qemu/`,
`build` a symlink to it there; the board checkout builds `TARGET=fpga` into `build-fpga/`, `build` a
symlink to it). Per-target code in the monitor sits under `CAPSTONE_TARGET_FPGA` /
`CAPSTONE_TARGET_QEMU` (`capstone_target.h`); the OpenSBI platform directory (`fpga/ariane` vs
`generic`) supplies the define for OpenSBI's own compile, `CAPSTONE_EXTRA_DEFS` for the capstone-c
regeneration. `CAPSTONE_CC_PATH` is required and printed on every build.

To check whether a monitor change is committed you must `cd` into the `capstone-sbi` directory of the
copy you edited and run `git status` **there**. `git diff` one level up (at `components/opensbi`) shows
only `sbi_capstone_dom.c` and silently omits `sbi_capstone.c` — it is a submodule deeper.

**Why the branch names differed until 2026-09-07 (the "zoo"), stated once, for the history.** Each shared repo is checked out two or
three times in this tree, and the checkouts hold different code: the board flavour and the QEMU
flavour of `caplifive-buildroot.git` last shared a commit in January 2025 (`2f56383`; 32 board commits
— FPGA clock/baud, the device-tree range for the 65536-node bitstream that named the branch, kernel
config, modcapstone fixes — against 12 QEMU commits); the monitor repo has the board line (29 commits
from `2f772bb`), the fw_jump stand-in line (4 from the same point) and the template-copy line
(`origin/capstone-bootstrap`, `04ac643`); the opensbi stand-in forks from `769939a` while the board
line has 25 commits past it. Every local checkout called its branch `capstone-bootstrap` and none was
pushed until 2026-09-07, so one name meant three histories; the `-board`/`-qemu` suffixes are those
histories made visible. Collapsing them was a merge of the two flavours into one source with
build-config switches, followed by rebuilding fw_jump and the board firmware and revalidating both —
done 2026-09-07 as `capstone-bootstrap-unified` (`docs/plans/monitor-unification.md`); the `-board`
and `-qemu` branches and the `pre-unify/2026-09-08/*` tags stay as the frozen pre-merge lines.

There is a second, stale copy of the monitor source at
`caplifive-system/sw/buildroot/package/capstone-sbi-domain/capstone-sbi/sbi_capstone.c`.
**Editing that one has no effect on the firmware** (it builds the board's `sbi.dom`; unported for Q-03).

---

## capstone-ariane branches — read this before touching the RTL

`fpga-testing-dev` is the collection point. As of 2026-08-12 it **contains every other branch's
content**: `origin/fpga-testing`, `origin/fpga-testing-dev-s06`, and `origin/capstone-bootstrap`
are all ancestors of it.

| branch | status | notes |
|---|---|---|
| `fpga-testing-dev` | **ACTIVE — the collection point** | 5 linear commits since 2026-08-12. Unpushed. Parent gitlink points here |
| `fpga-testing-dev-linear` | **ADOPTED** — same ref as `fpga-testing-dev` | see "History decision" below |
| `fpga-testing-dev-merged-backup` @ `467bdb970` | the old 13-commit merged history | keep until the linear version is pushed and clone-verified |
| `origin/fpga-testing` | upstream base | not ours; do not rewrite |
| `origin/fpga-testing-dev-s06` | **superseded** — folded into `fpga-testing-dev` | the source `caplifive_s06.bit` was synthesised from (`c767626a8`) |
| `origin/capstone-bootstrap` | **superseded** — folded into `fpga-testing-dev` | local copy is 4 ahead, all contained |
| `origin/fpga-testing-fix` | **INTEGRATED 2026-08-12** — now an ancestor of `fpga-testing-dev` | the collaborator's LDC write-permission check (closes R-23) and linear-source clearing. Eight tests had to be adapted; see below |
| `origin/r20-fix` | **redundant** | identical patch-id to the R-20 fix already on `fpga-testing-dev` (`f623c48a1`). Contributes nothing |
| `fpga-testing` | tracks `origin/fpga-testing` | |

### Bitstream provenance

The resident bitstream is **`caplifive_s06.bit`**, synthesised from `c767626a8` on
`fpga-testing-dev-s06`. Against the current `fpga-testing-dev` the **only** RTL difference is a
comment-only change in `core/anvil_build/capstone_unit.anvilh`, which recompiles to identical
SystemVerilog. So the resident bitstream is valid for this branch. **If that stops being true,
say so here** — a board result taken against RTL the bitstream does not match is void.

Board drivers default to expecting `caplifive_fixed_forward.bit`; override with
`FPGA_BITSTREAM=caplifive_s06.bit` or they hard-stop.

---

## History decision for capstone-ariane — DECIDED AND APPLIED 2026-08-12

`fpga-testing-dev` is now **5 thematic commits, 0 merges** since `origin/fpga-testing`. It was
13 commits with 2 merges, and the S-06 enabler appeared **twice** under the same subject
(`f89aad25c` local, `c767626a8` published — not duplicates: the local one also carried
`s06-lcc-scoping.S` and the improved `s06-lcc-total-query.S`). That is what "looked weird", and it
is gone.

```
9b2ce30cd  verif: R-20 does not reproduce, the linear clear is incomplete, and mcause is pinned
e33efdf67  verif: S-06 reproduced in simulation, and the repair sequence
efffa7c47  verif: the store-misclassification family, and its refutation
8e6600a1b  RTL: fix an off-by-one in the ex_code mcause comments
55b7f88bc  RTL: S-06 enabler -- make LCC's type query total
f623c48a1  Fix R-20: keep the CAPENTER x10 clobber additive   <- pre-existing base
```

**Why this was safe.** The rewrite was content-neutral, verified by TREE HASH, not by eyeball:
both shapes resolve to tree `d71c357bc0db7f103dcfb207d705a68bed03bd64`. Nothing was lost, because
nothing *could* be lost — the trees are the same object. And `fpga-testing-dev` had never been
pushed, so no published history was rewritten.

**What it cost, stated plainly.** `origin/fpga-testing-dev-s06`, `origin/capstone-bootstrap` and
`origin/r20-fix` are **no longer ancestors**. Their content is still fully present (same tree), but
containment is no longer a git-provable property. Two consequences to remember:

* **Bitstream provenance is now a documented fact, not an ancestry fact.** `caplifive_s06.bit` was
  synthesised from `c767626a8`; the RTL on `fpga-testing-dev` differs from it by exactly one
  comment-only change to `capstone_unit.anvilh`. See "Bitstream provenance" above. If that diff
  ever grows beyond comments, board results taken on this bitstream are void.
* **Do not merge those branches into `fpga-testing-dev` later.** They would replay content that is
  already there. They are superseded anchors, not inputs.

**Reversible.** The 13-commit shape is kept at `fpga-testing-dev-merged-backup` (`467bdb970`), and
`fpga-testing-dev-linear` still points at the adopted tip. Delete neither until the branch has been
pushed and a fresh clone has been verified.

Do **not** delete `origin/fpga-testing-dev-s06` or `origin/capstone-bootstrap` — leave them as
historical anchors.

---

## Integrating fpga-testing-fix — what it changed and what it broke

Two real RTL fixes, both spec-checked before being taken:

* **LDC now requires WRITE permission before clearing the source.** Closes **R-23**, which this
  repo carried as an open SPEC VIOLATION. `check_load_data` raises `INSUFFICIENT_PERMISSION`;
  `load_unit.sv` gates the clear on `rs1_perm_write` as a second barrier. The shape matters:
  skipping the clear WITHOUT trapping would be worse than the bug, since memory keeps its copy
  and the register gains one. Permission encoding verified rather than assumed — `R=4, W=2, X=1`,
  so bit 1 is write, matching the spec's `2 <=p x[rs1].perms` (`mem-access-insn.adoc:46`).
* **Linear sources are consumed.** `STC` clears `rs2`; `CINCOFFSET`/`CINCOFFSETIMM`/`SCC` clear
  `rs1` when it is linear and `rd != rs1`. Previously the source register kept its copy, so
  storing or deriving from a linear capability DUPLICATED it.

**Eight tests had to be adapted, and the two failure modes are worth knowing because both will
recur.**

1. **Five incoming tests** used the LCC **type** query as a "was this cleared" detector, requiring
   it to TRAP on `NOT_CAP`. The S-06 enabler makes that query TOTAL. Only the cleared-detector
   probes move to the **validity** query (`x0`); probes that READ A TYPE VALUE keep `x1`, because
   `x0` returns validity, not type. A blanket swap was tried first and just moved the failure.
2. **Three of ours** built a `CAP_TYPE_LIN` base authority and derived from it repeatedly, so the
   linearity fix nulled it on first use and they HUNG. Now `CAP_TYPE_NONLIN`, which is what a
   repeatedly-derived-from authority should be.

**Read cycle counts, not the verdict word.** Those three hangs printed
`*** SUCCESS *** (tohost = 0) after 2000013 cycles`, and 2000013 IS the `+time_out` value — a hang
wearing a pass. Any summary grepping for `SUCCESS` calls that merge green. The integration is
accepted because all 19 tests with a known-good reference run in EXACTLY the cycle count they run
in on the branch where they were authored; a pass alone would not show a test had not been
silently neutered.

**Not yet on silicon.** The linearity change alters behaviour beneath every board result taken so
far, and the domain entry glue derives `gp` from `sp` via `scc` — the exact construct that now
consumes its source. Nothing here is validated on hardware until a bitstream carries it.

---

## Gitlink audit — 2026-08-12

Every submodule checked three ways: does the parent's gitlink equal the local HEAD, is the gitlink
reachable on a remote, is the local HEAD reachable on a remote. A gitlink that is not on a remote
breaks `git submodule update` for everyone; that has already happened once on `capstone-ariane`.

| submodule | branch | gitlink == HEAD | both on a remote |
|---|---|---|---|
| `capstone-ariane` | `fpga-testing-dev` | yes `7e4dc440f` | yes |
| `capstone-qemu` | `capstone-bootstrap` | yes `62bf0f1d6` | yes |
| `caplifive-buildroot` | `capstone-bootstrap` | yes | yes |
| `capstone-c` | detached `8cda52c6` | yes | yes |
| `capstone-spec` | `caplifive` | yes | yes |
| `capstone-academic-spec` | `caplifive-s06` | **was stale — BUMPED** to `fa9600fa` | yes (`origin/caplifive-release`) |
| `caplifive-system` | `capstone-bootstrap` | **DIFFERS — left alone deliberately** | **no, see below** |
| `paper` | `main` | **DIFFERS — left alone deliberately** | yes |

**Two divergences are INTENTIONAL. Do not "fix" either.**

* **`paper`** — never bump its gitlink and never push it; Overleaf owns that remote. The divergence
  is the correct steady state, not drift.
* **`caplifive-system`** — local HEAD `6c5c740` ("traced monitor") is on **NO remote**, and the
  gitlink `50e9ca8d` predates it. Bumping the gitlink now would point the parent at a commit nobody
  can fetch, which is exactly the bug just fixed on `capstone-ariane`. **The fix is to push first,
  then bump — in that order.** The chain is four levels deep and two of them are unpushed:

  | level | branch | on a remote? |
  |---|---|---|
  | `caplifive-system` | `capstone-bootstrap` | **NO** |
  | `sw/buildroot` | `capstone-bootstrap-dts-65536` | yes |
  | `…/components/opensbi` | `capstone-bootstrap` | yes (`origin/genesys-testing`) |
  | `…/capstone-sbi` | `capstone-bootstrap` | **NO** |

  Push bottom-up: `capstone-sbi` → `caplifive-system`. Note `caplifive-system` has TWO remotes,
  `origin` (upstream) and `fork` (`caplifive-system-dev`); the current gitlink lives on `fork/master`,
  so the fork is the one in use.

  **Consequence while this stands:** a fresh clone cannot reproduce the firmware the board runs,
  because the monitor that emits the `EXCX`/`MCAU`/`MEPC` trap report is only in the unpushed chain.

## Gitlink audit — 2026-09-07 — the chain is pushed; the 08-12 divergence is closed

Every nested repo fetched and its HEAD compared with its remote branch; every superproject-recorded
SHA checked for reachability on the nested remote (`git branch -r --contains`). All nine lines are on
their remotes and every gitlink resolves, so a fresh clone reproduces both the board firmware
(`fw_payload 44c88d9ebeb1`, boot sw30) and the QEMU stand-in (`fw_jump d9b11509c2f4`).

| checkout | remote | branch | HEAD |
|---|---|---|---|
| parent `dev` | llvm-capstone | `dev` | `9927476d933b` |
| `caplifive-system` | `caplifive-system-dev` (now `origin`, the fork rule is retired) | `capstone-bootstrap` | `28f9436` |
| `…/sw/buildroot` | `caplifive-buildroot` | `capstone-bootstrap-dts-65536` | `5764378` |
| `…/components/opensbi` | `caplifive-opensbi` | `capstone-bootstrap` | `0ac0290` |
| `…/lib/sbi/capstone-sbi` | `capstone-sbi` | `capstone-bootstrap-board` (new) | `56dfbe9` |
| `caplifive-buildroot` | `caplifive-buildroot` | `capstone-bootstrap` | `4d97ecf` |
| `…/components/opensbi` | `capstone-opensbi` | `capstone-bootstrap-qemu` (new) | `1048a61` |
| `…/lib/sbi/capstone-sbi` | `capstone-sbi` | `capstone-bootstrap-qemu` (new) | `e1ccb49` |
| `…/package/capstone-sbi-domain/capstone-sbi` | `caplifive-sbi` | `capstone-bootstrap` | `977af95` |

The two new `-board`/`-qemu` names exist because those lines are siblings of the remote's
`capstone-bootstrap`, not descendants (see "Why the branch names differ" above); a plain push was
rejected non-fast-forward and must never be forced. Push order stays bottom-up.

## Gitlink audit — 2026-09-08 — the unified state, local until the lead pushes

After `docs/plans/monitor-unification.md`: one branch per nested repo, both checkouts on it. Every
row below is a **branch creation** on its remote (the `pre-unify/2026-09-08/*` tags mark the
previous tips, also unpushed). Parent `dev` references `caplifive-system` and `caplifive-buildroot`
gitlinks that are not on their remotes until these pushes land — push nested repos deepest-first,
then the parent is already consistent.

| checkout | remote | branch | HEAD |
|---|---|---|---|
| `caplifive-system` | `caplifive-system-dev` | `capstone-bootstrap` (fast-forward) | `4686afa` |
| `…/sw/buildroot` and `caplifive-buildroot` | `caplifive-buildroot` | `capstone-bootstrap-unified` (new) | `41a2a1a` |
| `…/components/opensbi` (both) | `caplifive-opensbi` and `capstone-opensbi` — two remotes, one history: push to both | `capstone-bootstrap-unified` (new) | `ea34f91` |
| `…/lib/sbi/capstone-sbi` (both) | `capstone-sbi` | `capstone-bootstrap-unified` (new) | `1b87df8` |
| `…/package/capstone-sbi-domain/capstone-sbi` | `caplifive-sbi` | `capstone-bootstrap` (unchanged) | `977af95` |

After pushing from one checkout of a shared remote, `git fetch` in the other so both see the branch.

---

## Outstanding

* ~~`capstone-ariane/fpga-testing-dev` is unpushed.~~ **RESOLVED 2026-08-12** — pushed, gitlink in
  sync at `7e4dc440f`, and `origin/fpga-testing-fix` is an ancestor of it.
* Several gitlinks are behind their local HEADs (`caplifive-system`, `capstone-academic-spec`,
  `paper`). Normal mid-work; resolve before pushing the parent, and never bump `paper`.
* **`capstone-qemu` is RESOLVED (2026-08-12).** The parent's gitlink used to point at
  `2b87bc9d2b38`, which sits on `origin/capstone-qemu` — **a different lineage from our work**, and
  it did **not** contain the S-06 LCC mirror. `s06-lcc-total-query` was a clean fast-forward ahead
  of `capstone-bootstrap` (two commits: the pre-existing gp-fabrication work, then the mirror), so
  `capstone-bootstrap` was fast-forwarded onto it and pushed as `62bf0f1d61`. The gitlink now points
  there. `capstone-bootstrap` is the branch to use for this submodule; `s06-lcc-total-query` is
  superseded and needs no further merge.

---

## Rules that are not negotiable

* **Never push `capstone/paper`.** Overleaf owns that remote. Never bump its gitlink either.
* **Push the submodule before the parent**, so the parent never references a commit that does not
  exist remotely.
* **Run `capstone/tests/precommit-scan.sh --msg <file>` before every commit and every push**,
  in every repo including submodules.
* **No real-person names anywhere** — commit subjects included.
