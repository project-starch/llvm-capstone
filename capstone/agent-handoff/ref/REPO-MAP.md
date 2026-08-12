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
| what it holds | the LLVM fork (`llvm/lib/Target/Capstone`), `capstone/agent-handoff/`, `capstone/tests/`, `capstone/benchmarks/`, and the submodule gitlinks |

`main` is the default branch but is not what we develop on.

---

## Submodules

Declared in `.gitmodules`. **`branch` is what is checked out locally; `gitlink` is what the parent
records.** When they differ the submodule has moved and the parent has not been told — that is
normal mid-work, and must be resolved before pushing the parent.

| path | purpose | branch (local) | pushed? |
|---|---|---|---|
| `capstone/capstone-ariane` | **the RTL.** CVA6 + Capstone, Anvil sources, directed tests | `fpga-testing-dev` | **NO — unpushed, no upstream** |
| `capstone/capstone-qemu` | the emulator; functional oracle for the board | `s06-lcc-total-query` | yes, in sync |
| `capstone/capstone-c` | Capstone C runtime / cap-table reference | detached `8cda52c` | n/a |
| `capstone/caplifive-system` | board bring-up: buildroot, OpenSBI, device tree | `capstone-bootstrap` | local ahead of gitlink |
| `capstone/capstone-academic-spec` | **the spec we cite.** Exception codes, instruction semantics | `caplifive-s06` | branch local; tracks `origin/caplifive-release` |
| `capstone/capstone-spec` | older spec checkout, branch `caplifive` | `caplifive` | — |
| `capstone/paper` | the paper. **Overleaf owns the remote — never push** | `main` | — |
| `capstone/caplifive-buildroot` | second buildroot copy, **not the live one** | `capstone-bootstrap` | tracks origin |

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
capstone/caplifive-system                                        branch capstone-bootstrap
└── sw/buildroot                                                 branch capstone-bootstrap-dts-65536
    └── components/opensbi                                       branch capstone-bootstrap
        └── lib/sbi/capstone-sbi        <-- THE LIVE MONITOR      branch capstone-bootstrap
            sbi_capstone.c
```

To check whether a monitor change is committed you must `cd` into
`caplifive-system/sw/buildroot/components/opensbi/lib/sbi/capstone-sbi` and run `git status` **there**.

Note `sw/buildroot` is on a differently-named branch (`capstone-bootstrap-dts-65536`) from
everything around it. That is deliberate, not drift.

There is a second, stale copy of the monitor source at
`caplifive-system/sw/buildroot/package/capstone-sbi-domain/capstone-sbi/sbi_capstone.c`.
**Editing that one has no effect on the firmware.**

---

## capstone-ariane branches — read this before touching the RTL

`fpga-testing-dev` is the collection point. As of 2026-08-12 it **contains every other branch's
content**: `origin/fpga-testing`, `origin/fpga-testing-dev-s06`, and `origin/capstone-bootstrap`
are all ancestors of it.

| branch | status | notes |
|---|---|---|
| `fpga-testing-dev` | **ACTIVE — the collection point** | unpushed. Parent gitlink points here |
| `fpga-testing-dev-linear` | candidate: same content, 5 linear commits | see "History decision" below |
| `backup/…` / `fpga-testing-dev` @ `467bdb970` | the 13-commit merged history | keep until the linear version is pushed |
| `origin/fpga-testing` | upstream base | not ours; do not rewrite |
| `origin/fpga-testing-dev-s06` | **superseded** — folded into `fpga-testing-dev` | the source `caplifive_s06.bit` was synthesised from (`c767626a8`) |
| `origin/capstone-bootstrap` | **superseded** — folded into `fpga-testing-dev` | local copy is 4 ahead, all contained |
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

## History decision for capstone-ariane (open)

`fpga-testing-dev` currently has **13 commits and 2 merges** since `origin/fpga-testing`, and the
S-06 enabler appears **twice** with the same subject line (`f89aad25c` local, `c767626a8`
published — not identical: the local one also carries `s06-lcc-scoping.S` and the improved
`s06-lcc-total-query.S`). That is the history that "looks weird".

`fpga-testing-dev-linear` is a prepared alternative: **5 thematic commits, 0 merges**, and its tree
is **byte-identical** to `fpga-testing-dev` (verified with `git diff --quiet`). All five commit
messages pass `precommit-scan.sh`.

```
9b2ce30cd  verif: R-20 does not reproduce, the linear clear is incomplete, and mcause is pinned
e33efdf67  verif: S-06 reproduced in simulation, and the repair sequence
efffa7c47  verif: the store-misclassification family, and its refutation
8e6600a1b  RTL: fix an off-by-one in the ex_code mcause comments
55b7f88bc  RTL: S-06 enabler -- make LCC's type query total
```

**Recommendation: adopt the linear version.** `fpga-testing-dev` has never been pushed, so
rewriting it costs nothing externally. The one real consequence is that `origin/fpga-testing-dev-s06`
and `origin/capstone-bootstrap` stop being ancestors — acceptable because both are superseded and
their content is fully contained, but it means **the bitstream provenance link becomes a
documented fact rather than a git ancestry fact**, which is why it is written down above.

To adopt:

```bash
cd capstone/capstone-ariane
git branch -f backup/fpga-testing-dev-merged-13 fpga-testing-dev   # keep the old shape
git branch -f fpga-testing-dev fpga-testing-dev-linear
git checkout fpga-testing-dev
git push origin fpga-testing-dev
cd ../.. && git add capstone/capstone-ariane && git commit   # bump the gitlink
```

Do **not** delete `origin/fpga-testing-dev-s06` or `origin/capstone-bootstrap` — leave them as
historical anchors.

---

## Outstanding

* **`capstone-ariane/fpga-testing-dev` is unpushed.** Until it is, the parent's gitlink points at a
  commit that exists on no remote, and a fresh clone plus `git submodule update` **fails**. This
  has already happened once. It needs a push by someone with write access.
* Several gitlinks are behind their local HEADs (`capstone-qemu`, `caplifive-system`,
  `capstone-academic-spec`, `paper`). Normal mid-work; resolve before pushing the parent, and never
  bump `paper`.

---

## Rules that are not negotiable

* **Never push `capstone/paper`.** Overleaf owns that remote. Never bump its gitlink either.
* **Push the submodule before the parent**, so the parent never references a commit that does not
  exist remotely.
* **Run `capstone/tests/precommit-scan.sh --msg <file>` before every commit and every push**,
  in every repo including submodules.
* **No real-person names anywhere** — commit subjects included.
