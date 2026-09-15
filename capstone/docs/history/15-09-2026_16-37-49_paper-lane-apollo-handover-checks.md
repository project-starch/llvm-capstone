# The paper lane on apollo: what the handover's claims survive, and one blocker that is structural (2026-09-15)

*Written by the paper lane after taking the handover on the apollo host. Read
`docs/plans/2026-09-15-paper-lane-handover.md` and
`docs/history/15-09-2026_15-44-40_sublet-draft-audit.md` first; this note records what checking
those two against primary sources produced.*

## 0. The structural blocker: the manuscript is unreachable from this host

Established by trying, not assumed:

* `git ls-remote https://github.com/project-starch/nested-allocators-paper.git` returns **403
  "Write access to repository not granted"** with the stored agent credential
  (`~/.agent-creds/cred-kisp`). GitHub's wording is misleading: this is a **read** operation, so
  the refusal covers fetch, not just push.
* Anonymous access (`git -c credential.helper= ls-remote`) prompts for a username, i.e. the
  repository is **private**.
* There is no clone on this host: no `.git/modules/capstone/paper-nested-allocators`, and nothing
  matching anywhere under the host user's home directory. The superproject's working tree carries an uncommitted
  `.gitmodules` change adding `update = none` for that submodule, which is a workaround for the
  recursive-clone abort and not for this.

The outgoing paper lane confirmed, by reading its own reflog rather than from memory, that **no
agent session ever had read access on either host**. The copy on the previous host exists because
the project lead ran `git pull` there under their own credentials; the agent read from the local
object store afterwards. It also reports that the lead's authenticated `gh` does **not** resolve
this repository either (`gh repo list` enumerates 25 repositories in the organisation without it,
including other private ones), so whatever credential the lead's `git pull` used is a third thing
— worth knowing, because that is the sort of split nobody notices until a script fails.

**Consequence for this lane.** Every line number in the audit that points into the paper
repository — `appendices/c-validation-and-accounting.tex:34-37` and `:55-58`, `METHODS.md:86`,
`scripts/check_experiments.py:196` and `:129-132`, and the `studies.json` evidence states — is a
**claim pending verification**, not a checked fact, and is marked as such here. The owed
`studies.json` edits (R1 re-apply, **M2 → partial** although its bundle is committed) are not
executable at all.

**The fix is one of two things**, both the lead's: carry
`~/capstone-artifacts/board-branch-2026-09-15.bundle` (2.8 MB, on the previous host, built to
carry both `board/e1-s1s2-hardware` and `origin/drafts`) onto this host, or grant read access to
the account this host uses. Until then the lane can analyse claims against the measurements doc
and the RTL, which is what the rest of this note is.

## 1. The audit's unquoted positive claim is TRUE — checked against the RTL

The audit says the four plain-data-access rows of `tab:safety` are unsupported, and then that
**"The capability-access rows stand — the DYN unit's node-validity query carries no privilege or
capmode gate at all."** The first half had a quoted line; the second half did not. It holds:

* `capmode` appears in exactly five core files, and in only three places that do anything:
  `load_store_unit.sv:966` (the gate that is the subject of R-34 and of the audit's §1),
  `commit_stage.sv:208` (`priv_lvl_i == PRIV_LVL_M && capmode_i`, the PC-metadata path), and
  `csr_regfile.sv` where it is produced.
* In `ex_stage.sv` it is a port (`:98`) and a single pass-through to the LSU (`:782`). It never
  enters the Capstone unit block that begins at `:794`.
* **`capstone_dyn_unit` has no `capmode` and no `priv_lvl` port at all** (its full instantiation,
  `ex_stage.sv:939-1008`). Its request `capstone_dyn_req_data` is built at `:810` from `fu_data_i`
  with no privilege or capmode term.
* The node-validity query itself is the `_ep_query_req_*` / `_ep_query_res_*` endpoint pair of
  `capstone_rev_node` (`ex_stage.sv:1092-1097`, matching ports on the DYN unit at `:984-989`).
  Neither endpoint carries privilege or capmode.

So the asymmetry the audit asserts is real and is visible in the port lists: the LSU's plain-data
check is privilege-gated and the DYN unit's capability check is not. This matters because it is
the half of the safety-table decision that **survives**, and the lead is about to rule on that
table.

Two further facts from the same reading, both confirming claims that were already recorded:
`riscv_pkg.sv` does define **24 = `DEBUG_REQUEST`** with the capability causes at 25-28 (so
R-24's retraction of the pre-registered "traps, cause 24" is correct), and the LSU gate is
literally `CVA6Cfg.CAPSTONE_EXT && capmode_i && ld_st_priv_lvl_i == riscv::PRIV_LVL_M`
(`load_store_unit.sv:966-969`).

## 1b. The live contradiction: its SILICON half is confirmed here, its MANUSCRIPT half is not

The audit's §1 sets a manuscript claim against a measurement. Only one side of that is checkable
from this host, and it is the measurement side, which holds:

* §7r's matrix carries the row **"p5 stop 5 (old pointer reads the new occupant)"** with
  `C5005B` in the expected column **and** `C5005B` in the read column
  (`docs/ref/fpga-silicon-measurements-for-paper.md:3460`) — the protected arm read what the
  unprotected arm reads.
* The prose at `:3558-3559` states it directly: *"after the same address was handed to a new
  object, the old pointer read the new occupant's first byte 0x5B on silicon, exactly what the
  plain arm reads (p5)."*
* `:3596` records **6 `unsafe-success`** results in the bundle.

(Unrelated coincidence worth not tripping over: `:997` notes that every primitive in `sublet.h` is
**opcode** `0x5b`. That is a different `5b` from the occupant byte in `C5005B`.)

What is **not** checkable here is the other side of the contradiction — that `tab:safety` claims
"Stops at access" on those rows and that the prose at
`appendices/c-validation-and-accounting.tex:55-58` attributes the `0x5B` return to the
*unprotected* arm. Those line numbers are in the paper repository. They are reported by the
outgoing lane and are consistent with everything measurable here, but this lane has not read them;
see §0.

The decision itself — whether the six unsafe-success cells bind the P1/R1 timing numbers under
`METHODS.md:86`, or are a labelled configuration fact — is the lead's, and the handover has
already put it. Nothing in this note changes its shape.

## 2. The audit's result table reproduces against the measurements doc

Every figure in the audit's §2 was checked against the section it cites. All reproduce, and the
three caveats that must travel with them — the C-32 workaround, the lookaside-OFF exclusion, and
the WNS −12.425 ns bitstream with §7s's 0.02-0.05 % cross-boot bound — are stated **at** the
numbers rather than only in a header. The ledger's arithmetic checks: 1,344,064 + 693,680 +
215,076 = 2,252,820, and 2,252,820 / 1,291,712 = 1.744.

Two cosmetic differences, neither an error, recorded so nobody re-derives them:

* the audit gives depth flatness as **0.0132 %** where §7t rounds to **0.013 %** (478,067 to
  478,130 across depth 1-8 is 0.01318 %);
* the audit pairs **1.3624 × 1.1797** where §7s multiplies the other boot's **1.3627 × 1.1791**.
  Both land on 1.607.

## 3. What that check turned up instead: the doc's header contradicted its own entries

Fixed in `286a9e4bc968`. The "Vehicle throughout" paragraph named a single bitstream for the
whole file and stated that *every number below predates the 2026-08-04 reflash and must be
re-measured before it is compared with anything taken after*. It was written 2026-07-27 and is
false for most of the file it introduces — which matters more here than in any other document,
because that file's own opening sentence says it is what a paper author lifts from. A reader
following the header would have discarded or re-measured the entire Sublet series.

§4e is dated 2026-08-14 and so already postdates the reflash the header treated as the end of the
file; at least five bitstreams appear below, not two; and within §7 the vehicle changes twice
(§7b-§7c on `s12fix_5097eb166`, §7e-§7l on `r25r26r27_66c4e7517`, everything from §7m on
`r30r31_1bfff7776`). The correction quotes the old wording rather than replacing it silently, and
changes no number.

## 4. `precommit-scan.sh --range` blocks on ordinary project history

Not a new rule, a measurement of an existing gate, made because F1's note describes a narrower
version of it ("the scan's range mode blocks the merge on author lines of the collaborator's S2
commits already on dev"). It is much broader than the collaborator's commits.

`bash capstone/tests/precommit-scan.sh --range dev~40..dev` **exits 1**, and every hit is an
**author or committer identity line that the scan generated itself** from its own
`git log --format='%H%n%an <%ae>%n%cn <%ce>%n%s%n%b'`. No commit message, no diff, no file content
is implicated. Each hit trips two independent detectors, the denylist name check and the email
check.

Of the three distinct identities authoring the last 400 `capstone/` commits on `dev`, **two match
the denylist** — one in its name field, one in its email field — and they account for **399 of
those 400 commits**. The script's existing mitigation drops only the committing user's *own*
configured identity, which by construction can never drop the other identities present in shared
history.

This is the lead's ruling and is listed as decision 3 of the handover. It is recorded here with
numbers so that the ruling is cheap to make, and because the tempting local fix — relaxing a
pattern until a push succeeds — is forbidden by both the script's own closing message and
CLAUDE.md. `--msg` and staged-diff mode are unaffected; only `--range` is.

**Apollo-specific corollary.** `git config user.name` and `user.email` are unset on this host:
`~/.gitconfig` does not exist and the agent gitconfig sets only `include.path` and
`credential.helper`, so `git var GIT_AUTHOR_IDENT` fails and **`git commit` refuses outright**
until a lane sets one. Both identities already present in the history match the denylist, so this
lane set a repo-local identity that does not (the agent account), and its commits are clean under
the gate. Any lane arriving on this host hits the same wall before its first commit.

## 6. R-34: a sufficiency condition on the FIX that nothing has measured

This section contains **a retraction of my own**, made before anything was recorded, and one live
finding that replaced it.

**Retracted.** I proposed that `ex_stage.sv:1015` —
`assign load_exception_o = forward_normal_load_valid ? load_exception : '0;`, introduced by fork
commit `f4a306d86` where the pre-fork wiring was a direct port connection — is a **second,
fork-introduced site** at which R-34's exceptions are lost, and therefore that the attribution
"a base-core defect, not an extension's" was unproven. **That is wrong, and the mask is inert.**
`load_exception_o` has no consumer that is not already gated by the same signal:
`ex_stage.sv:1013` drives `load_valid_o` from `forward_normal_load_valid`; that becomes
`wt_valid_i[LOAD_WB]` at the scoreboard, and `scoreboard.sv:215`'s
`if (wt_valid_i[i] && mem_q[trans_id_i[i]].issued)` **encloses** `:236`'s
`if (ex_i[i].valid) mem_n[trans_id_i[i]].sbe.ex = ex_i[i];`. The forwarding path is gated the same
way (`issue_read_operands.sv:648`). Upstream's unmasked assignment would behave identically at
every consumer. Masking by a signal that already gates the only readers removes nothing.

Two things to say about how that retraction happened, because they generalise. I first withdrew
the claim on reading the R-34 box's waveform sentence — *"At every fire the waveform shows
`ex_i.valid = 1`, `state_q = IDLE`, `ex_o.valid = 0`"* — which places the loss upstream of
ex_stage. That was the right conclusion for the **wrong reason**: the sentence is not reproducible
from any committed artifact (the repro folder's `sim/vcd-timing.txt` selects 14 signals, twelve
under `lsu_i` and two CSR, and contains no `ex_i`, `ex_o`, `state_q` or `cap_exception`; the VCD
itself is overwritten by every run and is gone). The argument that actually settles it — "is
`load_exception_o` read anywhere not already gated by `load_valid_o`?" — is a grep, and I did not
run it before reaching for the waveform.

**What is live, and is not covered by any existing observation.** The substantive fork edit is one
line above the mask. `ex_stage.sv:1013` re-sources `load_valid_o` from the LSU's own `load_valid`
to the syncer's `forward_normal_load_valid` (the original is still there, commented out at
`:1017`). That matters because of an asymmetry in what the syncer carries:

* the syncer's message type is `lsu_result_pack_t` — **`core/anvil_build/capstone_unit.anvilh:565-568`,
  exactly two fields, `trans_id` and `cap_result`. There is no exception field.**
* `trans_id` and the data come from that message (`ex_stage.sv:1014`, `:1016`), but the exception
  comes **straight from the LSU** (`ex_stage.sv:721`).

Data and trans_id are timing-independent; an exception is not. So **if `forward_normal_load_valid`
lags `load_valid` by even one cycle, every load still retires with correct data and a correct
trans_id, and every single-cycle exception is silently dropped** — by the scoreboard gate above,
not by the mask. Whether the syncer's bypass is same-cycle is argued by the Anvil source
(`capstone_dyn_unit.anvil:589-595`, and the one-cycle `normal_res` lifetime at
`capstone_unit.anvilh:579`) but **not settled**: the language reference's sync-mode grammar ends in
`TODO: more info needed`, the generated `*.anvil.sv` is gitignored and absent from this host, and
the Anvil compiler is not buildable here.

**Why this is worth a paragraph rather than a shrug.** All three exception deliveries ever observed
on this core — the cause-24 event and the two in the refused run — were **multi-cycle holds**. A
restored SEND_TAG delivery is a **one-cycle** exception, a shape never yet observed. So a one-cycle
syncer lag is invisible in every existing artifact and would defeat the R-34 fix **after** it is
built. That is the CLAUDE.md sufficiency question, and a bitstream here costs ~90 minutes plus a
reflash.

**It is answerable without a bitstream, without an RTL change and without the board.** The R-34
repro already dumps a full VCD and its `sim/readvcd.py` takes signal patterns as arguments. Re-run
it extracting `ex_stage_i.load_valid`, `ex_stage_i.forward_normal_load_valid`,
`ex_stage_i.load_exception`, `ex_stage_i.load_exception_o` and the store equivalents, alongside
`lsu_i.i_load_unit.ex_o` and `state_q`. **Predictions, written before the run:** *same-cycle* —
`forward_normal_load_valid` rises with `load_valid` on every plain load and `load_exception_o.valid`
is 1 at the cause-24 event; *delayed* — it lags by one cycle throughout and `load_exception_o.valid`
is 1 only where `load_exception.valid` held ≥ 2 cycles. Both readings are informative, so the run
qualifies under the go/no-go rule.

**This lane cannot run that experiment, and the failure is recorded rather than assumed.** On
apollo: `core/*.anvil.sv` does not exist, `core/anvil.Flist` does not exist (and
`core/Flist.cva6:138` pulls it in, so verilation dies at file-list expansion), there is no `anvil`
binary on the host, and `docker images` returns *"permission denied while trying to connect to the
docker API"* — the account has no passwordless sudo. Verilator 5.008 **is** present at
`capstone-ariane/tools/verilator-v5.008`, so the only missing piece is the Anvil-generated RTL,
which comes from a container image. The run therefore belongs to a lane on the host that owns the
toolchain; it is handed over rather than dropped.

*Provenance of the lines in this section:* `scoreboard.sv:215`/`:236`,
`capstone_unit.anvilh:565-568`, `ex_stage.sv:1013`/`:1015`/`:1017`/`:721`/`:1014`/`:1016`,
`base_isa_tests.sh:53-58` and the `testlist_capstone.yaml` counts were **re-read here**. One
citation in the report this came from was wrong and was corrected by re-reading — the message type
is at `capstone_unit.anvilh:565-568`, not `:470-473`. The remaining citations —
`issue_read_operands.sv:648`, `load_unit.sv:698`/`:702`/`:715-717`, `capstone_dyn_unit.anvil:589-595`
and the `sim/vcd-timing.txt` signal list — are **reported, not re-read by this lane**, and should be
checked before anything is built on them.

Two adjacent items for whoever fixes R-34, from the same reading: `load_unit.sv:718`'s delivery is
nested inside `:698`'s `req_port_i.data_rvalid`, so a restored exception is delivered only when
rvalid coincides with `SEND_TAG` (the miss / `kill_req` path is unchecked); and the comment at
`load_unit.sv:715-717` describes **pre-#2528** timing, so an RTL lane reading it as the current
contract would wrongly conclude the `:423-424` fix option does not exist. Flag the comment in the
same commit as the fix.

**Not affected by any of this:** R-34's existence, the four safety-matrix rows, and §7x's
uncontamination argument, which never uses the attribution and holds either way. I did **not** find
evidence that the attribution is false. What I found is that one of its three legs is currently
unauditable, and that a fix targeting only the LSU/MMU has an unverified sufficiency condition.

**One more thing the fork's own gate cannot tell you** (re-read against the files, not taken on the
auditor's word). `verif/regress/base_isa_tests.sh:53-58`
runs five tests — `rv64ui-v-{add,ld,sd,beq,jal}` — and the list it drives
(`verif/tests/testlist_capstone.yaml`) contains **121 tests and zero `rv64mi` of any kind** — both
counts re-derived here — so no machine-mode exception test at all. The commit messages "Non Interference with Base ISA verified"
and "Tests passing" therefore carry no information about `rv64mi-p-ma_addr` in either direction.
That does not show the fork broke it — whether the pre-fork parent passed it is **UNRESOLVED**, and
nothing in-tree records a result for any revision — but it is why a defect of this shape could sit
undetected for seventeen months, and it belongs in the R-34 box.

## 7. A tree-wide name sweep found four committed violations, and why the gate never saw them

`precommit-scan.sh` is a **flow** gate: it reads the staged diff, the unstaged diff, the untracked
files and the commit message. **It has no whole-tree mode.** So anything already in the tree is
invisible to it permanently — including content committed before the rule was tightened, before a
name entered the denylist, or on a day the gate was not run.

Sweeping all 3,205 tracked files in our own trees against the denylist found a personal first name
in five files. Four are genuine violations and are scrubbed (two boot notes and a gdb note carrying
our own build-host string in the `user@host` form the rule names explicitly; one history note
attributing an upstream commit by full personal name; one archived plan recording a backup path
under a personal home directory). The fifth, `docs/history/SQLiteProposal.tex:497`, is a
**published-paper citation URL**, which CLAUDE.md exempts by name — left alone. After the scrub the
tree has exactly that one hit.

**The gate works; it had simply never been pointed at the tree.** Proven rather than asserted: from
a clean baseline (exit 0), copying one of the offending committed files in as an untracked file
makes the same gate exit 1 and name the exact line.

**A scrub commit cannot pass a content scan**, because a diff that removes a name contains that
name on its `-` lines. No pattern was weakened. Instead: all five name hits were checked
mechanically to sit on `-` lines with **zero on `+` lines**, and the post-scrub tree was re-swept
directly. The two remaining `user@host` hits are that same removal line plus a pre-existing,
unchanged build string of the form *root-account at a generic build host* — a role account, not a
person, and the false-positive class
the script's own closing message invites you to confirm by eye.

**STATUS, and read this before trusting the paragraph above.** The scrub exists **only in the
working tree on apollo. It is not committed and not on `origin`.** Its commit was refused twice by
this session's permission classifier — reasons *"Security Test Removal"* and *"Out-of-Place
Publication"* — and I stopped rather than reword the change to get past a refusal. So a reader who
runs the sweep against `origin/dev` today finds **five** hits, not one, and the four files still
carry the name. It also means the scrub sits **unstaged in a checkout other lanes on this host may
share**, where a broad `git commit` from any of them would carry it under their own message. This
needs the lead: either approve the commit, or discard the working-tree change.

**What a commit would still not fix, and is also the lead's:** the names remain in git history, on
`dev` and 27 other remote branches. Scrubbing forward does not touch that, and the remedy CLAUDE.md prescribes
for an already-pushed name — rewrite and force-push — is irreversible and outward-facing.

## 8. Method notes, because several of these were nearly mistakes

* **A gate is not evidence until it is shown to fire.** `precommit-scan.sh` was negative-tested
  both ways before being relied on: a commit message seeded with a denylisted name **exits 1**, a
  clean message **exits 0**. The denylist now lives at `~/.claude-kisp/secrets/name-denylist.txt`
  following the secrets-directory rename; CLAUDE.md still names the old `~/.claude-c/secrets/`
  path, and the script is the one that is right.
* **A count taken with a hand-rolled instrument reproduced a known false positive.** A first pass
  at the authorship question matched the denylist against every commit touching our trees and
  reported 2,455 of 4,100 — a number that includes exactly the self-match class the script's
  2026-09-07 fix exists to drop. It was discarded and replaced with the real gate's own exit
  status and output. The published numbers in §4 are the script's, not that script's.
* **A `git reset --soft` onto a newer upstream tip stages the upstream commits in the reverting
  direction.** Re-applying this note's sibling commit on a moved `origin/dev` left the two files
  of the intervening commit staged as deletions, and one of them absent from the working tree.
  `git commit -o <path>` committed only the intended file and nothing else — which is the whole
  point of the rule — but the index stayed poisoned afterwards and had to be restored explicitly
  with `git reset HEAD -- <paths>` and `git checkout -- <paths>`. The `-o` rule protects the
  commit; it does not clean up the index behind it.
