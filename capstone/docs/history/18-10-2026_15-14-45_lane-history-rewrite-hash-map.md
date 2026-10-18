# Lane history rewrite: old -> new commit hashes (lane/compiler-exact-bounds-iii, 2026-10-18)

**Why.** A commit pushed to this lane branch added, in `capstone/tests/rtl-smoke/vp2-flat-table-2026-10-17/csr-census.py`, a hard-coded absolute path containing a personal account name. A later commit on the same branch removed it, so it was absent from the tip and from `dev`, but it stayed in the branch's history. The project's name rule requires a pushed named commit to be rewritten and force-pushed. The project lead confirmed this rewrite in the session.

**What changed.** Only that one literal, replaced by `/mnt/hdd/$USER/`, in the two commits whose trees carried it, plus the hashes of every commit built on them. Nothing else:
- the new tip's tree is byte-identical to the old tip's;
- all 126 commits in the rewritten range have identical author, committer, dates and message;
- 78 of them are `dev` commits merged into the lane, and all 78 keep their hashes;
- 48 lane commits are rehashed, each with a tree identical to the original or differing only in that one file;
- a case-insensitive search over every commit in the range found the literal in 2 commits before the rewrite and 0 after.

Old tip `dec1de0e6077` is now `00a99eb17bdb`. No other remote branch contained the old commits.

**Use this table to resolve a citation of an old hash.** 18 citations on `dev` point into it: 2 in commit messages and 16 in files under `capstone/`. They are marked below. Every other old hash in this range is an unchanged `dev` commit and still resolves as written.

**A gate limitation found while doing this.** `precommit-scan.sh --range` reads one NET diff of the range (`git diff "$RANGE"`), so a line that one commit adds and a later commit in the same range removes never reaches it. It passed the original range as CLEAN. A history rewrite therefore needs a per-commit check: `git log -i -G<pattern> <range>`, with a commit known to contain the pattern as its positive control.

| old | new | cited on dev | subject |
|---|---|---|---|
| `dec1de0e6077` | `00a99eb17bdb` |  | Tier 3 mpt predicted, and a bundle's platform fields do not identify what built it |
| `db22d0fc41a0` | `38b38b831190` |  | Merge origin/dev: keep both sides of virtualvm's batch script |
| `2ca32e29d0df` | `3164c594f0bb` |  | The arena pin cuts faults 244 to 176 with no exit moved, and trips a test that encodes ... |
| `ed7561c8d91f` | `f0e30b6d071b` |  | S-25's narrow arena pin: one line widening pin_ahead's limit, default off |
| `52b2e742d2d8` | `222935e8112c` | file | The bisection names the Q-26 wait rewrite, and it is reverted to the board-proven source |
| `4ae9edeec902` | `66d98b59c7eb` |  | C-81's image arms run: the prototype converts one site per image, and my void gate is m... |
| `199118c5153a` | `f24b77208ef2` |  | RETRACTED: resolve_dirty's default was not the boot-4 regression, and the QEMU control ... |
| `695fdc90e774` | `d8b097012cf1` |  | A QEMU 34/41 does not license a default that changes a fault path -- and the subset I m... |
| `8bc0c4dd88cc` | `bd0db482acdf` | file | RETRACTED: resolve_dirty must not default ON -- it regressed every virtual context on s... |
| `66d9f10361de` | `eb62d28c1381` |  | run.py: the same -cpu knob as run-staged.py, and resolve_dirty on by default with the Q... |
| `e55611a374d9` | `2e8ae60b2950` |  | S-26: print the frame's event words RAW at each escape, budgeted at 16 |
| `48e6e9e48c0e` | `13d3b05b5575` |  | r13: S-24 resolved on silicon -- the PTE reads D = 0, and the fix clears the loop |
| `c594bd503a67` | `12c790ef916b` |  | run-staged.py: append to the -cpu string from the environment, and write the resolved s... |
| `f6f87502ea6d` | `be3f83e8a7f9` |  | run.py: module parameters at insmod, so a two-sided module experiment is one run |
| `a05468f63cc7` | `1f30db344c3e` |  | Merge remote-tracking branch 'origin/dev' into lane/compiler-exact-bounds-iii |
| `3afc812421e2` | `4b80aeced7cf` |  | fixup_user_fault needs the caller to hold mmap_read_lock; I asserted the opposite in a ... |
| `5f7c33ba94fe` | `78368cee36c7` |  | Record the refuted delivery explanation, and pre-register the PTE readings for both tar... |
| `3f8ea1e5ba42` | `a9ebbe700ef7` | file | Is an STC ever the first touch of a fresh page? Yes per page, no per slot -- so the nar... |
| `b212abb1b841` | `3d292bc23c91` | file | S-24's fix: resolve a kind-4 STORE through the kernel's own write-fault path, so D gets... |
| `eb7a903d1b38` | `599c04294b5f` | file | RETRACTION: GUP does not set A or D on an anon page -- I quoted follow_pfn_pte, not fol... |
| `4f4c1f396d09` | `865f9b93c276` |  | r12: none of the five sites fired, the S-24 bound did, so the !same deduction is refute... |
| `1080498eb174` | `ebb509215b36` | file | Scope the eager-pin workaround: it is 2 lines, the cost is 30x the estimate, and one ch... |
| `b8020329a9fd` | `13c376a3f52b` |  | run.py reads the module's lines from the ring buffer, and my own QEMU records are audit... |
| `d9750cf1d817` | `e736a5366b84` |  | pre-register which FIELD of the refusal line decides for r13, and the pair it cannot se... |
| `15a2e6620f2e` | `753ed3e7f319` |  | the A/D candidate narrows to three disjuncts: every resolve() sets A and D and fences t... |
| `81235c27a7b7` | `1ec56fc2efcd` |  | every diagnostic a verdict reads becomes level-selectable, and three of them stop being... |
| `0b963e269b4e` | `67b7df422e11` | file | RETRACTION: "the board console drops KERN_INFO, so pr_warn shows" was wrong, and the ba... |
| `0174153bc37a` | `3a46536e02bf` |  | C-81: record that the toolchain behind the compiler arms is stale, and what that does n... |
| `bf6e8e6a32ec` | `04a28c07a0ec` |  | C-81: the narrow-decline prototype's compiler arms hold, and convert one shape more tha... |
| `671ce4502786` | `c5a4033e2dbe` |  | no fence stands between the module's stores and the domain entry, and adding one is free |
| `d66030133aa6` | `0fbc99edad51` | file | fence_before_entry is writable at runtime, so the arm is two-sided in one boot |
| `afbaff6e029e` | `615111f2e881` | file | an optional fence before the domain entry, off by default, as one module with two arms |
| `2b70e429d585` | `3bd4674e79e7` |  | the refusal diagnostic fires on QEMU, and every pre-registered prediction holds |
| `741db7c84605` | `6effa1f70ef0` |  | resolve()'s refusal lines name the faulting pc |
| `df117205f6da` | `54887519fd58` |  | resolve() names which of its five refusals fired, with a must-fire load banner |
| `fe8c4283f8ad` | `9bbf4d2ba700` | file | svadu=false costs nothing, and refutes the synthesis I offered |
| `7307c5d5649d` | `b5549625e72b` |  | S-21's positive control: the escalation succeeds without the fix and is denied with it |
| `69db2f133b93` | `35b246605b85` | file | Board-facing refusals must be pr_warn: KERN_INFO is invisible in a board transcript |
| `b929ea962a70` | `ef709163f446` |  | File C-81, a day late: the number was in use by three folders and two docs with no regi... |
| `2aab20db74e1` | `2b860da17701` |  | S-21: retire a child thread's frame grant at CV_THREAD_CREATE, before it can be entered |
| `b04983b244e1` | `1fef42dc28f1` | file | S-24: bound a fault that resolves without changing anything, instead of spinning silently |
| `457b78f27e10` | `cbf2bd85a819` | message, file | Q-30: the adapter overwrites the reply register slot instead of requiring it empty |
| `39cf6763d3a6` | `3a5e3bb3f302` | message, file | Q-26: the module serialises contexts on a one-slot machine, refusing at thread_create |
| `77b3bccd4626` | `54b0d66e6a71` |  | Remove a personal name from csr-census.py, and fix the gate order that let it through |
| `6ec2d7b380c2` | `73484d068e05` | file | Retract C-82's exposure claim: the defective walks are dead code the runtime already ov... |
| `619e038ea550` | `9e6bc8d4b313` |  | File C-82: musl routes function pointers through uintptr_t, so the init/exit walks unta... |
| `cbfa22b4c957` | `783cf9fe7bbf` | file | Q-21 two-sided on the strict-CSR pin: the control fires with the board's own line, the ... |
| `c1411ddbde54` | `c7f366863d55` |  | Q-21 evidence: census 0 with the pre-fix images as control, behaviour unchanged, cost m... |
