# Live at 17.0, already fixed by 17.5

These three cases are correct, measured and reproducible. They are parked here
because the PostgreSQL study targets **17.5**, and their upstream fixes landed
before it:

| case | upstream fix | fix date | in 17.0 | in 17.5 |
|---|---|---|---|---|
| `tidstore_context_deleted_first` | `83ce20d671` | 2024-12-04 | no | **yes** |
| `child_sjinfo_shared_relids` | `727bc6ac33f6` | 2025-02-19 | no | **yes** |
| `windowagg_partition_reset` | `9d5ce4f1a00a` | 2024-12-09 | no | **yes** |

Established by `git merge-base --is-ancestor <fix> REL_17_5`, not by comparing
dates: REL_17_0 is 2024-09-23 and REL_17_5 is 2025-05-05, and a date alone does
not say whether a fix was backpatched into the branch.

The corpus had declared `upstream.version: 17.0` while every other PostgreSQL
corpus in the study is pinned at 17.5, and that inconsistency is what let three
already-fixed defects sit in a 17.5 result table. The corpus is now pinned at
17.5 and these three no longer count.

They are kept rather than deleted: each carries results from five arms, and
they remain valid evidence about 17.0. Nothing reads this directory -- the
checker discovers corpora by `corpus.json`, and there is none here.
