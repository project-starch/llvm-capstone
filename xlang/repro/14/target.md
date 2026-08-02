# target.md — mruby #3596 (Row 14)

* **CVE/Issue:** mruby #3596
* **Product:** mruby
* **Vulnerability Type:** Heap Use-After-Free (UAF) in the GC stack-root scanner (`mark_context_stack`)
* **Status:** REPRODUCED
* **Vulnerable Tag/Commit:** `491d68bb3004eb8d7deec4a3a682b25de0d4afc2` (vulnerable parent of `5c114c91`)
* **Fix Commit:** `5c114c91d4ff31859fcd84cf8bf349b737b90d99`
* **Crash Site:** `src/gc.c:556` (`mark_context_stack`) — freed in `incremental_sweep_phase` (`src/gc.c:1055`)
* **Determinism:** aborts on 10/10 consecutive native-ASan runs; no fuzzing or heap grooming needed.

> **Note — this row was previously filed as SKIPPED.** The earlier rationale held
> that the defect "does not produce an immediate ASan abort unless precise,
> non-deterministic fuzzer-driven heap layouts and reallocation alignments are
> met". That is not the case: the trigger in `trigger.rb` aborts deterministically
> on every run at exactly the site the benchmark table names
> (`mark_context_stack`). The row is reclassified as REPRODUCED and `asan.txt`
> records the trace.
