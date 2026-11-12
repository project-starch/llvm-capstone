# 2026-11-12-crash-139

Why CRASH-139 (case 41) reads differently in run19 than in run18, measured, and the repair of its reading. This record is the current one for the three virtual arms: its bundles are run19's boots, judged again by `ports/php/app/run-realphp.py --from-record` after CRASH-139 declared a reach witness. Of the 560 cells of run19 (70 cases, four arms, two scorings) 552 are byte-identical rows and 8 moved, all of them CRASH-139's: MISSED became NO-READING `not-reached` on the four arms in both scorings. No boot was repeated for the bundles. The boots this record adds are measurements of the one case and probes, under `runs/` and `probes/`.

- measured image: run19's, `php-real-nested.dom` sha256 `a8a77eadd554cc8577e139654d634270d21052294bd0d75e9cecf05789a87949`, on run19's platform (`images.sha256`)
- judge: run19's, plus the reach rule stated here (each bundle's `inputs.json`, `scoring.reach`)
- declared before judging: `41_9a98904ddd0e_crash-139/oracle.json` `reach_witness`, written by `tools/build-cases.py` (`REACH_WITNESSES`)
- every table in this file is computed: the measurement summary from `runs/matrix.tsv`, the moved cells from `rejudge/cells.tsv`, the result tables by `tools/result-tables.py` from the bundles

## The defect, and what run19 left open

`zend_read_property()` (Zend/zend_API.c:1957) builds a zval on the C stack, points it at the static string `"message"` and never sets its reference count. When the class has `__get()`, the engine passes that zval as an argument: it adds one to the count for the call and subtracts one afterwards. If the count is then zero, `_zval_ptr_dtor` frees the zval's string, which is the static string: an invalid free (CWE-590). So the free executes when the uninitialised field holds zero, and only then.

In run18 the two nested arms stopped with `ZEND_CAP_FAIL code=3`: the nested layer refused the pointer by name, and the judge read NO-READING `unattributed`. In run19, and on unmodified `dev`, all four arms completed and the judge read MISSED, the one MISSED of the nested arms over the 55 temporal cases. Run19's README says: "What changed them is not established here."

## Question 1: does the invalid free execute?

It does not, on any arm, in any run of the measured image. The uninitialised field holds 48 on every arm in every run, so the count never reaches zero and `_efree` never receives the static string. The outcome does not vary between runs, between boots or between arms.

| image and script | runs per arm | stock | system allocator | nested | nested exact |
|---|--:|---|---|---|---|
| measured image `a8a77ead…`, the trigger | 6 | 6 of 6: exit 0, completes | 6 of 6: exit 0, completes | 6 of 6: exit 0, completes | 6 of 6: exit 0, completes |
| measured image `a8a77ead…`, the probe script | 3 | 3 of 3: exit 0, completes, script prints refcount(51) | 3 of 3: exit 0, completes, script prints refcount(51) | 3 of 3: exit 0, completes, script prints refcount(51) | 3 of 3: exit 0, completes, script prints refcount(51) |
| image from unmodified `dev` `9e4d529f…` on `dev`'s QEMU, the trigger | 3 | 3 of 3: exit 0, completes | 3 of 3: exit 0, completes | 3 of 3: exit 0, completes | 3 of 3: exit 0, completes |
| image from unmodified `dev` `9e4d529f…` on `dev`'s QEMU, the probe script | 3 | 3 of 3: exit 0, completes, script prints refcount(51) | 3 of 3: exit 0, completes, script prints refcount(51) | 3 of 3: exit 0, completes, script prints refcount(51) | 3 of 3: exit 0, completes, script prints refcount(51) |
| probe image *witness* `f5a8de29…`, the trigger | 3 | 3 of 3: exit 0, completes, field = 48, foreign frees = 0 | 3 of 3: exit 0, completes, field = 48, foreign frees = 0 | 3 of 3: exit 0, completes, field = 48, foreign frees = 0 | 3 of 3: exit 0, completes, field = 48, foreign frees = 0 |
| probe image *witness* `f5a8de29…`, the probe script | 3 | 3 of 3: exit 0, completes, script prints refcount(51), field = 48, foreign frees = 0 | 3 of 3: exit 0, completes, script prints refcount(51), field = 48, foreign frees = 0 | 3 of 3: exit 0, completes, script prints refcount(51), field = 48, foreign frees = 0 | 3 of 3: exit 0, completes, script prints refcount(51), field = 48, foreign frees = 0 |
| probe image *forced* `9c88ec19…`, the trigger | 3 | 3 of 3: exit 139, fault cause=2, field = 0, foreign frees = 1 | 3 of 3: exit 139, fault cause=2, field = 0, foreign frees = 1 | 3 of 3: exit 139, fault cause=2, field = 0, foreign frees = 1 | 3 of 3: exit 139, fault cause=2, field = 0, foreign frees = 1 |
| probe image *proposed* `5ad90d86…`, the trigger | 3 | 3 of 3: exit 0, completes | 3 of 3: exit 0, completes | 3 of 3: exit 0, completes | 3 of 3: exit 0, completes |
| probe image *proposed* `5ad90d86…`, the probe script | 3 | 3 of 3: exit 0, completes, script prints refcount(51) | 3 of 3: exit 0, completes, script prints refcount(51) | 3 of 3: exit 0, completes, script prints refcount(51) | 3 of 3: exit 0, completes, script prints refcount(51) |
| probe image *proposed, forced* `6c35fddd…`, the trigger | 3 | 3 of 3: exit 134, ZEND_CAP_FAIL code=3 (free) | 3 of 3: exit 134, ZEND_CAP_FAIL code=3 (free) | 3 of 3: exit 134, ZEND_CAP_FAIL code=3 (free) | 3 of 3: exit 134, ZEND_CAP_FAIL code=3 (free) |

Four instruments, from the least to the most invasive.

**The trigger on the measured image.** Two boots, three runs per arm in each, the arms in opposite order in the second boot, every run its own process with the arm's `PHP_CAP_MODE`. All 24 runs complete with the pristine output (`v`, `end`) and exit 0, with no fault and no `ZEND_CAP_FAIL` line (`runs/measured-boot1.json`, `runs/measured-boot2.json`).

**A probe script on the measured image** (`probes/probe-refcount.php`). It is the trigger with one change: `__get()` prints the reference count of the zval it receives, with `debug_zval_dump`. Everything that runs before `__get()` is the trigger's, statement for statement. It prints `refcount(51)` in all 12 runs. The witness build calibrates the offset: the engine has added three by the time the script prints, so the field held 48.

**A diagnostic build, *witness*** (`probes/witness-*.diff`). It differs from the measured image in two translation units. `zend_std_call_getter` stores the member zval's reference count in a static before the engine counts it. `zend_cap_reclaim`, the layer's free entry, counts its calls and reports every pointer whose page no mapping of the heap owns. The additions use statics and no new locals, and the frames of the functions they touch keep their size (`probes/frames.tsv`). In all 12 trigger runs the field is 48 and no foreign pointer arrives. In the 12 runs of the probe script the build reads 48 where the script prints 51.

**A second diagnostic build, *forced*** (`probes/forced-zend_API.diff`). It is the witness build with the two uninitialised fields set to zero in `zend_read_property`. This is the defect executing. In all 12 runs, on all four arms, the free arrives at `zend_cap_reclaim` as its 91st call with a capability over exactly the 8 bytes of `"message"` in `.rodata`, and the run ends with a fault. The call path at the fault is ASan's own stack for this case: `_efree` (zend_alloc.c:284) from `_zval_dtor` (zend_variables.c:45) from `_zval_ptr_dtor` (zend_execute_API.c:391) from `zend_ptr_stack_clear_multiple` from `zend_call_function` (zend_execute_API.c:856).

The measured image is certain by the first and the last of these together. The layer looks up every pointer `_efree` is given, in every mode, before anything ends. A run in which the free executes therefore stops inside the layer, as the forced build shows on every arm. The 24 runs of the measured image complete, so in none of them did the free execute. The probe script and the witness build add the reason: the field is 48, not zero.

The measured image was rebuilt for this record from the first pass's sources and reproduces `a8a77ead…` byte for byte. The probe images relink that build's own objects with the named translation units recompiled, and relinking the unchanged objects gives `a8a77ead…` again (`probes/probe-image.sh`, `probes/ns-build.sh`). A probe image is a probe: none of its runs enters a bundle.

## What the layer does when the free does execute

It stops the run in software, on every arm, the stock arm included. `zend_cap_reclaim` calls `cap_block_of`, which calls `heap_access`. `heap_access` finds no heap mapping for the string's page and executes `__builtin_trap()`: a domain fault with cause 2 (illegal instruction) in `heap_access` (heap_sublet.h:106), called from `cap_block_of` (zend-cap-mode.c:187), `zend_cap_reclaim`, `_efree`. The process ends with exit 139. In that call no lifetime instruction runs and no capability check fires: the pointer is a live, tagged capability over a static string, and the refusal is the layer's own lookup.

This is not a processor fault at a declared site, and this record does not count it as a catch. With the reach rule of this record the judge reads such a run as reached, by the witness, and NO-READING `unattributed` (`probes/judged.tsv`). Before the rule it would have read NO-READING `setup-fault` on every arm, because the stock arm stops in the same way and could not license reach. Whether a refusal by the layer's software check should count for the nested arms is not decided here.

## What changed since run18

Established:

- The layer of run18 made the same check in every mode. Its `cap_block_of` asked the libc's registry for the pointer's block (`__capstone_malloc_object`) and printed `ZEND_CAP_FAIL code=3 ... a=<cursor> b=0x0` when there was none. Run18's bundles show that line on the two nested arms and a completed run on the stock and system-allocator arms, and run18's sensitivity bundles show the same check refusing CRASH-012's pointer on the stock arm. It follows that in run18's image the free executed on the nested arms and did not execute on the other two: the field was zero on the nested arms only, and the outcome did depend on the arm.
- In today's image the field is 48 on all four arms, on the branch and on unmodified `dev` alike (`runs/dev-control-boot1.json`: the probe script prints `refcount(51)` on `dev`'s image and `dev`'s QEMU). The change is therefore older than the one-operand CREVOKE and is not the instruction change's.
- Between run18's source and `dev` the layer changed how it finds a pointer's block. Today's `cap_block_of` reads the block's capability from a prefix before PHP's header, through the heap's mapping index (`heap_access`). The refusal changed form with it: a pointer outside the heap traps in `heap_access` before `cap_block_of` reaches its `cap_fail`.

Not established:

- That the field was zero on run18's nested arms is inferred from run18's record and source, not measured: run18's image is not on the build host.
- Which earlier call writes the 48, and that the layer's rewrite is what moved it. The image is compiled without optimisation, so every argument of every layer call has a stack slot, and 48 equals `ZEND_CAP_HDR`, the header size each layer entry receives. That suggests the origin and does not prove it.

## The reach rule

**The rule.** A case whose defect executes only on a condition the script does not control, here uninitialised memory, declares a reach witness in its `oracle.json`: an event that the arm's own transcript shows when the defect executes. For such a case the stock arm's completed run licenses reach on no arm, the stock arm included. An arm is reached only by its own transcript: the witness, or a fault at the declared site as before. A completed run without the witness reads NO-READING `not-reached`. CRASH-139 declares the witness `nested-free-entry`: the layer's free entry stopped on the pointer, shown by a `ZEND_CAP_FAIL` code 3 or 4 line that names a free or by a fault whose call path is `_efree` to `zend_cap_reclaim`. A witness establishes reach and nothing else, so a run that shows it reads `unattributed`, never CAUGHT.

**Why the old reading was an artefact.** The judge took reach from the stock arm: the stock arm completed with the pristine output, one image serves every arm, so the trigger ran on every arm. That argument shows that the trigger ran. It shows that the defect ran only when the defect follows from the script alone. CRASH-139's defect also needs the field to be zero, and on this image it is 48. Every arm completed because the defect never executed, and the judge read that silence as the mechanism missing a defect.

**What the old reading was.** MISSED on stock, system allocator, nested and nested exact, in the default and the sensitivity scoring of run19 (and of `dev`, run19's `control-dev/`). Run19's bundles and README are unchanged and keep that observation.

**The witness is measured in the same run.** It is the event itself, the free arriving at the layer, read from the transcript of the run that the judge reads. Its absence is evidence because the layer looks at every pointer `_efree` receives on every arm, and the forced build shows the witness on all four arms when the free executes.

**The rule touches one case.** Only a case that declares `reach_witness` takes the new route, and only CRASH-139 declares one (`tools/build-cases.py --self-test` checks the committed files against the table). With the changed runner and no declaration, all 560 rows of run19 come out byte-identical (`rejudge/cells-moved-without-declaration.txt`). With the declaration:

| scoring | arm | case | run19 | this record |
|---|---|---|---|---|
| default | stock | 41_9a98904ddd0e_crash-139 | MISSED | NO-READING:not-reached |
| default | system allocator | 41_9a98904ddd0e_crash-139 | MISSED | NO-READING:not-reached |
| default | nested | 41_9a98904ddd0e_crash-139 | MISSED | NO-READING:not-reached |
| default | nested exact | 41_9a98904ddd0e_crash-139 | MISSED | NO-READING:not-reached |
| sensitivity | stock | 41_9a98904ddd0e_crash-139 | MISSED | NO-READING:not-reached |
| sensitivity | system allocator | 41_9a98904ddd0e_crash-139 | MISSED | NO-READING:not-reached |
| sensitivity | nested | 41_9a98904ddd0e_crash-139 | MISSED | NO-READING:not-reached |
| sensitivity | nested exact | 41_9a98904ddd0e_crash-139 | MISSED | NO-READING:not-reached |

`rejudge/cells.tsv` lists all 560 cells. `rejudge/rejudge.sh` is the procedure: run19's stored run records and `platform.json`, the runner as it stands, the first pass's working-directory layout.

The `cheribsd-revocation` arm is not covered. It runs no nested layer, so this witness does not exist there, and its runner (`bug-corpora/php/tools/run-php-corpus.py`) does not read the declaration. Its MISSED for CRASH-139 (`results/2026-10-30-cheribsd`) rests on "the trigger printed its own output, so it ran", which shows the trigger ran and not that the free executed. Nobody has measured the field on that platform. That cell is unchanged and is open.

## The tables

### Default scoring (native crashers out of the denominator)

Arms in this record: stock, system allocator, nested, nested exact. Not measured here: CheriBSD revocation.

**all 70 cases** (70):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 6 | 1 | 32 | infra 2, not-reached 2, out-of-denominator 28 |
| system allocator | 17 | 1 | 20 | infra 2, not-reached 2, out-of-denominator 28, unattributed 1 |
| nested | 31 | 1 | 9 | infra 1, not-reached 1, out-of-denominator 28 |
| nested exact | 36 | 2 | 3 | infra 1, not-reached 1, out-of-denominator 28, unattributed 1 |

**the temporal cases (CWE-416, CWE-415, CWE-590)** (55):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 1 | 1 | 23 | infra 2, not-reached 2, out-of-denominator 27 |
| system allocator | 12 | 1 | 11 | infra 2, not-reached 2, out-of-denominator 27, unattributed 1 |
| nested | 26 | 1 | 0 | infra 1, not-reached 1, out-of-denominator 27 |
| nested exact | 26 | 1 | 0 | infra 1, not-reached 1, out-of-denominator 27 |

**the CWE-416 cases** (51):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 1 | 1 | 23 | infra 2, not-reached 1, out-of-denominator 24 |
| system allocator | 12 | 1 | 11 | infra 2, not-reached 1, out-of-denominator 24, unattributed 1 |
| nested | 26 | 1 | 0 | infra 1, out-of-denominator 24 |
| nested exact | 26 | 1 | 0 | infra 1, out-of-denominator 24 |

**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):

- stock: 1 of 28 / 0 of 23
- system allocator: 5 of 28 / 7 of 23
- nested: 5 of 28 / 21 of 23
- nested exact: 5 of 28 / 21 of 23

### Sensitivity scoring (every case scored by fault site)

Arms in this record: stock, system allocator, nested, nested exact. Not measured here: CheriBSD revocation.

**all 70 cases** (70):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 7 | 1 | 32 | infra 4, not-reached 14, out-of-denominator 1, setup-fault 7, unattributed 5 |
| system allocator | 30 | 1 | 20 | infra 3, not-reached 3, out-of-denominator 1, setup-fault 9, unattributed 4 |
| nested | 54 | 2 | 9 | infra 2, not-reached 2, out-of-denominator 1, unattributed 2 |
| nested exact | 59 | 3 | 3 | infra 2, not-reached 2, out-of-denominator 1, unattributed 3 |

**the temporal cases (CWE-416, CWE-415, CWE-590)** (55):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 2 | 1 | 23 | infra 4, not-reached 13, out-of-denominator 1, setup-fault 7, unattributed 5 |
| system allocator | 25 | 1 | 11 | infra 3, not-reached 2, out-of-denominator 1, setup-fault 9, unattributed 4 |
| nested | 49 | 2 | 0 | infra 2, not-reached 1, out-of-denominator 1, unattributed 2 |
| nested exact | 49 | 2 | 0 | infra 2, not-reached 1, out-of-denominator 1, unattributed 2 |

**the CWE-416 cases** (51):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 2 | 1 | 23 | infra 3, not-reached 12, out-of-denominator 1, setup-fault 7, unattributed 3 |
| system allocator | 25 | 1 | 11 | infra 2, not-reached 1, out-of-denominator 1, setup-fault 9, unattributed 2 |
| nested | 49 | 2 | 0 | infra 1, out-of-denominator 1 |
| nested exact | 49 | 2 | 0 | infra 1, out-of-denominator 1 |

**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):

- stock: 2 of 28 / 0 of 23
- system allocator: 18 of 28 / 7 of 23
- nested: 27 of 28 / 22 of 23
- nested exact: 27 of 28 / 22 of 23

Against run19, per arm and scoring: MISSED one lower, `not-reached` one higher, nothing else. The nested arms have no MISSED over the 55 temporal cases again, as in run18, and for a stated reason: CRASH-139's defect did not execute on this image.

## The two other cases that depend on stack contents

**CRASH-092.** The rule would change nothing. The case already has per-arm reach by another route: it declares an execution requirement (a working loopback), which closes the stock route, so a completed run reads `not-reached`, a hang reads `infra` and an arm's own fault in the declared function licenses that arm alone. That is how it reads in run18 and in run19. Its dependence on stack contents (by the corpus generator's note, an uninitialised `tv_usec` decides whether the script hangs before its defect) shows as movement between those three readings, each of which is honest for the run it describes. A reach witness would need an event that appears when the use-after-free executes without being caught, and a silent use-after-free leaves none in a transcript. This record did not repeat the case, so whether run19's CAUGHT on the nested arms repeats is not measured.

**CRASH-153.** The rule does not apply in this form. By the trigger's own account the defect is itself the uninitialised read: `_phpi_pop` dereferences a pointer it never set, and that read executes whenever the script reaches it, whatever the memory holds. What depends on stack contents is then whether the read faults, not whether the defect executes. In the default scoring the case is out of the denominator on every arm (native PHP dies on it), under any reach rule. In the sensitivity scoring the stock route is already closed for a case whose pristine record is a signal, so reach is already per arm: run18's fault in `_phpi_pop` read `unattributed` (cause 24), run19's completed runs read `not-reached`. With the rule both read the same. Run19's runs print `NULL` and `after`, and the transcript does not say whether the read executed on a harmless value or the function returned before it. A different witness could say so, and if it showed the read executing, a completed run would read MISSED in the sensitivity scoring on every arm. This record measured nothing on that case and declares no such witness.

## A change to the layer that this record proposes and does not apply

`probes/proposed-zend-cap-mode.diff` restores the refusal by name that run18's source had: `cap_block_of` asks the mapping index whether the pointer's page belongs to the heap before it calls `heap_access`, and calls `cap_fail(ZEND_CAP_FAIL_UNKNOWN, ...)` when it does not. Two probe images show what it does (table rows *proposed* and *proposed, forced*; `probes/proposed-boot1.json`, `probes/proposed-forced-boot1.json`). With the free forced, all four arms print `ZEND_CAP_FAIL code=3 free of a pointer this heap did not issue, or not at its start a=<cursor> b=0x0` and abort, which is run18's line. Without forcing, the sanity control, the trigger and the probe script complete on all four arms, and the field is still 48.

The change is not applied, for two reasons. The measurement answers the question without it: the free does not execute, so the layer lost no check that this case exercises. And any change to the layer's code can change what its calls leave on the stack, which is what this case, CRASH-092 and CRASH-153 depend on, so it belongs with a full run of the corpus and not with a re-judgement.
