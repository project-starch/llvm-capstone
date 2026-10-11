# caller-args-freed-sv

Upstream fix `d2cddbe1df` (2023-03-15), *caller(): don't copy freed SV pointer to @DB::args*. It is an ancestor of
v5.45.3-85-gdb19522155 and is **not** an ancestor of the `v5.36.3` tag, so it never reached the
5.36 maintenance branch — note that a date comparison is not the test here, and for
four of this corpus's eleven cases it would give the wrong answer. The defect is live
in the release this project ports, and the pin's own tree shows why: pin pp_ctl.c:2064 Copy(AvALLOC(ary), AvARRAY(PL_dbargs), AvFILLp(ary)+1+off, SV*).

**How it was found.** The release sweep read every commit from `v5.36.3` to that
master tip that touches both C source and a test, kept the ones whose diff has the
shape of a spatial or temporal memory defect, and triaged the survivors by hand
against the pin's sources. `trigger.pl` is the test upstream added with this fix,
extracted onto the shared harness (`harness/shim.pl`) so that one script yields one
verdict line.

**Host oracle, measured both ways.** At the pin: no ASan report; Perl's own check: panic: attempt to copy freed scalar (rc=255).
The same trigger, same harness, same ASan options, on v5.45.3-85-gdb19522155: no report
(rc=0). Liveness here is measured, not inferred from the fix date.

**What the arms read.** See `case.json`. The three Capstone arms are one image each
of the same source and differ only in the heap they link, so a difference between
them is the heap's protection and nothing else.
