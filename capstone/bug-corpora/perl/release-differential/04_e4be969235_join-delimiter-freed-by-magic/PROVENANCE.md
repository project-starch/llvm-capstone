# join-delimiter-freed-by-magic

Upstream fix `e4be969235` (2023-09-21), *join: save the delimiter string before anything magical happens to it*. It is an ancestor of
v5.45.3-85-gdb19522155 and is **not** an ancestor of the `v5.36.3` tag, so it never reached the
5.36 maintenance branch — note that a date comparison is not the test here, and for
four of this corpus's eleven cases it would give the wrong answer. The defect is live
in the release this project ports, and the pin's own tree shows why: pin doop.c:667 const char * const delims = SvPV_const(delim, delimlen); used at doop.c:708.

**How it was found.** The release sweep read every commit from `v5.36.3` to that
master tip that touches both C source and a test, kept the ones whose diff has the
shape of a spatial or temporal memory defect, and triaged the survivors by hand
against the pin's sources. `trigger.pl` is the test upstream added with this fix,
extracted onto the shared harness (`harness/shim.pl`) so that one script yields one
verdict line.

**Host oracle, measured both ways.** At the pin: ASan: heap-use-after-free (rc=77).
The same trigger, same harness, same ASan options, on v5.45.3-85-gdb19522155: no report
(rc=0). Liveness here is measured, not inferred from the fix date.

**What the arms read.** See `case.json`. The three Capstone arms are one image each
of the same source and differ only in the heap they link, so a difference between
them is the heap's protection and nothing else.
