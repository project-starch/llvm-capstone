# Host run, 2026-11-10

The first run of this corpus. `build-host.sh` with a configured PostgreSQL 17.5
tree; `result-lines.txt` is the whole record.

What it establishes, and what it does not:

* **Establishes** that both defects reproduce at the pin against real
  `resowner.c`, that each one's negative control completes, and that the fault
  lands in the labelled free. The second free is reached, which is the defect.
* **Establishes** that the host allocator SEES both: glibc's own tcache check
  raises SIGABRT, and ASan names the double free with the probe frame. That is
  the point of this corpus -- the object freed twice is libc's, so an allocator
  that keys on `free()` is a real oracle here rather than the structural
  silence it is in `../mmgr-repros`.
* **Does not** establish anything about a capability machine. No Capstone,
  Sublet or CheriBSD arm has been run; each case declares those arms
  `not written` rather than silent.
