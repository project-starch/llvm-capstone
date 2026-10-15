# task-stack-envs

Upstream fix `cb51fce92` (2026-07-31), *Detach envs from a task stack before freeing it*, which is in master and postdates the
4.0.0-rc2 pin. The defect is therefore live in the release this project ports.

**How it was found.** The release sweep extracted the Ruby test that upstream
added with this fix onto a harness and ran it at the pin and at master
(`ledger.json`, `probe/`). `trigger.rb` is that extraction, unmodified.

**The C-API driver.** At the pin `trigger.rb` does not reach the defect: its
`Task#close` postdates the pin (306d265cd), and the other paths that free a
task's stack are C API, reached by upstream's test only through C helpers in
mrbtest. `capi.c`, written 2026-10-11, takes the path of upstream's
`TaskTest.run_sync` test without the helper (`mrb_execute_proc_synchronously`
with a closure escaping into `$esc`, then a full collection) and is the
case's `trigger`. Natively it crashes at the pin and completes with the fix's
`task.c` hunks applied (`probe/native-control.sh`).

**Host oracle at the pin.** ASan: SIGABRT; with `MRB_HEAP_PAGE_SIZE=1`, which puts
each GC object slot in its own malloc block: SIGABRT. On master: no report. These are
`trigger.rb`'s runs, so they do not show which of the two task defects aborted.

**What the arms read.** See `case.json`. The 2026-10-06 arms ran `trigger.rb`;
the two virtual arms ran `capi.c`, each linked against its own arm's
`libmruby.a`.
