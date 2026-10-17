# Result: the refusal diagnostic fires, and every pre-registered prediction holds

Control for `PREREGISTRATION-resolve-refusal-sites.md`, run on QEMU before any board boot.
Module `e8e65536cd342609` (QEMU arm), v9 images, `supervisor_slots` default 1, `svadu=false`
wrapper — the same configuration as the preceding run, so the diagnostic was the only variable.

## Every pre-registered prediction held

| pre-registered | read |
|---|---|
| `34 / 41` | **34 / 41**, `run.py rc=1` |
| the seven false checks | **exactly** `threads, vm, heap-threads, threads_output, vm_contract, heap_threads, shared_heap_pages_released` |
| the same `EXITS` line | **byte-identical** to the preceding run, including `vm:1` and `preempt:143` |
| banner present | **present**: `capstone_vm: loaded, diag resolve-sites-2, supervisor_slots=1` at 3.38 s |
| **no `remap` line** (the store passes under QEMU) | **zero** `remap` lines |

The pass count and false set are unchanged, which is the result that matters: the change is
observation-only, so any movement there would have been a defect in it rather than a finding.

## The instrument is proven to FIRE, with correct values

Six refusal lines, all at site `vma`, all on the suite's **deliberate-violation** cases. This is
the positive control, and it runs through the production harness rather than beside it:

```
capstone_vm: resolve refused at vma: addr 0x3fa0bac000 cause 15 need 0x2 pc 0x3fa0b13254 \
    prev(addr 0x3fa0bac000 cause 15 rep 0) found 1 flags 0x100071 file 0
capstone_vm: resolve refused at vma: addr 0x3f9c734000 cause 13 need 0x1 pc 0x3f9c6934d0 \
    prev(addr 0x3f9c734000 cause 15 rep 0) found 1 flags 0x100070 file 0
```

Decoded, the values are right for their cause in both directions:

- `flags 0x100071` = `VM_READ | VM_MAYREAD | VM_MAYWRITE | VM_MAYEXEC | VM_ACCOUNT` — `VM_WRITE`
  **absent**, against `need 0x2` = `VM_WRITE`. A store to a read-only mapping: `vm-ro-write`.
- `flags 0x100070` = the same **without** `VM_READ`, against `need 0x1` = `VM_READ`. A load from a
  `PROT_NONE` mapping: `vm-none-read`.

So the line names the right site and gives the reason, on two different causes, in a run whose
other 34 checks pass. A detector that fires only on cases that must trip it, and stays silent on
the 34 that must not, is the two-sided form.

## What this does and does not establish for the board

It establishes that the lines compile, fire, carry populated fields, and print at a level the log
retains; and that the banner distinguishes "level dropped" from "did not fire". It does **not**
test the hypothesis: `vm` exits **1** under QEMU because the store *passes* there, which is why no
`remap` line appeared and why it was pre-registered not to. Only the board can produce the
terminal store.

## A defect in my own reporting script, recorded because the shape recurs

The run's status file printed `BANNER=0 REFUSALS=0 DIAGTAG= WRAPPER_FIRED=0` while the serial log
held `banner=1` and six refusals. The counters grepped `runpy.log` — which holds only `run.py`'s
end-of-run summary — instead of the `serial.log` under the per-run `virtual-app.*` directory, and
`wrapper.log` was not at the path the script assumed once `CAPSTONE_TMP_ROOT` was set. Nothing was
wrong with the module or the run; the reporting layer pointed at the wrong artifact and rendered
"no data" as a zero.

Had the serial log not been read directly, the honest reading of that status file would have been
"the diagnostic did not fire" — the exact inversion of what happened, and it would have been
reported as a finding. This is already covered by the standing rules ("make *no data* an ERROR,
not a zero"; a positive control must run through the harness), so it is filed as an instance
rather than a new rule. The operational habit it reinforces: **read the primary artifact, not the
summary your own script wrote about it.**

`svadu=false` application is evidenced by the live QEMU command line read from `/proc` during the
run (`-cpu rv64,...,svadu=false`), not by a persisted artifact, since the `-cpu` string is not
echoed into the guest serial log; the identical 34/41 and `EXITS` match against the preceding
`svadu=false` run corroborates it.

Serial-log fault counters for this run: `cause=15` ×1, `cause=25` ×3.
