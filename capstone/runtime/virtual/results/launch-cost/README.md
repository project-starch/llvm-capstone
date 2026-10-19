# Launch cost: the image snapshot

What the write denial (`CV_IMAGE`, see [the virtual runtime's README](../../README.md))
replaces, measured on one job of a SPEC CPU2017 test workload: 502.gcc_r's
`t1.c -O3 -finline-limit=50000`, run on QEMU's virtual profile, its output
validated by SPEC's own comparison. The unit is dynamic instruction attempts
over the whole job (fork, exec through `capstone-vexec`, the program, exit; every
privilege level), counted by the capprof TCG plugin in six windows that the
measured launcher marks: exec, snapshot, load, grants, run, teardown. Not time,
not cycles. [`gcc-test-job.json`](gcc-test-job.json) holds every repetition's
per-window, per-class counts (kernel, module, launcher, program, firmware) and
the input hashes.

| window | dev | dev + write denial |
|---|---|---|
| exec | 0.59M | 0.58M |
| snapshot | 55.26M | 0.62M |
| load | 33.18M | 32.96M |
| grants | 42.56M | 42.56M |
| run | 69.37M | 69.16M |
| teardown | 45.92M | 28.99M |
| whole job | 246.9M | 174.9M |

Medians of three fresh guests each; the totals are 246.9, 247.2 and 246.2M
against 174.9, 174.9 and 174.7M, every run VALID. The snapshot window loses the
copy into the sealed memfd (55.3M to 0.6M) and teardown loses the release of the
copy's pages (45.9M to 29.0M). Load falls by 0.23M, all of it in the kernel's
share (26.18M to 25.98M) and more than its repetition spread of 0.02M; exec and
grants are unchanged, and run's 0.21M difference lies inside dev's own spread
of 0.81M.

Arms (p12, one hart, 2026-10-18):

- **dev**: launcher and module from dev `b456c1b4`, built with the window marks
  and the profiling hooks the plugin reads; the monitor with the SBI RFENCE
  extension (caplifive-sbi #4; without it every newly mapped code page costs a
  kernel warning, in both arms alike).
- **dev + write denial**: the same, plus `runtime/image-write-denial` at its
  first commit (`68d794e3`, the launcher choosing denial or copy from
  `fstatfs`). The image ran from tmpfs, where the module-side decision of the
  second commit (`0f4ad10a`) takes the same path.
- Byte-identical across the arms: the program, the plugin, QEMU, the kernel,
  the firmware and the root filesystem (hashes in the JSON). The program is one
  gcc binary built by the profiling kit's toolchain, an integration build of the
  compiler with the frame-pointer default, `-fno-math-errno` and the gp
  constant-pool fold (since opened as #266, #267, #268), linked against a
  runtime with the memset and memcpy word paths (#269, #270). The same binary in
  both arms, so the difference is the launcher's and the module's alone.

The plugin and its report tool live on the profiling lane's branch, not yet on
dev; the JSON reduces each window's report to its per-class totals, which is
what the table sums. The SPEC sources and the kit stay outside this repository.
