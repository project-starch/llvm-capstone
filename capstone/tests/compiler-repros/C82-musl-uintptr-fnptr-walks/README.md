# C-82 — the positive control that REFUTED the exposure claim

`c82-destructor.c` exists to make C-82's `.fini_array` site reachable. It did its job by proving
the site cannot be reached, which is the opposite of what it was built for and the reason it is
kept.

## What it is

One `__attribute__((destructor))`, and a `main` that prints before returning so a fault at exit is
distinguishable from a fault earlier. The ordering matters: `C82_MAIN_OK` present with
`C82_DTOR_RAN` absent and a fault would be the defect; `C82_MAIN_OK` absent would mean the run says
nothing about C-82 at all.

## It loads the gun

Built with the v9 SDK, image `a250d387ddfd352d`:

* `__fini_array_start = 0x29270`, `__fini_array_end = 0x29280` — **16 bytes apart, one entry**, so
  the `bgeu` in the walker cannot skip the loop, unlike every shipped image where all four
  array symbols sit at one address;
* the defective sequence is in the image at `0x20de0` (`addi s1, s1, -0x10; mv a0, s1;
  ldc a0, 0(a0)`).

## The reading

On the strict-CSR pin, module `c4440adde0d02347`:

```
C82_MAIN_OK
C82_DTOR_RAN
C82_EXIT:0
```

No fault. The destructor ran.

## Why

musl's `libc_exit_fini` at `0x20d5c` has **zero call sites** — its one appearance in the
disassembly is its own definition header. The runtime supplies a **strong** `__libc_exit_fini` at
`0x10770` which overrides musl's `weak_alias(libc_exit_fini, __libc_exit_fini)`, and that override
is capability-correct (`cincoffsetimm`/`movc`/`scc`/`stc`), like the startup walker
`hc_run_init_array` at `0x10be0`. musl's `__libc_start_main` is not linked at all.

## Keep it as the gate's control

When the RTL lane's integer-ALU TSV row lands and the decode gate turns the candidate into a
refusal, this image is the artifact that must still be *examined*: it carries the pattern, so the
gate must report it, and the runtime must still exit 0. A gate that has only ever seen images
without the pattern is unproven, and an image that carries the pattern while running correctly is
the only thing that distinguishes "reports the pattern" from "predicts a fault".

Rebuild: `<kit>/out/sdk/capstone-cc -O1 c82-destructor.c -o c82-destructor.dom`.
