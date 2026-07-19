# Questions for Jason — Capstone domain benchmark on CapliFive CVA6 (captype-fixed)

Context in one paragraph: we replaced the benchmark's glibc Linux controller with a
freestanding soft-float one (your "use soft float" steer) — that cleared the `fsd` hang, and
on `working-caplifive-captype-fixed.bit` it now boots, `insmod`s `capstone.ko`, creates the
domain, and maps both regions. The domain `cscall` is finally reached — but the core wedges
at the domain's **own entry** (vaddr `0x10044`, the `<test>`/`_start` glue, right before
`domain_main`): sometimes it resets to the bootrom, sometimes it spins there. We're about to
instrument the monitor's M-mode `mtvec` to read the exact trap cause, but a few answers from
you would save shared-board time and possibly settle it outright.

## Highest priority

1. **How do you actually run these benchmarks on the board?** As **bare-metal `.dom`
   domains** launched directly, or via a **glibc Linux userspace controller** (ioctl
   `/dev/capstone` → `modcapstone` → SBI), the way our benchmark is structured? If you never
   use a Linux controller on the FPGA, our whole controller layer may be unnecessary — we'd
   rather match your harness.

2. **At the domain `cscall`/`csreturn` boundary, is software expected to issue a `fence.i`
   (or any icache/TLB maintenance)?** We found the CVA6 domain switch does no icache
   invalidate, and the reference monitor issues no `fence.i` before entering freshly-placed
   domain code — so the domain's first fetch at `0x10044` may be against a stale icache. On
   this bitstream, **does `fence.i` actually flush the L1 icache?** (We've seen icache-coherence
   trouble before around RFENCE/insmod, so we're unsure it does.)

3. **Is the domain entry glue we emit the same as your reference domains?** Ours is:
   `_start` reads capability CSR `0x4` (stack cap) → `lcc`/`scc`/`delin sp` → jumps to entry;
   entry does `delin gp`, sets up the frame, then a `.capstone_cap_init` loop that `cjalr`s
   to capability-global initializers **before** `domain_main`. Does anything in that prologue
   need something the monitor must set up first (e.g. a valid stack/gp cap, a `fence.i`), or
   differ from the buildroot example-program domains you mentioned?

## Useful to confirm

4. **Which firmware image + DTB is the intended pairing for `working-caplifive-captype-fixed.bit`?**
   We build `--mode fpga` via caplifive-system as a **UP (`CONFIG_SMP=n`)** OpenSBI FW_PAYLOAD
   (the SMP kernel floods the console with `remote fence ... not available in SBI v1.0` and
   buries the login prompt; an earlier SMP `--mode fpga` build also load-access-faulted reading
   the 8250 LCR at `0x1000000c`). Is UP + our built payload the right approach, or is there a
   canonical captype-fixed image/DTB we should use instead?

5. **On the reference FPGA monitor, are these behaviors expected?** (a) S-mode FP is left
   **off** (`mstatus.FS` never written), so Linux userspace FP traps; (b) `medeleg` does **not**
   delegate illegal-instruction (bit 2) to S-mode; (c) `handle_exception` `while(1)`s on any
   unhandled cause. If any of these is a known gap vs your working setup, that would explain a
   lot of our earlier hangs.

6. **When did the board's resident bitstream last change?** We found it had been overwritten
   with stock `ariane_xilinx.bit` (no capability unit) at one point — we re-flashed your
   `captype-fixed`. If your team re-flashes it, a heads-up would keep our evidence clean.

## What we'll do regardless

Instrument the monitor's M-mode `mtvec` to dump `mcause/mepc/mtval` at the `0x10044` wedge, so
we can tell a stale-icache fetch fault (→ `fence.i` at the domcall boundary) from a capability
violation (causes 25–28 → RTL vs monitor). Answers to 1–3 could shortcut this.
