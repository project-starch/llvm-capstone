# Privileged tag transport and unpinned page lifecycle

This is the supported **copyable-only QEMU/Linux prototype contract**. It is
not an RTL specification or a claim of general Linux process compatibility.
The Linux patch, module and matching processor model form one implementation.
A module alone cannot safely replace Linux's page-copy and page-reuse hooks.

## Authority boundary

`CLDRAW`, `CSTRAW` and both `CPTAG` modes are permitted only when the virtual
profile and raw transport feature are enabled, the caller is ordinary S-mode
or M-mode, virtualization is off, and no protected capability context is active.
U-mode, VS-mode and protected applications receive illegal-instruction faults.
Privilege rejection precedes memory access, including for invalid operands.
Linux and its kernel module execute these operations in S-mode. M-mode firmware
also has permission; this implementation does not restrict them to S-mode alone.
There is no userspace raw-tag syscall or ioctl.

A raw load transports the 128 stored bits and their tag as opaque register cargo.
It does not create a usable capability, check liveness, consume the source, or
mint new authority. A raw store republishes that cargo, including a stale tag;
normal capability use must still pass bounds, permissions and lifetime checks.
Scalar register writes destroy opaque cargo. Ordinary memory writes clear storage
tags. Both raw memory operations require aligned ordinary RAM and obey the
current privilege's translation and memory permissions. Device memory is refused.

`CPTAG` takes a physical page address. Query reports a conservative may-hold-tags
summary. Acknowledge clears the summary only after verifying that no storage tag
remains; it neither clears a live tag nor transfers page ownership. Tag publication
and summary operations synchronize in the model. Linux must independently own or
stabilize the relevant page. These operations make the OS part of the trusted
page-maintenance boundary; they do not protect against a malicious kernel.

## Linux and module responsibilities

The kernel activates tag-aware page maintenance only after validating the
`capstone,unpinned-v1` platform declaration, supported ID width, bounds mode,
4096-byte page size and 16-byte granule size. It refuses activation with multiple
possible CPUs, KSM or transparent huge pages. `captag=off` also disables it.
`/sys/module/captag/parameters/enabled` reports actual activation, read-only.
Counting possible CPUs prevents later hotplug from invalidating a one-hart check.

The `copy_highpage` and `copy_user_highpage` hooks preserve bytes and tags.
Tag-free pages use the ordinary copy path. The free hook clears surviving tags
before a physical frame is returned for reuse by any owner, and verifies empty
acknowledgement. Swap of possibly tagged pages is refused: no swap-side tag store
exists. The optional bulk path remains default off and is separately qualified.

The module validates the `capstone,copyable-v1` grant profile and obtains the
kernel's activation/format result through `capstone_unpinned_config`. It rejects
missing or mismatched support and independently refuses multiple possible CPUs
for `unpinned=1`, before exposing its device. Application pages use Linux fault
handling instead of persistent pins; supervisor frames remain pinned. Grants,
mapping permission checks, lifetime revocation and recycling sweeps remain.

## Interrupt and concurrency contract

The current raw-load/store pair temporarily carries metadata that an ordinary
integer trap frame cannot preserve. Linux masks local supervisor interrupts for
the copy. With the qualified ordinary OpenSBI firmware, a machine timer interrupt
is detected by the STIP transition and the copy is retried. The source must stay
stable during this interval. This is a tested one-hart firmware protocol, not a
solution for arbitrary machine interrupts, another firmware, SMP or DMA races.

Supported copy/reuse tests do not establish general shared-memory authority
transfer, protected fork, tagged swap, THP, KSM, or DMA ownership. Those need their
own designs and qualification. Passing application benchmarks does not lift
these restrictions.

## Executable acceptance checks

Source `capstone/tests/capstone-test-env.sh` before building or testing.

- QEMU `tests/copyable-only/run.sh`: removed ownership/control instructions fault
  in M/S/U; linear mint fails before allocation/publication; copyable mint works.
- QEMU `tests/copyable-only/raw-boundary.sh`: M/S tagged copy and acknowledgement;
  four precise operation denials in U/VS and with either required feature off.
  The tests use no legacy instructions or test-only CSRs.
- `platform.py`: four protected-application operation denials, raw feature off,
  kernel feature off, QEMU's two-hart refusal, and Linux's independent two-hart
  refusal under a synthetic unsupported platform declaration. It checks both the
  kernel's activation state and module refusal; boot failure is not a substitute
  for the kernel refusal oracle.
- `run.sh`: positive runtime cases, precise bounds/lifetime/protection faults,
  old ABI refusal, timer wakeup and module unload in both backing modes.
- The page-copy/reuse, interrupted-copy and COW movement gates in `README.md`
  remain required lifecycle evidence. The COW stimulus is not protected fork.

Build `raw-op.c` with the same SDK as the application contract, then run:

```sh
"$SDK/capstone-cc" -O2 \
  capstone/runtime/virtual/tests/copyable-only/raw-op.c -o "$OUT/raw-op.dom"
python3 capstone/runtime/virtual/tests/copyable-only/platform.py \
  --qemu "$QEMU" --images "$IMAGES" --adapter "$ADAPTER" \
  --probe "$OUT/raw-op.dom" --firmware "$FIRMWARE" --out "$OUT/platform"
```

The host needs `dtc` (or pass `--dtc PATH`). All result directories must be new.
The synthetic device tree is confined to the negative test. Production runs use
the model's actual device tree and ordinary OpenSBI.
