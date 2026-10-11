# tshark safety on QEMU: the level0 and shrink heap arms (2026-09-25)

**Question.** The tshark domain dissects correctly under capability enforcement
(`../2026-09-24-qemu-tshark-staged/`). What does it do with a heap overflow, a use after free, a
stale free, or a stale pointer into a reset wmem scope? This is the plan's Safety milestone for
the two cheap heap arms. The revoking sublet arm is the next step.

The fixtures (`src/tsapp-safety.c`) and their predictions (`host/safety-expect.txt`) were pushed
to `dev` in `fc2ee56` at 07:20, before the first fixture boot at 07:35. The fixture images had been
built at 07:14–07:17 from the committed source. Nothing in the predictions file has changed since.

**One premise in that file was wrong, and the prediction held anyway.** It says fixtures 10–12
return "on BOTH arms, since level0 narrows only the block". level0 narrows nothing: on level0 the
wmem pointers carry the whole 40 MiB arena (`bounds=[a1b96cd0,a4396cd0)` for fixture 10), not
their block. The predicted RETURNs came true on level0 because of the arena bounds, and on shrink
because of the block bounds the premise describes. An audit found this; the file keeps what was
predicted.

## Verdict

**Every counted run is as pre-registered: 72 of 72.** That is 12 fixtures × 2 arms × 3 repeats.
Each cell's three runs are distinct boots of one image. The bytes each boot ran hash to its arm's
`SHA256SUMS`, and every boot used the one host binary.

| fixture | level0 | shrink |
|---|---|---|
| 1 g_malloc(64): the pointer's length | returns; the pointer covers the 40 MiB arena (41,909,248 bytes from its cursor) | returns; **exactly 64** |
| 2 write into the neighbouring object | returns `0xee`: the neighbour was overwritten | **bounds fault** at the neighbour's first byte |
| 3 read one byte past the end | returns `0x90`, the low byte of the free tail block's size | **bounds fault** at `p + 64` |
| 4 read after g_free | returns the freed object's own byte | returns the same: shrink has no temporal safety |
| 5 read after g_free and reuse | returns the NEW occupant's byte, same address | the same |
| 6 stale g_free, then allocate | a later allocation **aliases the live object** (`0x77` written through it) | the same |
| 7 one byte past a global | bounds fault | bounds fault |
| 8 one byte past a stack array | bounds fault | bounds fault |
| 9 a second global through the first | returns the second's byte: the two share one capability | the same |
| 10 stale pointer after a BLOCK_FAST reset (packet scope) | returns the new scope's byte, same address; the pointer carries the arena | the same; the pointer carries the 2 MiB block |
| 11 stale pointer after a BLOCK reset (file scope) | the same | the same; the pointer carries the 8 MiB block |
| 12 write into the next wmem allocation | returns `0xee`: the neighbour was overwritten | the same |

What it says:
- **level0 gives no protection between heap objects.** Every heap pointer is confined to the
  40 MiB arena and covers all of it. Only the compiler's own bounds, on globals and stack arrays
  (7, 8), fault.
- **shrink bounds a g_malloc'd object to the bytes asked for.**
  - Measured for 64-byte objects: the write into the neighbour and the read one past the end
    both fault at the printed address. The fault is in `tsapp_fix_poke` and `tsapp_fix_touch`,
    with p's own 64-byte bounds in the register.
  - By construction only, not measured: other sizes and realloc. `level0.c` narrows to the `n`
    asked for, not the rounded size, in malloc and in both realloc paths. Odd sizes, calloc,
    realloc, a read below the object and aligned allocations were not tested.
  - It changes nothing temporal: freed memory stays readable, and a stale free still lets the
    next allocation alias a live object. free finds its header through the arena capability,
    from the pointer's address, so a stale pointer is enough.
- **Neither arm reaches inside wmem.** A wmem block is one g_malloc, and every allocation carved
  from it carries that g_malloc's bounds:
  - on shrink, the block's: 2,097,104 bytes from the cursor in BLOCK_FAST's 2 MiB block, and
    8,388,560 in BLOCK's 8 MiB;
  - on level0, the arena's.

  So an overflow between two wmem allocations, and a stale pointer after the scope is reset, both
  go unnoticed on either arm. By reserved bytes that is most of tshark's heap: natively, 27.27 MB
  of its 27.77 MB peak is wmem blocks, though they hold only about 1.5 MB (plan, "M0 open items",
  item 4). This is the gap the wmem hooks (`ports/wireshark/wmem`) exist to close, and it is now
  measured rather than assumed.
- **Merged globals share bounds** (9): the compiler put the two static arrays in one
  `.L_MergedGlobals`, as FFmpeg's fixture 10 showed.
- **The shrink arm is a working tshark.** Its M1–M5 all reach their stage and match. Its `-V -n`
  stdout is byte-identical to stock on dhcp, dns_port, http, arp, their flipped copies (each flip
  fires) and dns-ooo, and its stderr is identical to the native minimal build's. ntp's stdout
  differs from stock, as the negative control must. Its M5 heap peak is 28,124,032 bytes, 112
  above level0's in the paired run (one run each; level0's own peak varies by 64 bytes across
  recent boots).

## Method

- **Arms** (`host/build-domain.sh TSAPP_HEAP=level0|shrink`): the same objects, archives and
  link; only `level0.o` differs (shrink: `-DCAPSTONE_LEVEL0_SHRINK=1`). Rebuilding level0
  through the new script gave M1–M5 byte-identical to the images built before it, and shrink's
  M1–M5 differ from level0's only through `level0.o` (`tshark_m5.o` is identical across arms).
- **Fixtures** are linked in `tshark.c.o`'s place, as M1–M4 are: the same runtime, heap, GLib,
  wmem and link as M5. No fixture calls into a dissector.
- **Boots** (`host/run-qemu.sh safety`): at most one predicted fault per boot, and it runs last;
  the runner refuses any other order. At most six fixtures per boot, because every fixture image
  needs a 128 MiB block (the 40 MiB arena is in its `.bss`), and 1792M is the largest cma
  measured to boot. Per repeat: level0 `1 2 3 4 5 7` and `6 9 10 11 12 8`; shrink
  `1 4 5 6 9 2`, `10 11 12 3`, `7`, `8`.
- **Verdict** (`host/safety-verdict.py`): a fault counts only after the fixture's touch line and
  at the address it printed; a RETURN is the exit status on the host's `LT-RESULT` line.
- **Instrument checks, each made to fire:**
  - the verdict's `--self-test` classifies 13 doctored cases correctly, and three deliberately
    broken classifiers each fail it (`instrument-checks.txt`);
  - the first boot held both RETURN fixtures and a FAULT one, and each classified as predicted;
  - the collector hashes the bytes each boot actually ran (kept beside its log) against the
    boot's start-of-boot sidecar and the arm's `SHA256SUMS`. It flags every boot when the two
    arms' lists are swapped;
  - the setup watchdog (below), run with a 1-second budget, fired and powered the guest off. That
    was an earlier version. The final version's "power off only if nothing moved in 30 s" was
    tested in a dry run under busybox on the host: it fires when a file stops growing, and exits
    quietly once setup is done. In a guest it has only been seen reporting a slow copy and letting
    it finish.
- **The compiler** is the port's, `b7b31421e9fa`, the one M1–M5 were measured with. `dev`'s
  compiler has since moved to `3979abd8e9a3` (C-46, C-47). The runtime is `dev`'s, C-64 and
  I-11 fixes included.

## Stalls

28 boots ran in this campaign's window (07:20–08:58), not counting the watchdog's two positive
controls. 21 reached their first section; the other 7 were retried, and none is counted.
`result-lines.txt` keeps every attempt. By where each stopped (`stall-classes.txt`):
- **Setup was slow: 3 boots completed and 3 were cut.** The copy from the 9p share, normally about
  30 s, took about 5 minutes.
  - **The diagnosis.** Once the runner's setup watchdog could report, every slow setup had its
    `cp` waiting in `p9_virtio_zc_request`, a 9p zero-copy read, while the file being written kept
    growing.
  - **The 3 cut boots** came from the watchdog's first two versions, which powered off at 300 s
    whatever the state. One of them showed the copy still moving.
  - **The 3 completed boots:**
    - two ran under the final version, which reports at 300 s and powers off only if nothing
      moved in the next 30 s;
    - one ran under an intermediate version. Its power-off was due 15 s after the report, but the
      boot's last fixture had already ended QEMU.

    Each finished its copy within 30 s of the report, and ran every fixture. Once the watchdog
    could see the copy, it never found one that had stopped outright.
- **The guest stopped in setup with no watchdog output: 2 boots.** One predates the watchdog. The
  other ran under it, so userland was not running either: the guest wedged. Only the outer guest
  budget ends that.
- **The boot stopped between init and login: 2 boots.** The login timeout ended each in 2
  minutes.

So the stall is not only a 9p problem: a guest that crawls or stops before login is the same class
(ISSUES I-12). On the host, load average was about 2.3, and the stalled QEMU was at 100% CPU.

`LT-RESULT … FAIL UNSERVED` on a fixture's line is the host reading bit 8 of the mark (for
example `0x50015b`), not an unserved syscall. The image's own report says `unserved=none`.

## What this does not establish

1. **Temporal safety.** Neither arm revokes anything. The revoking sublet heap is the next step.
   Its predictions will be registered before it runs. The plan already expects 10 and 11 to
   return there too, because a wmem scope reset never calls free.
2. **The board.** The ABI depends on QEMU's fabricated `gp`, and the deployed silicon lets a stale
   access retire (ISSUES Q-11).
3. **tshark's own misbehaviour.** The fixtures create each condition directly. Whether a real
   dissector bug reaches one is a different question.

## Files

- `result-lines.txt`: the shrink gate's stage and oracle lines, then every fixture boot's verdict
  in order, with stalled attempts, and a per-cell count.
- `instrument-checks.txt`: the verdict's self-test, three broken classifiers failing it, and the
  setup watchdog's positive control.
- `stall-classes.txt`: every boot in the window, by where it stopped.
- `SHA256SUMS`: both arms' fixture images and M1–M5, and the host binary every boot ran.
