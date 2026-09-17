# Board drivers, chains and invocation lists

The board lane's execution machinery, committed 2026-09-15 for the handover to another machine
(`docs/plans/2026-09-15-board-lane-handover.md`). Until then these lived in `~/capstone-artifacts/unify/`
and a session scratchpad with hard-coded paths; every path is now a knob with the old value as its
default, and the measurement logic is byte-for-byte what ran the boots the measurements doc cites.
Read the `board-run` skill first; it is the decision procedure these scripts implement.

## What runs a boot

| script | what it does | mandatory knobs |
|---|---|---|
| `board-r1e4.sh` | **the live entry point**: bake (three rebuilds under the memory lock) → stage the R1 harness image and the readback host → control (`lpc\|k800`) → the boot's 12 invocations from the list → control → summary through `fpga_driver.transcript` → marker (`done` / `failed` / `refused`) → restore-bake in the exit trap | `R1_BOOT` (1-based), `R1_IMG`, `R1_HASH` (sha256/16), `R1_LIST`, `R1_QEMU_LOG`, `R1_HOST` (the readback host) |
| `chain-r1.sh` | N boots of `board-r1e4.sh` over one list, ends with `CHAIN_<TAG>_DONE` | `CHAIN_TAG`, `CHAIN_BOOTS`, the six mandatory `R1_*` below, `R1_BOOT_TAG`, `R1_BOOT_DESC`, `R1_PREREG` |
| `chain-m1.sh` | the no-reclamation baseline: three boots at the diagnostic capacity (three repetitions of each of the four patterns per boot = nine per pattern, the measurement standard's five-over-three with margin) plus one deployed-table boot; prepared and NOT launched | `M1_IMG`, `M1_HASH`, `M1_QEMU_LOG` |
| `qemu-m1-flowcheck.sh` | the emulator flow check of the M1 series (writes the qemu-pass record) | `R1_DOM`, `R1_HOST` |
| `board-c6var.sh` | F1's boot shape: control, a cell-6 variant image at the 2 MiB arena, control, probe | `C6_TAG`, `C6_IMG`, `C6_HASH`, `C6_QEMU_DEFAULT`, `C6_QEMU_DEFAULT_LOG`, `C6_QEMU_2MIB`, `C6_QEMU_2MIB_LOG`, `R1_HOST` |
| `board-b80s.sh`, `board-b80a.sh`, `board-b80b.sh`, `board-b79.sh`, `board-b78-w2h.sh` | the P1 -O2 cells, the `--stats` boot, E2's boot and E1's per-arm boot — **references for the record shape**; their images live in a scratchpad that is gone, so they fail loudly on the named knob (`CELL_IMG`, `NATIVE_BASELINE`, `QEMU_LOG_*`, `E1_DIR`) rather than on a dead path | see each header |

**`board-r1e4.sh`'s six mandatory variables, in one place** (each is `:?`, so a missing one aborts before
anything is staged): `R1_BOOT`, `R1_IMG`, `R1_HASH`, `R1_HOST`, `R1_LIST`, `R1_QEMU_LOG`. A chain script must
set or forward every one of them; `chain-m1.sh` omitted `R1_HOST` and lost four boots to it in three seconds
on its first real execution (2026-09-15 — no board time, it fails above the bake), which is why both chains
now check all six at their own first line.

Optional knobs, every driver: `CAPSTONE_ARTIFACTS` (default `~/capstone-artifacts`; results land in
`unify/board-<tag>/`, pass records in `qemu-pass/`), `CAPSTONE_FPGA_URL_FILE` (default
`~/.claude-kisp/secrets/fpga-console-url`; never echoed, never committed), `MEMLOCK` (default
`~/bin/logs/machine-memory.lock`; a bake waits on it), `CAPSTONE_BR_FW`, `CAPSTONE_BR_TARGET`,
`CAPSTONE_BR_OVERLAY`, `CAPSTONE_BR_BUILD`, `CAPSTONE_KO` (the buildroot per-target layout: defaults are
`build-fpga/...` and `overlay/test-domains`, which is what both the original host and the apollo server have),
`ENTRY_STALL_S` (420; the JTAG upload is up to 227 s of legitimate silence), `WD_IDLE` (900),
`R1_HOST_HASH` (default `2c9e82d101b48160`, the host these drivers were run with; a correct rebuild reproduces it — see the recipe below before overriding).

## Gates the drivers keep (do not weaken any of them)

* the monitor at `2dcd3a5` and the FPGA buildroot copy at `d04bd83`, checked by commit;
* the image hash equals the knob, and `${CAPSTONE_ARTIFACTS}/qemu-pass/<sha256>` exists (the emulator
  pass, written by `sublet/r1/run-r1-qemu.sh` on `R1_RC=0`), and the emulator log holds the gate line;
* the readback host's hash, and its `RR/share` probe string; `lpc` on the overlay at `3b93a2b6e2adfa36`;
* the baked `.ko` carries the `#3` marker; the initramfs membership check; `preflight-board-run.sh`;
* no board runner live (`pgrep -f 'python3 -m fpga_driver'` inside the driver: the overlap guard);
* the marker agrees with the exit status (ISSUES M-11): `refused` on a preflight BLOCK, `failed` on
  any other non-zero, `done` only on rc 0 and a parsed transcript.

## Invocation lists (`lists/`)

One line per harness invocation: `run arm series pattern arena extra...` — `run` the repetition number,
`arm` S/P/D (or the M1 pattern name), `series`, `pattern`, the arena in bytes, then the harness's own
arguments. **`board-r1e4.sh` takes list lines `12*(R1_BOOT-1)+1 .. 12*R1_BOOT`**, so `R1_BOOT` is an
index into the list, not a boot counter: a single-line list is always slice 1, and a boot that reuses a
slice must be given a different `R1_OUT_TAG` or it writes into the earlier boot's directory (never
overwrite a run — `EXECUTION.md`). A boot with **no** invocations in its slice is refused, so there is no
control-only boot: the smallest real boot is control, one invocation, control.

`f5-chase.txt` is M2's 45-line seeded permutation (four boots); `m1-diag.txt` is the baseline's 36 lines
(three boots × three repetitions of each of the four retention patterns) and `m1-capacity.txt` its single
deployed-table run; `e4-calibration.txt` and `f4-linear.txt` are E4's and F4's.

## Pushing a board result

`precommit-scan.sh --msg` and its staged-diff mode are unaffected by any of this. `--range` scans author
and committer identity lines DELIBERATELY (`precommit-scan.sh:49-56`: a cherry-picked collaborator commit
carries a name) and drops only the committing user's OWN configured identity, guarded by
`if [[ "$ME" != " <>" ]]`. Two consequences, both measured 2026-09-15:

* **Set `git config user.name` and `user.email` on a new host before the first commit.** Without them
  `git commit` refuses outright, and `ME` collapses to `" <>"` so the filter is bypassed by design and
  every identity line in the range is fed to the denylist.
* **With an identity configured, a range blocks when it reaches commits authored by a DIFFERENT
  identity.** On the reference host `--range dev~40..dev` is CLEAN while `--range dev~70..dev` BLOCKS,
  22 hits, every one of them an identity line the scan emitted itself, no message or diff text. A board
  result's push range is `origin/dev..HEAD` — your own commits — and stays clean. If a range does reach
  the other identity, that is the project lead's open ruling: **do not weaken a pattern, do not bypass
  the gate, and do not push with `--no-verify`.**

## Inputs the successor must produce (a rebuild is a new hash; cite the new one)

* the R1 harness image: **`DOMAIN_BASE_VA=0x410000`** `OUT_DIR=... R1_OPT=-O1 bash capstone/sublet/r1/build-r1-silicon.sh`
  (ends `VERDICT: fits`); then the emulator pass `R1_DOM=... R1_HOST=... OUT=... bash capstone/sublet/r1/run-r1-qemu.sh "<args>" <arena>`.
  **The base VA is not optional for a board image.** The build script defaults to `0x10000`, which is where the
  `k800` control rung lands, so a default-base harness collides with the control and preflight C15 refuses the
  boot before the board is touched (R-3 hangs a second domain at a reused entry VA). Every recorded R1/E4/F4/M2
  board image was built at `0x410000`. An image validated only on the emulator may carry the default, because
  nothing is staged beside it there — which means **the emulator-validated build and the bootable build are then
  not the same file, and the bootable one needs its own emulator pass and its own hash**;
* the readback host `sqlite_host_rr.user`: `capstone/ports/sqlite/build-sqlite-host.sh` from
  `sqlite_host.c` with the module's `libcapstone.c` and **BOTH defines**:

      HOST_EXTRA_DEFS="-DSQLITE_HOST_REVOKE_RESHARE=1 -DSQLITE_HC_REGION_SIZE=65536"

  With both, the build reproduces `2c9e82d101b48160` byte-for-byte (verified on a second host,
  2026-09-15), so `R1_HOST_HASH` needs no override. Each define has a SILENT failure mode, and neither
  is caught by the driver's `RR/share` check:
  * without `-DSQLITE_HOST_REVOKE_RESHARE=1` the probe behind its `#ifdef` compiles out and the binary
    has zero `RR/share` strings — a clean build that fails the gate with no explanation;
  * without `-DSQLITE_HC_REGION_SIZE=65536` the host declares the 4 KiB default
    (`sqlite_hostcall.h`; `build-sqlite-host.sh:38` states the rule in a comment) while the harness
    treats `meta->result` as the size of its whole output sink (`sublet/r1/r1_slots_pools.c:48-60`).
    The domain runs correctly and its OUTPUT IS TRUNCATED: on the M1 flow check every arm emitted
    exactly 3,965 bytes from `R1 start` to the resume, cut mid-word at a different field each time, so
    the terminal `R1 m1 end` line — emitted after the loop — never appeared and the driver's emulator
    gate read 0 on all four arms. **A `grep -c` of the end line cannot separate that from an early
    exit; a byte count can** (identical lengths across arms with different content is a fixed buffer,
    not a program failure). Both builds read 4 `RR/share` strings, so that gate never separated them.

  Record the defs in the host's provenance file beside source, libcapstone sha, buildroot commit and
  dirty count. A host built differently is a new hash, passed by `R1_HOST_HASH`, and needs its own
  emulator pass;
* the control rung `lpc`: PINNED, `artifacts/lpc` (`3b93a2b6e2adfa36`, see `artifacts/README.md` — it cannot
  be rebuilt); `k800.dom`: `capstone/tests/runtime-qemu/silicon-ladder/` and the ladder's verify step
  (`verify-and-stage-rung.sh`), which writes the preflight's oracle and `.qemu-pass` under `PREFLIGHT_ORACLES`
  (default `/tmp/capstone/ladder-fpga`) — regenerate, never copy;
* the monitor: the drivers pin the WRAPPER copy `components/opensbi/lib/sbi/capstone-sbi` at `2dcd3a5`, held as
  a checkout ahead of its parent's gitlink (`components/opensbi` records `2c49c41`; the bake reads the working
  tree); the PACKAGE copy `package/capstone-sbi-domain/capstone-sbi` stays at `2c49c41` — that is how the
  reference host holds the two checkouts (memory `opensbi_monitor_rebuild_include_wrapper`). **The pin moved
  from `4274268` to `2dcd3a5` on 2026-09-17** — the D3 writeback fix, a PREREQUISITE for the bitstream that
  makes capability exception delivery live. The old monitor's `add t5, sp, t5` was harmless only because the
  exception was being dropped; on the new silicon it is delivered inside the trap handler at every `rdtime`.
  Verify the rebuilt firmware with `tests/monitor/scan-integer-bases.py`: exit 0 and `cap_text 0` is the
  passing shape, exit 1 names the site that remains;
* the SQLite cells for F1: `capstone/ports/sqlite/build-sqlite-silicon.sh`.

## Reading a run

`unify/board-<tag>/`: `driver.log` (the framed runner log — the record; read it ONLY through
`fpga_driver.transcript`, ISSUES M-11), `boot.txt` (the run-scoped UART window: the tracked raw record for a
bundle), `watchdog.log`, `fw.sha`, `marker`, `r1-lines.txt`, `log`, the six bake logs, `retired/`. The
runner and watchdog logs are never committed (build-host identifiers in banners); bundles carry
`*-boot.txt` and result lines. Cite every result by the image hash from the run-scoped transcript.
