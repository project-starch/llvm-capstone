# Pre-registration — boot 2 of the vp2 flat-table kit, at module revision `179686bcf198`

**Written and pushed BEFORE the boot.** Cite this commit from whatever reports the result. Nothing
here may be edited afterwards; corrections go in the result.

Sibling pre-registrations in this folder: `PREREGISTRATION-ceiling-row.md` (the 70,000-node ceiling
row, flat arm only, still unrun). This one is boot 2 of the bug-corpus kit and reads nothing about
the ceiling.

## Why there is a new record at all

The board lane asked whether the prediction carries over from the kit built at `2f15ed14b4f6`, or
whether the run is repeated at the module's own revision. It is repeated, because the carry-over
argument does not hold: `module_param(node_flat_records, ...)` and its bound were added OUTSIDE the
flat `#ifdef`, so the **paged** module changed as well as the flat one — `71e6b71826cd8651` ->
`69a7016b6d10cd0a`. Neither arm of the old record covers the modules now in the kit.

Mixed provenance in the kit is deliberate and is recorded so it cannot read as staleness:

| kit status file | SRC | covers |
|---|---|---|
| `status` | `2f15ed14b4f6` | the SDK and the three applications |
| `status-179686bc` | `179686bcf198` | the three adapters and both `run.py` arms |

The justification for not rebuilding the SDK or the applications: `git diff
2f15ed14b4f6..179686bcf198` touches exactly two files — `runtime/virtual/module/capstone_vm.c` and
a `tests/` README — and **zero** files under `build-sdk.sh`, `ports/`, `runtime/include` or
`runtime/application`. The applications therefore keep their recorded hashes and stay comparable
with the superseded record.

## The staged artifacts

| artifact | sha256 (16) | path |
|---|---|---|
| board adapter `capstone_vm.ko` | `ed24f827b5beb373` | `<kit>/out/adapter-board/module/capstone_vm.ko` |
| `sqlite3.dom` | `a4b2c0f05809f433` | `<kit>/out/sqlite/sqlite3.dom` |
| `contract.dom` | `bf433e107032fa8a` | `<kit>/out/contract.dom` |
| `mruby` | `5fee8d8f604229ea` | `<kit>/out/mruby/.../bin/mruby` |

`<kit>` is the v7 kit root on the large disk; the full path is in the kit's own `status-179686bc`.
The board ko was built twice, from two different build directories, and hashed `ed24f827b5beb373`
both times — the FPGA-kernel module does not embed its build path, so that hash is a usable
identity. (The QEMU-kernel modules DO embed it; see "What a module hash is not" below.)

## Gates on the staged module, with their controls

| gate | board ko | control | control reads |
|---|---|---|---|
| `ldc`/`stc` encodings (funct3 3/4 of opcode `0x5b`) | 0 | boot 1's module | `ldc` 2, `stc` 2 |
| CSR `0x5c0` (`scapctl`, absent on vp2) | 0 | paged ko `69a7016b` | 1 |
| CSR `0xcc0` (`urevavail`, absent on vp2) | 0 | paged ko `69a7016b` | 1 |
| RVC (16-bit) encodings | 0 | — | — |
| CSR `0x5c1` (`srevroot`, EXISTS on vp2) | 10 | paged ko | 13 |

Every zero above is a measured absence from a detector shown able to produce the opposite reading
on a sibling artifact. The two CSR gates are the ones that would otherwise be unproven, and their
control is the paged module, where both encodings are present because the paged branch reads them.

Register-form custom instructions in the board ko: 8 — `csretire` x2, funct7 `0x24` x4, `csmint`
x2. Counted independently by the board lane and by this lane, with the same result.

## The reading predicted for boot 2, computed at THIS revision

`run.py` on the flat adapter at `179686bcf198`, record
`<kit>/out/result-flat-179686bc.json`, log `<kit>/runpy-flat-179686bc.log`:

* **status PASS, 41 / 41**;
* 22 `VM_EXIT` lines, **byte-identical as a set** to the superseded `2f15ed14b4f6` record's — same
  case names, same codes (0 for the clean exits, 139 for the nine deliberate-fault cases, 143 for
  `preempt`);
* geometry `node_bytes=1048576 node_capacity=65536 node_growths=0`.

So the re-run and the carry-over argument agree: the record moved, the readings did not. Had they
disagreed, the re-run would have been the answer and the argument discarded.

**The geometry is a readback, not an echo of the invocation.** The chain, for anyone checking:
`capstone_vm.c:120` defines `CV_FLAT_RECORDS` as the parameter; `:1203` publishes it into the live
table header (`table[0] = CAP_REV_TABLE_RECYCLE | CV_FLAT_RECORDS`); `:148` reads it back masked
(`table[0] & ~CAP_REV_TABLE_FLAGS`); `:955` puts that into `stats.node_capacity`. The value printed
is therefore the capacity QEMU itself reads out of the header, not the number passed to `insmod`.
`node_bytes=1048576` is exactly 65,536 x 16, and `node_growths=0` is the flat table never growing.

The kits load the parameter **unset**. Record its loaded value from the stats line above, not from
the invocation.

## A correction that belongs in the record

The source comment and `MODULE_PARM_DESC` at `179686bcf198` both say an over-large
`node_flat_records` is "refused at insmod". **That is wrong for the module staged here.** The bound
sits in `vm_open` (`capstone_vm.c:1174` at that revision); `module_init`'s body contains no such
check and there is no `param_ops` set handler, so at this revision `insmod node_flat_records=131072`
SUCCEEDS and the `-EINVAL` lands on the first open of the device.

This does not affect boot 2 — the kits load the parameter unset, the default is 65,536, and the
check passes either way — but a test keyed to `insmod`'s exit status would read the bound as
unenforced, which is backwards. Any such test must be keyed to the first device open for THIS
module. The fix (bound moved to load time, parameter guarded flat-only) is a later commit and is
deliberately NOT in the staged module.

## What a module hash is not

Recorded here because the kit's own records cite module hashes as identities, and for one of the two
kernels that is unsound. Measured today:

* the **FPGA-kernel** module is reproducible — two builds from different directories, same bytes;
* the **QEMU-kernel** module embeds its build output path in `.rodata.str1.8` (the string is
  `<outdir>/module/capstone_vm.c`, because `build-adapter.sh` compiles a copy inside the output
  directory), and `.note.gnu.build-id` then differs because it hashes the content. At a FIXED output
  path the build is deterministic — byte-identical across consecutive builds — so the rule is not
  "unreproducible", it is **hold the output directory fixed before comparing two QEMU-kernel module
  hashes.** Two builds of identical source into different directories differ, and that difference
  means nothing.

Consequence for this folder: comparisons of flat/paged module hashes across kits are only valid when
the output directory is the same. The board ko is unaffected.
