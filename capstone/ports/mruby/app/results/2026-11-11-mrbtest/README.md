# mrbtest with patch 0008 on the one-operand lifetime instructions (2026-11-11)

mruby 4.0.0-rc2's own test suite on the virtual profile, on the release image and on the protected one
(`MRBD_SUBLET=1`, patch 0008 converted to the one-operand `CREVOKE`), against the native build of the same
configuration. `mrbtest.json` has every run.

| build | run | exit | what arrived of the summary | failed tests |
|---|---|---|---|---|
| native (both builds) | | 0 | Total 1710, OK 1701, KO 0, Crash 0, Skip 9 | none |
| protected | first of a VM | 0 | Total 1710, KO 0 | none |
| protected | second of that VM | 1 | Total 1710, KO 17 | mruby-io only |
| release | third of that VM | 1 | OK 1681, Crash 6, Skip 9 | mruby-io only |
| protected | first of a fresh VM | 1 | Total 1710, KO 17 | mruby-io only |
| protected | second of that VM | 1 | OK 1681, Crash 6, Skip 9 | mruby-io only |
| release | first of a fresh VM | 1 | OK 1698, Crash 0, Skip 9 | mruby-io only (`FileTest.size?`) |
| release | second of that VM | 0 | OK 1701, Crash 0, Skip 9 | none |

**No run on either image has a capability fault.** Each image has one run that exits 0 with native's result, and each
has runs in which tests of the mruby-io gem fail. Every failing test is one that reads the suite's own scratch file
and finds it empty (`Expected: "mruby io test\n"`, `Actual: ""`; `FileTest.size?` nil; `EOFError`). That is the
failure the 2026-10-11 record had once per run on both images (`File#path`, "a file the guest gives back empty").
Here it comes and goes between runs of one image, and it is on the release image, which has no patch. So the suite
does not separate the two images, and it does not pass as a gate on this platform under this load either. Whether
unmodified dev shows the same was not run.

The guest's console drops lines of a long output, as on 2026-10-11, so a run's summary is given as far as it arrived;
the exit status is the suite's own (1 when a test failed or crashed).

A run took 397 to 1,012 s of host time, where 2026-10-11 has about 55 s. The host's load average was between 47 and 77
on 80 cores, with up to 24 QEMU sessions of the same campaign.

- Tree: branch `one-operand-crevoke-crenew` at 59201bdf2, QEMU at ea57497908. Patch 0008 is 31c512ae806b...; it adds
  58 lines of code by the count the port's README uses (lines that are not blank, comment or conditional directive).
- Builds: `build-mruby-domain.sh` with `MRBD_PIN=4.0.0-rc2`, `MRBD_TESTS=1`, `MRBD_SDK` naming a copy of the port SDK.
  `mrbtest` images: release `20b5f225737e375d`, protected `d252e1802cae1bd9`.
- Platform: QEMU 2d5a71078904..., launcher 0f8e0f903725..., module e47bca62addf... loaded without parameters (pinned,
  one context), `capstone_vm up --profile virtual --exact-bounds --cpu-prop x-capstone-test-csrs=true`, each run
  through `ports/common/application/run.py ... mrbtest -v`.
