# memcached allocator case 8 on the Capstone domain arms, with case 5 as the spatial control (2026-10-09)

**Verdict: case 8 is CAUGHT on both modes -- but EARLIER than predicted: in the defect itself, not
at the labelled probe.** The runner reports the two case-8 rows FAIL, and that is this reading,
not a regression; read on before taking the FAIL at face value.

| case | mode | runner | fault | where | access | object bounds |
|---|---|---|---|---|---|---|
| 5 | spatial | OK | cause 7 at `0x1018aa388` | the labelled write probe | 1-byte at `0xe1afff00` | `[0xe1affe60, 0xe1afff00)` = 160 B |
| 5 | sublet | OK | cause 7 at `0x10186a388` | the labelled write probe | 1-byte at `0xe1afff00` | `[0xe1affe60, 0xe1afff00)` = 160 B |
| 8 | spatial | FAIL | cause 5 at `0x1018ab79c` | `mc_case_body` + scan load (offset `0xb79c` from the code capability base; ELF `0x1b79c`) | 1-byte at `0xe4a04000` | `[0xe4a00000, 0xe4a04000)` = 16384 B |
| 8 | sublet | FAIL | cause 5 at `0x10186b79c` | `mc_case_body` + scan load (offset `0xb79c` from the code capability base; ELF `0x1b79c`) | 1-byte at `0xe4a04000` | `[0xe4a00000, 0xe4a04000)` = 16384 B |

## What the case-8 reading is

The case reduces `try_read_command_ascii`'s `while (*ptr == ' ') ++ptr;` at the pin (e8364b5's
parent) over a 16384-byte rbuf (`READ_BUFFER_SIZE`) filled with spaces, with its cache successor
also filled with spaces. The case was written to let that scan run past the object, then `mark(8)`,
then read through the labelled probe at the crossed address.

On both modes the domain halted with **cause 5 inside the scan itself**: the emulator's own fault
line names a 1-byte load at `0x...04000` against bounds `[0x...00000, 0x...04000)` -- 16384 bytes,
the rbuf object exactly -- so the fault is the FIRST byte past the object, taken by the upstream
defective statement. The instruction is `lb` feeding a compare against 0x20 (`' '`), in
`mc_case_body`, at the same image offset on both modes. `mark(8)` never ran, so the serial has
no case marker and the runner has no published probe address to compare against: `expected_pc`
is null and the row is FAIL by its rules, which require the fault AT the probe AFTER the marker.

**The pre-registered prediction is REFUTED on location and HOLDS on outcome.** It said
"faults at the labelled probe once the slab port narrows per object". The port does narrow per
object, so the defect is caught -- one statement earlier than the case's probe, which is the
stronger reading: the bound stops the overrun at the line upstream fixed.

Case 5 is the spatial control in the same build and runner: cause 7 at exactly its published
write probe on both modes, runner OK. The runner's negative control on case 8 fired 2/2.

## Inputs

- QEMU `6e5e136a070b` (the shared capstone-qemu build), the shared buildroot QEMU images.
- `defect-NN.dom` from `ports/memcached/allocators` preset `capstone-domain` with
  `-DMCP_CORPUS_SRC=<case>/case.c` (shared/build-cases.sh's recipe), compiler 7d01722aab88;
  the loader from the port's `linux-guest` preset.
- case 5 spatial: defects.dom 68126ef90c717b45, host.user 5f83ce2da2425d7f
- case 5 sublet: defects.dom 68126ef90c717b45, host.user 5f83ce2da2425d7f
- case 8 spatial: defects.dom 9ae2e10378dd25dc, host.user 5f83ce2da2425d7f
- case 8 sublet: defects.dom 9ae2e10378dd25dc, host.user 5f83ce2da2425d7f

## Result lines

    OK   case=5 item-data-one-past                 spatial  cause=7 pc=0x1018aa388 expected=0x1018aa388
    OK   case=5 item-data-one-past                 sublet   cause=7 pc=0x10186a388 expected=0x10186a388
    FAIL case=8 ascii-all-spaces-scan-past-rbuf    spatial  cause=5 pc=0x1018ab79c expected=None
    FAIL case=8 ascii-all-spaces-scan-past-rbuf    sublet   cause=5 pc=0x10186b79c expected=None
    case=5 spatial  Cap mem access OOB: insn=00a58023 pc=1018aa388 pcc_base=1018a0000 va=a388 addr=e1afff00 size=1 bounds=(e1affe60, e1afff00)
    case=5 sublet   Cap mem access OOB: insn=00a58023 pc=10186a388 pcc_base=101860000 va=a388 addr=e1afff00 size=1 bounds=(e1affe60, e1afff00)
    case=8 spatial  Cap mem access OOB: insn=00050503 pc=1018ab79c pcc_base=1018a0000 va=b79c addr=e4a04000 size=1 bounds=(e4a00000, e4a04000)
    case=8 sublet   Cap mem access OOB: insn=00050503 pc=10186b79c pcc_base=101860000 va=b79c addr=e4a04000 size=1 bounds=(e4a00000, e4a04000)
    negative control (case 8, both modes): 2/2 oracles fired

