# h264_refs: a reset bounded by the count, not by the array

`H264Ref` carries `data[3]`, pointers into a picture's planes, and `parent`.
When the reference list shrinks, the reset clears entries `[len, ref_count)`.
Every record from `ref_count` to the array's 32 slots keeps whatever it held,
and those pictures have since been returned to the pool.

The arms differ in the memset bound and nothing else.

    arm=fixed survivors_past_len=0 reuse_same_address=0 stale_read=0x00
    arm=buggy survivors_past_len=2 reuse_same_address=1 stale_read=0xCC

Two records past the cleared range survive in the buggy arm, the pool reissues
that storage at the same address, and the record still names it. The fixed arm
clears the whole array, so nothing survives to name anything.

## Paired arms in a Capstone domain

Case 37 of the port's pool lifetime probes:

| mode | outcome |
|---|---|
| 0 spatial | **completes** — the read through the uncleared record returns `61`, the reissued storage's new byte |
| 2 Sublet | **faults**, cause 24, at the published `ff2_probe_read` address |

    bash security-tests/qemu/run.sh <out> --cases 37 --modes 0,2 --rounds 1
