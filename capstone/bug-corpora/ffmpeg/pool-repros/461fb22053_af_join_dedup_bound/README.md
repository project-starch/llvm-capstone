# af_join: a dedup bound that drops a buffer reference

`try_push_frame()` copies each output channel's data pointer out of an input
frame and tracks the buffer that backs it, so the output frame can take a
reference to every distinct buffer it now points into. The tracking test asks
whether the buffer was already seen by comparing the search index `j` against
the **channel index** `i`, where it has to be compared against `nb_buffers`.

While every channel brings a new buffer the two are equal and nothing is wrong.
The moment one channel shares a buffer with an earlier one, `nb_buffers` stops
advancing with `i`, and from then on a genuinely new buffer ends the search with
`j == nb_buffers != i` and is never tracked. No reference is taken for it. When
the input frames are released, that storage returns to the pool while the output
frame's `extended_data` still names it.

## The minimum that triggers it

Three output channels over two inputs: input 0 supplies channels 0 and 1 from
one buffer, input 1 supplies channel 2 from its own.

| i | buffer | search ends at | test `j == i` | tracked |
|---|---|---|---|---|
| 0 | input 0 | `j = 0` | 0 == 0 | yes |
| 1 | input 0 (already seen) | `j = 0` | 0 != 1 | no — correct, it is a duplicate |
| 2 | input 1 (**new**) | `j = 1` | 1 != 2 | **no — the defect** |

## Result

    arm=fixed tracked_buffers=2 output_refs=2
    reuse_same_address=0 stale_read=0xB0 new_owner=0xCC
    VERDICT FIXED output holds a reference, storage not reissued
    arm=buggy tracked_buffers=1 output_refs=1
    reuse_same_address=1 stale_read=0xCC new_owner=0xCC
    VERDICT DEFECT-REPRODUCED stale pointer reads the new owner's payload

The buggy arm's stale read returns `0xCC`, the payload written by the next
consumer of that storage, at the identical address the pool reissued. The fixed
arm is the matched control: the output holds the reference, the pool has nothing
to reissue, and the same read returns its own `0xB0`.

## Paired arms in a Capstone domain

The same sequence is case 36 of the port's pool lifetime probes, so it runs
under the QEMU oracle that requires a completed-setup marker, exactly one fault,
and a fault PC equal to the address the domain published for its probe:

| mode | what the allocator does on a last return | outcome |
|---|---|---|
| 0 spatial | bounds and tags only | **completes** — the stale read returns `61`, the next consumer's byte |
| 2 Sublet | each last return to the pool is revoked before the storage is reissued | **faults**, cause 24, pc `0x101a101d8` = the published probe address |

    bash security-tests/qemu/run.sh <out> --cases 36 --modes 0,2 --rounds 1

The spatial arm is the control: it shows the defect is reachable and that the
pool really does reissue the same storage. Neither arm is a CHERI model.
