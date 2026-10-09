# ffmpeg/subobject-repros: the program's whole Sublet configuration (`sublet-full`), 2026-10-09

**Question.** Column 3 of the per-bug table ("Sublet in nested") for a plain case, by the lead's
choice: run the case in the program's full Sublet configuration -- the Sublet heap AND the program's
nested-allocator port, live -- rather than leave the cell empty.

**Build.** Each case is linked with FFmpeg's pools on their Sublet port (tools/full-config/ffmpeg.c, ffsublet.c, the app port's FF_SUBLET_POOLS=1 libavutil), whose constructor brings the port up before
`main()` and drives one issue and one revoke through it; every run must print its `FULLCONFIG ...
live` line or `tools/run-capstone-domain.py --arm sublet-full` refuses the run (exit 75).
The sub-object driver's calls into the buffer-pool port's replay backend are no-ops (tools/full-config/ffmpeg-subobject-shim.c): its pools are FFmpeg's own on their Sublet port. Two runs produced no reading before this one: the first, with the backend linked, because the constructor's pool got no memory (a28d6c33fd97); the second, with the shim, because it used the HEAP_LOG 22 SDK, on which the driver's own 64 + 16 MiB arenas do not fit (code 604, every fixed arm) -- the same refusal the sub-object corpus's `sublet` arm met and passed on the HEAP_LOG 27 SDK, `sdk-sublet-cap`, which this run uses too.

**Result: 0 of 10 caught, 10 of 10 as pre-registered (aeba7ddefb4d)** --
the same reading as the `sublet` arm on every case. The case's objects come from libavutil's allocator on the
Sublet heap (av_refstruct_allocz, av_malloc), not from a pool, so the live port is off the bug's path; the run shows it changes nothing for a direct-allocation
bug. Controls in the same boot as every arm of this tool (clean, oob, uaf, subobj) behaved.

**What this is, and is not.** A non-interference result: every reading is unchanged with the port linked and live, and the port is off these bugs' path, so no row here is a catch by the port. The image carries FFmpeg's real pool code (the application port's FF_SUBLET_POOLS=1 libavutil) over ffsublet.c. The SDK is the plain corpora's own Sublet SDK (HEAP_LOG 22); the sub-object corpus's HEAP_LOG 27 one, as its sublet arm, so the heap size differs from the `sublet` arm's as well as the port.

The shim also moves `av_malloc` from the replay backend's single arena (no per-object bound, `metadata-allocator.c`) to libavutil's on the Sublet heap: this corpus's `sublet` arm never had a per-object Sublet bound under the case objects. The reading is 0 of 10 either way -- each crossing stays inside one struct.
