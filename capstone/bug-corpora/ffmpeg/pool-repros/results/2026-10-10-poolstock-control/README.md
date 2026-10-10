# ffmpeg/pool-repros column 2 (poolstock) with an in-boot Sublet-heap control -- 2026-10-10 (R5)

Pre-registered at `49e8e23f4122` (R5). One physical Capstone VM boot (the 2026-10-10 audit's platform:
capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, module a88ed2159b43, capstone-exec 0752aa7c49c9).

Until now only the build tied this arm to the Sublet heap; a level0 heap would let the four cases complete too.
The app's safety fixtures 4 and 5 (a use after av_free of a direct heap object) FAULT temporal on the Sublet heap
and RETURN on level0 (ports/ffmpeg/app/host/safety-expect.txt); their poolstock rows were added before this run.

    FFAPP_SAFETY_OUT=<out> bash ports/ffmpeg/app/host/run-safety.sh poolstock 4 5
    FFAPP_CORPUS_OUT=<out> bash capstone/bug-corpora/ffmpeg/pool-repros/runners/run-sublet-port.sh poolstock 1

Control, same boot:

    fx4: AS PREDICTED: FAULT temporal
    fx5: AS PREDICTED: FAULT temporal

The four cases, fixed then buggy:

    fx41: AS PREDICTED  predicted: COMPLETE FIXED  got: VERDICT FIXED output holds a reference, storage not reissued
    fx40: AS PREDICTED  predicted: COMPLETE DEFECT-REPRODUCED  got: VERDICT DEFECT-REPRODUCED stale pointer reads the new owner's payload
    fx43: AS PREDICTED  predicted: COMPLETE FIXED  got: VERDICT FIXED the reset covers the whole list
    fx42: AS PREDICTED  predicted: COMPLETE DEFECT-REPRODUCED  got: VERDICT DEFECT-REPRODUCED a record past ref_count still names reissued storage
    fx45: AS PREDICTED  predicted: COMPLETE FIXED  got: VERDICT FIXED nothing is carried across frames
    fx44: AS PREDICTED  predicted: COMPLETE DEFECT-REPRODUCED  got: VERDICT DEFECT-REPRODUCED the parked pointer corrupts the new owner
    fx47: AS PREDICTED  predicted: COMPLETE FIXED  got: VERDICT FIXED SHORT_REF keeps the frame alive, so the tables stay with it
    fx46: AS PREDICTED  predicted: COMPLETE DEFECT-REPRODUCED  got: VERDICT DEFECT-REPRODUCED the decoder's side table went back to the pool and was reissued

Image sha256: ffapp_fx4 90a0d4dbeb8d448a, ffapp_fx5 7d4cec9bbe69d9b3, ffapp_fx40 6cff79b18b940c69, ffapp_fx41 727742a2a017ee6a, ffapp_fx42 796fe3de033cb482, ffapp_fx43 1c3393e0df74fb56, ffapp_fx44 7d90d0cd47d3e665, ffapp_fx45 6e8abc88f9545b79, ffapp_fx46 c3b80ac4d0713d6b, ffapp_fx47 d0870dc6cdc32e74

So column 2's four misses are readings on a Sublet heap that, in the same boot, faults on a direct use after free.
