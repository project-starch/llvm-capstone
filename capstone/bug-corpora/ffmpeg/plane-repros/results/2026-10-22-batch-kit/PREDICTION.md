# ffmpeg/plane-repros on the batch recipe: QEMU references for the vp2 arms (pre-registered 2026-10-22, before the runs)

**Why:** the board lane grades each vp2 arm against a QEMU reference built the way the board image is built. The
2026-10-13 virtual bundles beside this directory were built on an older stack (`virtstk/ff0`, QEMU a08a0e37, module
0551b066). Every other batch arm's reference was re-run on the batch recipe (SDK t3-freeze-nb-nomovc-cw-r60, kit
qkit7-batch2), and this corpus had no such run.

**The library:** libavutil from `build-virtual.sh ffmpeg` with lifetimes 0, the same recipe as the batch's CPython and
PostgreSQL images.
- Built with the frozen r60 toolchain, `CAPSTONE_VIRTUAL_MOVC_KEEP=OFF` and `-fno-builtin`.
- The tree is llvm-capstone `batch-recipe-ffmpeg` 015e4d66100c: the batch tree 8f132305eea1 plus dev's port-only fix
  ad68be50dc96. Without that fix FFmpeg's configure hands `-target` to capstone-cc, which refuses it.
- The build's codegen list equals the batch SDK's sdk.json codegen word for word.
- The FFmpeg source is the original run's, `src-9.0.1-f7e8b586f7b4`.

**The runs:**
- Runner: this branch's `run-virtual-cases.py`, which is dev's plus the plane additions.
- Kit and SDK: `--virtual-kit` qkit7-batch2 (QEMU ecaf50bb, module a05c198b) and `--sdk` t3-freeze-nb-nomovc-cw-r60/sdk.
- `--cc-arg`s: the 2026-10-13 inputs.json lists, re-pointed at the new build. The nested arm's runtime include comes from
  the batch tree; its only difference from dev's is delegate.h.
- One runner call per arm, on QEMU domain B (script `plane-vp2/run-plane.sh`). Its verdict parser was tested on the
  2026-10-13 nested bundle and on a missing directory (NO DATA, exit 2) before this commit.

**The prediction:** each arm reads what its 2026-10-13 bundle read.
- **virtual-malloc:** MISSED. The case is reached (VERDICT DEFECT-REPRODUCED) and completes with no fault. Controls:
  bounds-malloc fault, uaf-malloc fault.
- **virtual-nested-pools** (configuration virtual-ffmpeg-plane-carve): MISSED, the same way.
  - Controls: bounds-malloc fault, uaf-malloc fault, plane-live complete, plane-bound fault, plane-free fault.
  - The carve's bound and revocation both fire in their controls, so the miss is the case's row staying inside the
    alpha plane's allocated extent, as on 2026-10-13.
- **Any difference names the recipe change:** library, SDK, kit or runner.
