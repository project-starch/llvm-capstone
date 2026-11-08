# gcc train with direct calls and no frame pointer (compiler lane's image), board 3: pre-registration

Written and pushed before the board call (8 Nov, 12:1x). The compiler lane measured where gcc train's in-context
instructions exceed Plain-with-cache (13.735 G vs 10.977 G program instructions, +2.76 G) and built three images on
c3f's exact recipe, one change each (c1-gate/gt2, script build-arm.sh). This run takes the combined one:
- dcnofp: `-mllvm -capstone-direct-calls` (llvm-c4 = origin/dev c9611344eeb7 + 331d73414667, branch
  compiler-spec-c4, default off) and `-fomit-frame-pointer` on gcc's compiles; runtime and libc prebuilt as in c3f.
- Stripped exe 736f881f4958ac3b. QEMU twin (kit-profq): specdiff 1/1 PASS, every output hash equal to c3f's,
  in-context instret 12,868,888,921 against c3f's 13,735,649,769 (−6.31 %). Edge check SAFE (slack 1,760 B).
- The lead's trade, taken here as a measurement: without a frame pointer, a fault inside gcc's code gets no frame
  chain in the launcher's fault record (diagnostic only; nothing decides on it). Plain has no frame pointer either.

**Stage** c3f's board stage with only the executable replaced (5642377943d724c6 → 736f881f4958ac3b). **Kit**
bkit27-gtq. **Board 3**, after the tickcost boots there.

| # | Prediction |
|---|---|
| D0 | rc 0, specdiff 1/1, outputs equal c3f's; control passes; in-context instret within ±0.1 % of 12.869 G |
| D1 | inside time 3,514 s → 3,280-3,370 s (the removed instructions are moves, address arithmetic and spills, cheaper than average) |
| D2 | run 4,253 s → 4,030-4,110 s, i.e. 1.23-1.25× Plain-with-cache (3,280.9 s); outside about unchanged |
