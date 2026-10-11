# wmem-repros: the domain regression after the 2026-10-09 case.c edits

The 22 case.c files gained their upstream fix behind `wm_fixed` and a native-only reoccupation
(`wm_reoccupy`) for the fix differential. Both are dead on every domain arm (the hosted entry
`program 0 N buggy|fixed` is the only way to set them), so the domain readings must be UNCHANGED.
This re-runs every case, both arms, on both builds, through the corpus's own runner
(`shared/run-defects.py`, one invocation per case so an infrastructure refusal costs one boot),
and compares with the recorded outcomes.

Builds: `WM_CHUNKS=OFF` (spatial = level0 heap, sublet = the Sublet wmem port) and `WM_CHUNKS=ON`
(the tshark chunks configuration; its `sublet` arm is the corpus's `sublet-chunks` column; cases
0-17, the ones that build in that configuration, as recorded).

Infrastructure refusals (BOOT PRODUCED NO RESULT, exit 75, retried up to 4 times): 9 boots over 5 case-builds (the runner's own status lines); none left a case without a reading.

## Result lines

    build case arm      expected outcome   cause pc==expected  image(sha256/16)
    off      0 spatial  complete complete      - -            d12694efd4ec78e1  PASS
    off      0 sublet   fault    fault        24 True         d12694efd4ec78e1  PASS
    off      1 spatial  complete complete      - -            f8abc30dab5c6d76  PASS
    off      1 sublet   fault    fault        24 True         f8abc30dab5c6d76  PASS
    off      2 spatial  complete complete      - -            9b6e2ab090495346  PASS
    off      2 sublet   fault    fault        24 True         9b6e2ab090495346  PASS
    off      3 spatial  complete complete      - -            fcd4227ff3578ec6  PASS
    off      3 sublet   fault    fault        24 True         fcd4227ff3578ec6  PASS
    off      4 spatial  complete complete      - -            42d6e1415a7e0f73  PASS
    off      4 sublet   fault    fault        24 True         42d6e1415a7e0f73  PASS
    off      5 spatial  complete complete      - -            f8b0401a6b72f084  PASS
    off      5 sublet   fault    fault        24 True         f8b0401a6b72f084  PASS
    off      6 spatial  complete complete      - -            b4759eb4499ee256  PASS
    off      6 sublet   fault    fault        24 True         b4759eb4499ee256  PASS
    off      7 spatial  complete complete      - -            3726688f3d627511  PASS
    off      7 sublet   fault    fault        24 True         3726688f3d627511  PASS
    off      8 spatial  complete complete      - -            cf10ed3a879c4508  PASS
    off      8 sublet   fault    fault        24 True         cf10ed3a879c4508  PASS
    off      9 spatial  complete complete      - -            220977bc31aab6da  PASS
    off      9 sublet   fault    fault        24 True         220977bc31aab6da  PASS
    off     10 spatial  complete complete      - -            73ce8bec7404ac8b  PASS
    off     10 sublet   fault    fault        24 True         73ce8bec7404ac8b  PASS
    off     11 spatial  complete complete      - -            89dc907bdf8b1ffb  PASS
    off     11 sublet   fault    fault        24 True         89dc907bdf8b1ffb  PASS
    off     12 spatial  complete complete      - -            9087d456a7e2e1c1  PASS
    off     12 sublet   complete complete      - -            9087d456a7e2e1c1  PASS
    off     13 spatial  fault    fault         5 True         2b2330b8a5576170  PASS
    off     13 sublet   fault    fault         5 True         2b2330b8a5576170  PASS
    off     14 spatial  fault    fault         5 True         4adf0fcd1f61c2f2  PASS
    off     14 sublet   fault    fault         5 True         4adf0fcd1f61c2f2  PASS
    off     15 spatial  fault    fault         5 True         9807d7fd55bf18f1  PASS
    off     15 sublet   fault    fault         5 True         9807d7fd55bf18f1  PASS
    off     16 spatial  fault    fault         7 True         d75d1c3a617ff51c  PASS
    off     16 sublet   fault    fault         7 True         d75d1c3a617ff51c  PASS
    off     17 spatial  fault    fault         7 True         cf59a5f565ce936c  PASS
    off     17 sublet   fault    fault         7 True         cf59a5f565ce936c  PASS
    off     18 spatial  fault    fault         7 True         3d25531ea8855db5  PASS
    off     18 sublet   fault    fault         7 True         3d25531ea8855db5  PASS
    off     19 spatial  fault    fault         5 True         ecad13c17f96d844  PASS
    off     19 sublet   fault    fault         5 True         ecad13c17f96d844  PASS
    off     20 spatial  fault    fault         7 True         e19918d3f406e064  PASS
    off     20 sublet   fault    fault         7 True         e19918d3f406e064  PASS
    off     21 spatial  fault    fault         7 True         a68cc25c8667a6c1  PASS
    off     21 sublet   fault    fault         7 True         a68cc25c8667a6c1  PASS
    on       0 spatial  complete complete      - -            37ebb5378e77e969  PASS
    on       0 sublet   fault    fault        24 True         37ebb5378e77e969  PASS
    on       1 spatial  complete complete      - -            0ba9187608304075  PASS
    on       1 sublet   fault    fault        24 True         0ba9187608304075  PASS
    on       2 spatial  complete complete      - -            273c1847211d811e  PASS
    on       2 sublet   fault    fault        24 True         273c1847211d811e  PASS
    on       3 spatial  complete complete      - -            8fda315b14a32b1e  PASS
    on       3 sublet   fault    fault        24 True         8fda315b14a32b1e  PASS
    on       4 spatial  complete complete      - -            fa67b4f2ac10d626  PASS
    on       4 sublet   fault    fault        24 True         fa67b4f2ac10d626  PASS
    on       5 spatial  complete complete      - -            68c8c3c79e265322  PASS
    on       5 sublet   fault    fault        24 True         68c8c3c79e265322  PASS
    on       6 spatial  complete complete      - -            b53427a6b4747b5b  PASS
    on       6 sublet   fault    fault        24 True         b53427a6b4747b5b  PASS
    on       7 spatial  complete complete      - -            ead71a373b130c20  PASS
    on       7 sublet   fault    fault        24 True         ead71a373b130c20  PASS
    on       8 spatial  complete complete      - -            6c73e00af51656aa  PASS
    on       8 sublet   fault    fault        24 True         6c73e00af51656aa  PASS
    on       9 spatial  complete complete      - -            b0f0443c70610b7c  PASS
    on       9 sublet   fault    fault        24 True         b0f0443c70610b7c  PASS
    on      10 spatial  complete complete      - -            906366da762b50ec  PASS
    on      10 sublet   fault    fault        24 True         906366da762b50ec  PASS
    on      11 spatial  complete complete      - -            c28231760f7f165b  PASS
    on      11 sublet   fault    fault        24 True         c28231760f7f165b  PASS
    on      12 spatial  complete complete      - -            2697f50972f36790  PASS
    on      12 sublet   fault    fault        24 True         2697f50972f36790  PASS
    on      13 spatial  fault    fault         5 True         55476bcb8db3ffd1  PASS
    on      13 sublet   fault    fault         5 True         55476bcb8db3ffd1  PASS
    on      14 spatial  fault    fault         5 True         9c9c0b156fb3dc07  PASS
    on      14 sublet   fault    fault         5 True         9c9c0b156fb3dc07  PASS
    on      15 spatial  fault    fault         5 True         ca5749ce7fa787c0  PASS
    on      15 sublet   fault    fault         5 True         ca5749ce7fa787c0  PASS
    on      16 spatial  fault    fault         7 True         8ec9a0c23362606e  PASS
    on      16 sublet   fault    fault         7 True         8ec9a0c23362606e  PASS
    on      17 spatial  fault    fault         7 True         fe1a8ec857e5b84a  PASS
    on      17 sublet   fault    fault         7 True         fe1a8ec857e5b84a  PASS

## Against the recorded outcomes

Recorded: spatial completes 0-12 and faults 13-21; sublet faults 0-11 and 13-21 and completes 12 (OFF);
sublet-chunks faults 0-17 (ON). Rows compared: 80; mismatches: 0 -- the regression is clean.
Runner verdict (oracle and, for a fault, pc equal to the labelled probe): 80 of 80 PASS.
