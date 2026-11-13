# realloc-grow: musl patch 0011 on the native route

`rgrow.c` grows a large chunk twice (a move into a mapping with room, then in place), shrinks it, and checks every
word and every stored pointer; `rgrow stale` reads through the capability an in-place growth replaced, which must
fault. `stage-run.sh` is the stage's `run.sh`: build `rgrow.c` with the patched SDK (`rgrowB_base.native`) and with
an SDK without 0011 (`rgrowA_base.native`, the control), put both in `stage/bin/native/`, and run the stage with
`cscheck-emu/run-stage-twin.py --arm direct`. Expected `@@RGROW` lines:

    0 rgrowB ok    rc=0   RGROW moved=1 inplace=1 OK
    1 rgrowA ok    rc=0   RGROW moved=1 inplace=0 OK     (the in-place reading can be 0)
    2 rgrowB stale rc=139 Segmentation fault             (the replaced child is ended)
    3 rgrowA stale rc=139 Segmentation fault
