# DIAG-BTU: what silicon compared at a false duplicate key (lane only, never for dev)

sql0-nf2A case 03 fails on silicon only with `duplicate key value violates unique constraint
"pg_operator_oprname_l_r_n_index"`, in `CREATE EXTENSION ltree`. QEMU never reports it. This diagnostic image appends
one `errhint` to `_bt_check_unique`'s duplicate-key ereport (nbtinsert.c), so the client output carries:

    HINT:  DIAG-BTU existing <values> tid (<blk>,<off>) blk <page> off <offset> posting <inposting>/<posti>
           len <n> checked (<blk>,<off>) raw<existing tuple, up to 64 bytes hex> | new tid (<blk>,<off>) len <n> raw<...>

`existing` is the index tuple that compared equal, copied before the buffers are released, with its OWN heap TID (the
posting entry when it is a posting tuple). `checked` is the TID whose liveness the second check tested. `new` is the
tuple being inserted.

Build (the record's inputs, nothing else changed): `nbtinsert-diag.patch` applied to postgres-17.5's nbtinsert.c,
compiled with `cflags.txt` ($PG = the postgres-17.5 tree of t3-new6/postgres-lt0/source/domain) by the record's SDK `capstone-cc`. The unmodified file compiled the same way reproduces the
recorded object (e725dc31bee468e7). `relink.py diag` swaps it into the recorded link
(t3-new6/postgres-lt0-dl/image/commands.json), and `relink.py ctrl` with the control object reproduces the record
(image 81ee24a0, loaded bytes d50bc338ea18cf28 = 64010720's).

Images: diag nbtinsert.o 0ea073f6a08dda38 -> postgres.dom a2bf0a6ed7c6784b (loaded 28aef8f054ca4f71). Runtime and
every other input are the record's (runtime 5b3e6b79, archive 90f6e08a).

Positive control: `98_planted_dup` (this lane only) violates a primary key; on QEMU (kit qkit7-nf2) it prints
`DIAG-BTU existing (a)=(1) tid (0,1) ... checked (0,2) ... | new tid (0,2) ...`, and case 03 prints no DIAG line.
