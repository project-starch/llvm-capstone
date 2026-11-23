# The sqlite engine-repros on kit-p1v at P1's geometry (2026-11-23)

The QEMU verdicts that the P1 board run is predicted from, as amended (amendment 1 to "Pre-registration: item 5 on a
P1 board", `docs/ref/nested-allocator-workloads.md`). Same image bytes (`images.sha256`) and the same kit as
`../2026-11-23-p1v-twin/`. The module ran with P1's geometry: `supervisor_slots=1 resolve_dirty=1 resolve_pte=1`,
and the CMA reserve was 256 MiB. The runner read these back from `/sys/module/capstone_vm/parameters/` on both VMs:
`ids=55`, `index_bits=25`, `node_flat_records=1048576`, `supervisor_slots=1`, `resolve_dirty=Y`, `resolve_pte=Y`. The
QEMU applied `x-capstone-ids=55,x-capstone-addr=sv39` with no exact bounds. These parameters are not in
`inputs.json`, whose `exact_bounds` field is false here.

Every one of the 66 cells equals `../2026-11-23-p1v-twin/` in verdict, reason, cause, faulting function and offset.
Only absolute addresses differ (pc, ra, frame chain), because the load base moves between boots.

Account paths in `inputs.json` are written `<hdd>/`. Raw guest output is not committed.
