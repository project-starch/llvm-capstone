# The PHP engine-repros on kit-p1v at P1's geometry (2026-11-23)

The QEMU verdicts that the P1 board run is predicted from, as amended (amendment 1 to "Pre-registration: item 5 on a
P1 board", `docs/ref/nested-allocator-workloads.md`). Same image bytes (`images.sha256`) as `../2026-11-23-p1v-twin/`,
on kit-p1v's own QEMU, which since that run adds `cma=256M` itself.

At P1's geometry the module ran `supervisor_slots=1 resolve_dirty=1 resolve_pte=1` (`inputs.json` `module_args`).
QEMU had `x-capstone-supervisor-slots=1`; the earlier bundles had 32 slots. Each boot's serial log shows the applied
`-cpu` with `x-capstone-supervisor-slots=1` and `x-capstone-ids=55`, without exact bounds, whatever the record's
`qemu_cpu` requests. It also shows `capstone_vm: loaded, … supervisor_slots=1` and the flat table of 1,048,576
records from CMA.

Every one of the 280 cells (156 asan-blind, 124 temporal-asan-visible) equals `../2026-11-23-p1v-twin/` in verdict,
reason, cause and faulting function.

Account paths in `inputs.json` are written `<hdd>/`. Raw guest output is not committed.
