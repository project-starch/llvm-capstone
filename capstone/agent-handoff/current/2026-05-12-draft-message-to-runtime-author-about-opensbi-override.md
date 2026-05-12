# Draft message to the runtime / harness author

Subject: Need the intended Capstone OpenSBI rebuild path and canonical generated SBI wrapper files

Hi,

I am debugging the current `null_blk` split-path failure in the local `llvm-capstone` / `caplifive-buildroot` workspace.

I have now verified that the current runtime image behaves like a stock OpenSBI path rather than a Capstone-enabled OpenSBI path:

- host-side `DOM_CREATE` returns `error=-2, value=0`
- host-side `DOM_CALL_WITH_CAP` returns `error=-2, value=0`
- host-side `REGION_CREATE` returns `error=-2, value=0`
- baseline `null_blk` still works
- split `null_blk` no longer crashes after adding error checks, but now fails cleanly during region setup

I also found that `capstone/caplifive-buildroot/build/local.mk` had been deleted locally, and that file is what restores:

```makefile
LINUX_OVERRIDE_SRCDIR = $(BR2_EXTERNAL_CAPSTONE_PATH)/components/linux
OPENSBI_OVERRIDE_SRCDIR = $(BR2_EXTERNAL_CAPSTONE_PATH)/components/opensbi
```

So one likely explanation is that the local image stopped using the Capstone-enabled OpenSBI override.

The remaining blocker is that the local OpenSBI override path appears to depend on generated files that are currently absent from the tree:

- `components/opensbi/lib/sbi/sbi_capstone_dom.c.S`
- `components/opensbi/lib/sbi/capstone_int_handler.c.S`

I searched the child repositories in the current workspace and did not find copies of those files.

I also tried the documented regeneration path, but the current local Capstone-C toolchain fails while generating `sbi_capstone_dom.c.S` with a panic in `dag_builder.rs` on `split_out_cap`.

Questions:

1. What is the intended source of truth for these generated OpenSBI wrapper files?
   - Are `sbi_capstone_dom.c.S` and `capstone_int_handler.c.S` expected to be regenerated locally every time?
   - Or is there a canonical checked-in version somewhere else?

2. Is `build/local.mk` expected to remain committed/tracked in this repo as part of the normal local Buildroot override flow?

3. What is the recommended reproducible command sequence to rebuild a Capstone-enabled `fw_jump.elf` from the current tree when `components/opensbi` is modified?

4. If the Capstone-C generator panic is known, is there a pinned compiler revision / workaround that should be used for generating the OpenSBI wrapper assembly?

If helpful, I can also provide the exact logs showing:

- `DOM_CREATE ... error=-2 value=0`
- `DOM_CALL_WITH_CAP ... error=-2 value=0`
- `REGION_CREATE ... error=-2 value=0`
- the Capstone-C generator panic while building `components/opensbi/lib/sbi/sbi_capstone_dom.c.S`

Thanks.

