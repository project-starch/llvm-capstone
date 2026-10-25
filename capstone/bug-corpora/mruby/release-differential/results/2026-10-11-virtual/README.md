# mruby/release-differential on virtual Capstone -- 2026-10-11

The 23 cases on the virtual profile, two arms. Source: branch `nested/mruby-lifetimes` (from
`nested/sublet-instructions` `0146ba87be18`), mruby 4.0.0-rc2 with the port's patches, built by
`ports/mruby/app/build-mruby-domain.sh` with `MRBD_SDK` naming a virtual SDK (the PostgreSQL
lane's `pgv5-none` SDK, copied). One VM, virtual profile with exact bounds, the Sublet-lifetime
QEMU; each bundle's `inputs.json` carries the image and platform hashes. Runner:
`probe/run-virtual.py`, one `capstone-vexec` process per case, 120 s per case. Raw console
captures are not committed.

| arm | image | control `smoke.rb` | fault | completed | time |
|---|---|---|---:|---:|---:|
| `virtual-malloc` | `7b5872bc7141cbad` (stock) | completed, `SMOKE_DONE` | 20 | 3 | 137 s |
| `virtual-nested-pools` | `321dc9b98691cc0a` (`MRBD_SUBLET=1`, patch 0008) | completed, `SMOKE_DONE` | 23 | 0 | 138 s |

The two images differ by patch 0008's define alone: the stock image is byte-identical (same
sha256) to one built earlier the same day with the old adapter patch in the tree and its own
define off, so neither patch changes a build without its define; the protected one carries 11
more Capstone-specific instructions, the `CDERIVE`/`CREVOKE` sites.

Per case (`matrix.tsv` in each bundle): cause and the function the fault's pc lies in.

| case | virtual-malloc | virtual-nested-pools |
|---|---|---|
| 01 `13e017c2f` | 25 `mrb_vm_exec` | 25 `mrb_vm_exec` |
| 02 `39aecc143` | 24 `mrb_vm_exec` | **25 `mrb_gc_protect`** |
| 03 `59552ecb8` | 25 `mrb_str_format` | 25 `mrb_str_format` |
| 04 `606d9a6b2` | completes, wrong answer | **25 `mrb_field_write_barrier`** |
| 05 `628ccec60` | completes, wrong answer | **25 `mrb_vm_exec`** |
| 06 `84cc5aa60` | 25 `kh_get_set_val` | 25 `kh_get_set_val` |
| 07 `eb7693857` | 24 `mrb_vformat` | 24 `mrb_vformat` |
| 08 `fb4974528` | completes, wrong answer | **25 `mrb_vm_exec`** |
| 09 `0cf969a2b` | 25 `mrb_iv_foreach` | 25 `mrb_iv_foreach` |
| 10 `7c5915799` | 25 `mrb_struct_equal` | 25 `mrb_struct_equal` |
| 11 `cb51fce92` | 12 (not a capability fault)* | 12 (not a capability fault)* |
| 12 `4663fef45` | 28 `ar_get` | 28 `ar_get` |
| 13 `4a386f80e` | 28 `mrb_pack_pack` | 28 `mrb_pack_pack` |
| 14 `93eb74a59` | 28 `mrb_obj_ceqq` | 28 `mrb_obj_ceqq` |
| 15 `af6f23ddb` | 28 `memcpy` | 28 `memcpy` |
| 16 `ec89364c4` | 28 `ary_fill_exec` | 28 `ary_fill_exec` |
| 17 `bef45e223` | 25 `mrb_hash_pat_values` | 25 `mrb_hash_pat_values` |
| 18-21 (NULL `mt`) | 24 `mrb_mod_visibility` | 24 `mrb_mod_visibility` |
| 22 `d8911416c` | 24 `mrb_proc_parameters` | 24 `mrb_proc_parameters` |
| 23 `1737589f0` | 28 `mrb_obj_alloc` | 28 `mrb_obj_alloc` |

Causes: 24 a capability without a tag (or of the wrong type), 25 an invalid lifetime, 28 out of
bounds (`docs/design/virtual-capstone/isa.md`). Every difference between the arms is on a GC
object slot, and each is cause 25.

\* `trigger.rb`, which at the pin does not reach case 11's defect. The case is read through its
C-API driver in [2026-10-11-virtual-capi](../2026-10-11-virtual-capi/README.md): cause 25 in
`gc_mark_children` on both arms.

## Soundness of the patch

mruby's test suite (`mrbtest`, `MRBD_TESTS=1` builds of both arms) natively and on the VM:

| build | total | OK | KO | crash | skip | capability fault |
|---|---:|---:|---:|---:|---:|---|
| native | 1710 | 1701 | 0 | 0 | 9 | -- |
| virtual, stock | 1710 | 1700 | 1 | 0 | 9 | none |
| virtual, patch 0008 | 1710 | 1700 | 1 | 0 | 9 | none |

The one failure on both virtual images is the same test, `File#path` (mruby-io), which reads a
file the guest gives back empty. The patched build's line counts are taken over two runs: the
guest's console drops some lines of a long output, and the first run lost its `OK`, `Crash` and
`Skip` lines; the second has them. Both runs exited normally (status 1, from the one failure).
Each mrbtest run took about 55 s, each build about 2 minutes, the VM 18 s to boot.

N = 1 per case cell.
