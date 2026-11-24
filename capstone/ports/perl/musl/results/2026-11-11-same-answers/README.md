# Perl 5.36.3: protected build, same answers, on the one-operand lifetime instructions (2026-11-11)

The gate of [`../2026-10-11-same-answers/`](../2026-10-11-same-answers/README.md), run again after the lifetime
instructions changed (`CREVOKE` takes the object alone) and the SV-head patch
(`../../../sublet/patches/5.36.3/0001`) was converted to them: `t/base` (9 files) and `scripts/smoke.pl` on native
Perl, on the stock virtual image and on the protected one (`PERLD_SUBLET=1`). N = 1 per arm. Scripts: [`run.sh`](run.sh)
and [`smoke-sizes.sh`](smoke-sizes.sh); every value is in `results.json`.

## Outcome

| file | native | stock (virtual) | protected (`PERLD_SUBLET=1`) | protected = stock | = native |
|---|---|---|---|---|---|
| base/cond.t | 4/4 ok, exit 0 | 4/4 ok, exit 0 | 4/4 ok, exit 0 | yes | yes |
| base/if.t | 2/2 ok, exit 0 | 2/2 ok, exit 0 | 2/2 ok, exit 0 | yes | yes |
| base/lex.t | 120/120 ok, exit 0 | 120/120 ok, exit 0 | 120/120 ok, exit 0 | yes | yes |
| base/num.t | 56/56 ok, exit 0 | 56/56 ok, exit 0 | 56/56 ok, exit 0 | yes | yes |
| base/pat.t | 2/2 ok, exit 0 | 2/2 ok, exit 0 | 2/2 ok, exit 0 | yes | yes |
| base/rs.t | 41/41 ok, exit 0 | 41/41 ok, exit 0 | 41/41 ok, exit 0 | yes | yes |
| base/term.t | 7/7 ok, exit 0 | 6 of 7, `not ok 2`, exit 0 | the same | yes | no (port) |
| base/translate.t | 257/257 ok, exit 0 | 257/257 ok, exit 0 | 257/257 ok, exit 0 | yes | yes |
| base/while.t | 4/4 ok, exit 0 | 4/4 ok, exit 0 | 4/4 ok, exit 0 | yes | yes |
| smoke.pl, 2,000 objects | 18 lines, exit 0 | identical, exit 0 | identical, exit 0 | yes | yes |
| smoke.pl, 6,000 objects | the same 18 lines | identical, exit 0 | identical, exit 0 | yes | yes |
| smoke.pl as it is (20,000 objects) | the same 18 lines | identical, exit 0 | **not finished within the 600 s bound** (see the next section) | no reading | stock yes |

"= stock" and "= native" compare the whole stdout by sha256.

- **Protected equals stock on the nine `t/base` files and on smoke.pl at 2,000 and 6,000 objects**, stdout for stdout,
  exit status for exit status. No domain fault on either arm.
- **Eight `t/base` files and smoke.pl equal native** on both Capstone arms.
- **`base/term.t` test 2 differs from native on both arms, the same way.** The test runs
  `` `$^X -le "print 'hi there'"` ``. The guest shell runs the image as a script, since binfmt is not registered in
  this gate, and the backtick returns nothing: `not ok 2`. On 2026-10-11 the domain then faulted with cause 25 in
  `fileno_unlocked` and the file ended after `ok 1`; now there is no fault and tests 3 to 7 run.
- **The stock image completes smoke.pl at its default 20,000 objects**, with native's output. On 2026-10-11 both
  images ran out of memory there.

## smoke.pl on the protected image: the time grows with the object count

| objects | stock, guest s | protected, guest s | protected built from dev, host s |
|---|---|---|---|
| 2,000 | 89 | 165 | 158 |
| 6,000 | 92 | 281 | 306 |
| 20,000 | 159 | more than 600, stopped at the bound | 807 |

The stock and protected columns are one boot ([`smoke-sizes.sh`](smoke-sizes.sh), each run bounded by 600 s). A first
attempt inside `run.sh`'s boot had the protected image at 20,000 objects for more than 14 minutes before the boot was
stopped, which is why `results.json` has no protected smoke row in `same_answers_boot`.

The growth is not the instruction change. The last column is Perl built from unmodified dev (f8d6cb8bc) with the
two-operand SV-head patch, on the QEMU that dev pins, at host wall time on a persistent VM: 158 s and 306 s against
165 s and 281 s here, and it completes the 20,000-object run in 807 s. The cost per object is the same before and
after. Here it is roughly linear between the two measured points, about 29 ms per object on this host under the
campaign's load, so the 20,000-object run would need about 690 s, more than the 600 s bound it was given. What the time
goes into was not measured.

## What ran

- Tree: branch `one-operand-crevoke-crenew` at 59201bdf2, QEMU at ea57497908. The SV-head patch is 822a151fc302....
- Both images by `build-perl-domain.sh` with `PERLD_PROFILE=virtual`, default `PERLD_OPT` and
  `PERLD_VIRTUAL_BLOCKS`, port patches 0001-0012; protected adds `PERLD_SUBLET=1`.

| | sha256 |
|---|---|
| stock image | a0878a2b759dcd69de61dc6258b2fbd502b67329bd3d4ec3815b13524ed47f97 |
| protected image | 5507d74ec7dd89f083bb4fdb99ff206f62a28ef5c98d103b0a3ea905878f2f75 |
| smoke.pl | 15bf158879a2f5432b183ed944bd52d928e0e4bbf40a5f93a8d9d0f2c6af1f44 |
| qemu-system-riscv64 | 2d5a7107890e48475fada73439d90ffd5c7cff166eb1d547d7a9fc5f31722365 |
| capstone-vexec | 0f8e0f903725d734817b3e30d79127c290f3480a641d4f6de8bc7f699d05c76e |
| capstone_vm.ko | e47bca62addfc9356f8150755f2500b737912eb01088e91398e29a1ed7a4e180 |

- Lifetime instruction words: stock 6 `CDERIVE` and 3 `CREVOKE` (the C library's mallocng), protected 46 and 6, no
  `CRENEW`.
- Platform: staged boots through `runtime/virtual/run-staged.py`, one hart, 2 GiB, the module loaded without parameters
  (pinned, one context), guest CPU with `x-capstone-test-csrs=true`, which a pinned boot of this kernel needs.
- The guest times include the host's load: 24 QEMU sessions of the same campaign ran during `run.sh`'s boot and about
  ten during `smoke-sizes.sh`'s. They compare the arms inside one boot and are not a performance result.
