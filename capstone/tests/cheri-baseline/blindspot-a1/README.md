# A1 under CHERI purecap: MEASURED, and CHERI misses it

The first scored blind-spot case. mruby issue 6339: `mrb_ary_delete` keeps the
removed element in a local the GC does not know about, the element's `==` runs
Ruby which runs the GC, the object is swept while `delete` is still running, and
its slot comes back off the **page free list** as a `String`.

Nothing reaches `malloc` or `free`, so purecap sees an in-bounds access through a
tagged capability and revocation has nothing to revoke. The oracle is therefore
the ANSWER, not a crash.

## Result

| cfg | revocation | every_free | vehicle control | sanity | A1 | A1 rc |
|---|---|---|---|---|---|---|
| spatial | 0 | 0 | `rc=162 SIGPROT`, only `CONTROL=BEFORE` | `SANITY_OK=7` | `A1RESULT=2 class=String` | 0 |
| temporal | 1 | 0 | `rc=162 SIGPROT`, only `CONTROL=BEFORE` | `SANITY_OK=7` | `A1RESULT=2 class=String` | 0 |
| eager | 1 | 1 | `rc=162 SIGPROT`, only `CONTROL=BEFORE` | `SANITY_OK=7` | `A1RESULT=2 class=String` | 0 |

`rc=0` on A1 throughout, so no signal: a wrong answer, not a suppressed crash.
**CHERI purecap misses it in all three configurations, eager revocation
included.**

**And the same harness, in the same boot, CATCHES a plain heap overflow.**
`catch_control.c` mallocs 64 bytes and walks 4096. It dies on SIGPROT (162 =
128+34) after `CONTROL=BEFORE` and never reaches `CONTROL=AFTER`. Without that
row, "CHERI misses A1" could not be told from "this harness never reports
anything" -- every verdict it had produced up to that point was a MISS. The
control is a vehicle check, not a corpus case, and it is the difference between a
result and an assertion.

## What makes this a measurement rather than a clean run

Four controls, and three of them caught a real error while being set up.

* **The trigger fires.** On the host, the affected build answers 2 and prints
  `class=String`; mruby master answers 1 and prints `class=C`. Ten runs each.
* **The version had to be found, not assumed.** The catalogue said "affected
  <= 3.3.0". No release carries the C `mrb_ary_delete` at all -- 3.2.0, 3.3.0 and
  3.4.0 all still have the Ruby-level `Array#delete`. The purecap tree already on
  disk is mruby 3.0.0, and running the script there answered "correct" because the
  function under test did not exist in it. The measured build is `0972c8477^`.
* **The oracle needed an allocation burst.** Without one the freed slot has not
  been recycled and `is_a?`, `x.class == C`, `x.class.to_s` and `instance_of?` all
  answer 1 on BOTH builds. An earlier specimen appeared to work only because it
  evaluated five oracle expressions in a row and their own string building
  recycled the slot between them. An oracle that depends on its own evaluation
  order is not an oracle.
* **`sanity.rb` is why this table is not fiction.** The first harness read the
  EXIT CODE. This build has no `Kernel#exit` -- mruby-exit is not in the gembox --
  so every run reported `rc=1` for `NoMethodError`, which is indistinguishable
  from "the answer was correct". Three clean `rc=1` lines would have been recorded
  as "CHERI misses nothing". The sanity file failed in the same way at the same
  time, which is what exposed it.

## Reproducing

The purecap binary is not committed; build it:

```sh
SHA=$(git -C <mruby-clone> rev-parse 0972c8477^)
CHERI_ROOT=$HOME/cheri MRUBY_SRC=$HOME/cheri/mruby-a1-purecap MRUBY_REV=$SHA \
  bash xlang/cheri/mruby-port/build-purecap-mruby.sh
```

Then stage this directory plus `build/purecap/bin/mruby` as `/root/a1-6339` in the
guest image and drive it:

```sh
fakeroot -s $FR -- sh -c "mkdir -p $ROOTFS/root/a1-6339 && cp -a <this dir>/. $ROOTFS/root/a1-6339/"
fakeroot -i $FR -- $MAKEFS -t ffs -B little -s 1600m \
  -o version=1,bsize=32768,fsize=4096 <img> $ROOTFS
python3 capstone/tests/cheri-baseline/cheri-run.py <qemu-argv.txt> serial.log /root/a1-6339
```

The vehicle used was `~/cheri-clean` (SDK, bbl, kernel and a 1.6 GB UFS1 image);
`~/cheri/output/cheribsd-riscv64-purecap.img` is missing, which is why the argv
file under `~/cheri/xlang-run/` does not work as it stands. The image was built to
a scratch path so the working one stays untouched.

## Still missing

The Capstone column. The domain port does not reach Ruby yet; see
`benchmarks/mruby/README.md`.
