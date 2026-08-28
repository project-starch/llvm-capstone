# A1 -- mruby issue 6339, Array#delete.
#
# THE BLIND SPOT, in one script. mrb_ary_delete keeps the removed element in a
# local `ret` that the GC does not know about ("protect return value" is what the
# fix commit is called). The element's `==` runs Ruby, which runs the GC, the
# object is swept while delete is still running, and its slot comes back off the
# PAGE FREE LIST as a String.
#
# Nothing leaves the GC page. No malloc, no free, so:
#   * CHERI purecap sees an in-bounds access through a tagged capability;
#   * revocation has nothing to revoke, because nothing was ever freed;
#   * ASAN reports nothing at all, for the same reason.
# The only oracle is the WRONG ANSWER, which is why this returns a number the host
# can check rather than printing.
#
# VERSION. This needs a master build between the (re-)introduction of the C
# `mrb_ary_delete` and 0972c8477 (2024-09-09). NO RELEASE CARRIES IT: 3.2.0,
# 3.3.0 and 3.4.0 all still have the Ruby-level Array#delete, so the catalogue's
# old "affected <= 3.3.0" was wrong in the most misleading direction. Verified by
# building `0972c8477^` and mruby master and running this file on both.
#
# THE ALLOCATION BURST IS LOAD-BEARING, and is the whole reason the first version
# of this file did not work. Without it the freed slot has not been recycled yet
# and every oracle answers 1 on both builds; with it the answer separates 10 out
# of 10. The first version appeared to work only because the four oracle
# expressions it evaluated allocated enough between themselves to recycle the slot
# -- an oracle that depends on its own evaluation order is not an oracle.
$i = 0
class C
  def ==(other)
    GC.start
    ($i += 1) == 3
  end
end

a = 5.times.map { C.new }
x = a.delete(C.new)
GC.start

# Recycle the freed slot deliberately rather than by accident.
200.times { |k| "filler#{k}" }
GC.start

# 1 if the object survived as a C instance, which is correct.
# 2 if the slot came back as something else, which is the bug.
# instance_of? and inspect both separate; is_a?, x.class == C and x.class.to_s
# all answer 1 on BOTH builds and must not be used.
# The host reads a PRINTED MARKER, not the exit code. This build has no
# Kernel#exit -- mruby-exit is not in the gembox -- and an exit-code oracle
# therefore reported NoMethodError as rc=1, which is indistinguishable from
# "the answer was correct". The sanity file caught that; without it three clean
# rc=1 lines would have read as "CHERI misses nothing" in all three configs.
#   A1RESULT=1  the object survived as a C instance  -> correct
#   A1RESULT=2  the slot came back as something else -> WRONG ANSWER, a MISS
#   no marker + rc>=128                              -> CHERI caught it
puts "A1RESULT=#{x.instance_of?(C) ? 1 : 2} class=#{x.class}"
