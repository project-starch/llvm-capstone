# Row 7 probe -- exercises the code path the spec describes, under maximum GC
# pressure. It does NOT fault. See target.md for why, and why that is the
# expected outcome rather than a broken trigger.
#
# The spec describes Row 7 as "UAF: freed during GCD in mrb_bint_reduce (bigint
# gem), then read by the VM". mrb_bint_reduce() is reachable from Ruby only as
# Rational(bignum, bignum) -> rational_new_b() -> mrb_bint_reduce(), so that is
# what this drives, with the GC tuned as aggressively as the runtime allows and a
# churning heap to keep the collector busy.

GC.interval_ratio = 1
GC.step_ratio = 1

junk = []
3000.times do |i|
  junk << ("x" * 48)
  # Force the bigint path: numerator and denominator both exceed a machine word,
  # with varying magnitudes so the GCD does real work and allocates.
  Rational(2**(200 + (i % 11)), 2**(201 + (i % 7)))
  junk.shift if junk.size > 64
  GC.start if (i % 250).zero?
end

puts "completed without fault (expected -- see target.md)"
