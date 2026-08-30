# Proves the purecap interpreter runs and that the marker channel works. Without
# it a uniform result cannot be told from "mruby never ran" -- which is exactly
# what happened with the first, exit-code-based version of this harness.
a = 5.times.map { |i| i * 2 }
puts "SANITY_OK=#{a.length == 5 && a[4] == 8 ? 7 : 9}"
