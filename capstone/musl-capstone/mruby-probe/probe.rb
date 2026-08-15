# The chunk the Lua probes use, in Ruby, so the two are comparable.
#
# Deliberately core-only: Array#[]=, Integer arithmetic and `while`. No Range,
# no block, no Enumerable, because those live in mrblib and in gems and would
# make a failure ambiguous between "the VM does not run" and "that method is
# not linked". Reaching 400 still needs the VM, the object system, the GC and
# an array that GROWS, which is a realloc that moves.
t = []
i = 1
while i <= 20
  t[i - 1] = i * i
  i += 1
end
t[19]
