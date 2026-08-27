# A1 -- mruby issue 6339, Array#delete.
#
# THE BLIND SPOT, in one script. The object returned by delete is swept while
# delete is still running, its slot comes back off the PAGE FREE LIST as a String,
# and the interpreter hands the wrong object back to Ruby.
#
# Nothing leaves the GC page. No malloc, no free, so:
#   * CHERI purecap sees an in-bounds access through a tagged capability;
#   * revocation has nothing to revoke, because nothing was ever freed;
#   * ASAN reports nothing at all, for the same reason.
# The only oracle is the WRONG ANSWER. That is why this returns a number the host
# can check rather than printing.
#
# Needs mruby <= 3.3.0; fixed by 0955539cf9bb and 0972c8477.
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

# 1 if the object survived as a C instance, which is correct.
# 2 if it came back as something else, which is the bug.
x.is_a?(C) ? 1 : 2
