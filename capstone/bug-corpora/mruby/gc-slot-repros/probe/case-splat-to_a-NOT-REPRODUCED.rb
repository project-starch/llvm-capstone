# 7b503f3a3: regs[a] = mrb_ary_splat(...) fixes the destination address BEFORE the
# call, so a user-defined to_a that re-enters the VM and grows the data stack makes
# the store land in the freed buffer. The commit names this trigger exactly.
wide = "def wide_frame(d)\n" + (0...220).map { |i| "  v#{i} = #{i}\n" }.join +
       "  d > 0 ? wide_frame(d - 1) : v0 + v219\nend\n"
eval(wide)
class G
  def initialize(d) = @d = d
  def to_a
    wide_frame(@d)
    [1, 2, 3]
  end
end
bad = 0
1.upto(80) do |d|
  a = [*G.new(d)]
  bad += 1 unless a == [1, 2, 3]
end
p ["wrong splats", bad]
p ["PASS"] if bad == 0
