# c52faebb7, with a GROWING demand so the data stack reallocates on many of the
# calls rather than extending once and staying large.
wide = "def wide_frame(d)\n" + (0...200).map { |i| "  v#{i} = #{i}\n" }.join +
       "  d > 0 ? wide_frame(d - 1) : v0 + v199\nend\n"
eval(wide)
class D
  def initialize(d) = @d = d
  def +(other)
    wide_frame(@d)
    :from_plus
  end
end
bad = 0
1.upto(90) do |d|
  r = D.new(d) + 1
  bad += 1 unless r == :from_plus
end
p ["non-symbol results", bad]
p ["PASS"] if bad == 0
