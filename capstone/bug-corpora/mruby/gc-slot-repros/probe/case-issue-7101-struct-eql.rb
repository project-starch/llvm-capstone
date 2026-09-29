class Probe
  attr_accessor :victim, :donor
  def ==(_)
    @victim.replace_from(@donor)
    true
  end
end

klass = Struct.new(*(0...24).map { |i| "member_#{i}".to_sym }) do
  def replace_from(other)
    initialize_copy(other)
  end
end

probe = Probe.new
left  = klass.new(probe, *((1...24).map { |i| i }))
right = klass.new(Object.new, *((1...24).map { |i| i }))
donor = klass.new(*((0...24).map { |i| 10_000 + i }))
probe.victim = left
probe.donor = donor
left == right
