$fails = []; $cur = nil
def assert(name, *x)
  $cur = name
  begin; yield
  rescue Exception => e; $fails << [name, "EXCEPTION", e.class.to_s, e.message]
  end
end
def _f(kind, *d) $fails << [$cur, kind] + d end
def assert_equal(a, b = :__none, m = nil)
  b = yield if b == :__none && block_given?
  _f("NOT_EQUAL", a.inspect, b.inspect) unless a == b
end
def assert_not_equal(a, b = :__none, m = nil)
  b = yield if b == :__none && block_given?
  _f("EQUAL", a.inspect) if a == b
end
def assert_true(v, m = nil)  _f("NOT_TRUE", v.inspect)  unless v == true  end
def assert_false(v, m = nil) _f("NOT_FALSE", v.inspect) unless v == false end
def assert_nil(v, m = nil)   _f("NOT_NIL", v.inspect)   unless v.nil?     end
def assert_not_nil(v, m = nil) _f("NIL") if v.nil? end
def assert_include(c, v, m = nil) _f("NOT_INCLUDE") unless c.include?(v) end
def assert_kind_of(k, v, m = nil) _f("NOT_KIND_OF", v.class.to_s) unless v.kind_of?(k) end
def assert_raise(*k)
  begin; yield; _f("NO_RAISE")
  rescue Exception => e
    _f("WRONG_RAISE", e.class.to_s) unless k.empty? || k.any? { |c| e.kind_of?(c) }
  end
end
def assert_nothing_raised(*a)
  begin; yield; rescue Exception => e; _f("RAISED", e.class.to_s, e.message); end
end
def skip(*a) end
def pass() end
def flunk(m = nil) _f("FLUNK", m.to_s) end
def assert_predicate(o, m, msg = nil) _f("NOT_PREDICATE", o.inspect, m.to_s) unless o.__send__(m) end
def assert_not_predicate(o, m, msg = nil) _f("PREDICATE", o.inspect, m.to_s) if o.__send__(m) end
def assert_operator(a, op, b, msg = nil) _f("NOT_OPERATOR", op.to_s) unless a.__send__(op, b) end
def assert_raise_with_message(k, msg, m = nil)
  begin; yield; _f("NO_RAISE")
  rescue Exception => e
    _f("WRONG_CLASS", e.class.to_s) unless e.kind_of?(k)
    _f("WRONG_MESSAGE", e.message.inspect, msg.inspect) unless e.message == msg
  end
end
def assert_raise_with_message_pattern(k, pat, m = nil)
  begin; yield; _f("NO_RAISE"); rescue Exception => e; end
end
assert('Struct#== and #eql? with a member that replaces the storage') do
  # A member's #== runs while both structs are being compared, and
  # initialize_copy() there moves the member storage out from under the walk.
  probe = Class.new do
    attr_accessor :victim, :donor
    def ==(_); @victim.replace_from(@donor); true; end
    alias eql? ==
  end
  k = Struct.new(*(0...24).map { |i| "m#{i}".to_sym }) do
    def replace_from(other); initialize_copy(other); end
  end
  donor = k.new(*(0...24).map { |i| 10_000 + i })

  [:==, :eql?].each do |op|
    p1 = probe.new
    left = k.new(p1, *(1...24).to_a)
    right = k.new(Object.new, *(1...24).to_a)
    p1.victim = left
    p1.donor = donor
    assert_false left.__send__(op, right)

    # and with the right operand replaced instead
    p2 = probe.new
    l2 = k.new(p2, *(1...24).to_a)
    r2 = k.new(Object.new, *(1...24).to_a)
    p2.victim = r2
    p2.donor = donor
    assert_false l2.__send__(op, r2)
  end
end

if $fails.empty?
  p ["PASS"]
else
  $fails.each { |f| p f }
end
