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
assert('Hash scans with the matched entry vacated by an eql? callback') do
  # The same delete-and-reinsert from eql?, reached from the scans that read
  # the entry after the comparison: #assoc and #rassoc build a pair from it,
  # #== and #eql? compare its value. In #assoc the STORED key answers eql?,
  # in #rassoc the stored value does, and in #== / #eql? the key of the
  # receiver hash does while it is looked up in the other hash.
  swapper = Class.new do
    attr_accessor :armed
    def initialize(h, key) @h, @key, @armed = h, key, false end
    def eql?(other)
      if @armed
        @armed = false
        @h.delete(@key || self)
        @h[:added] = :added_value
      end
      other.class == self.class
    end
    def hash; 42 end
  end

  h = {}
  20.times { |i| h[i] = i }
  k = swapper.new(h, nil)
  h[k] = "value"
  k.armed = true
  assert_raise(RuntimeError) { h.assoc(swapper.new(h, nil)) }
  h.keys.each { |x| assert_true(h.key?(x)) }

  h = {}
  20.times { |i| h[i] = i }
  v = swapper.new(h, :k)
  h[:k] = v
  v.armed = true
  assert_raise(RuntimeError) { h.rassoc(swapper.new(h, nil)) }
  h.keys.each { |x| assert_true(h.key?(x)) }

  [:==, :eql?].each do |op|
    h1 = {}
    20.times { |i| h1[i] = i }
    k = swapper.new(h1, nil)
    h1[k] = "value"
    h2 = {}
    20.times { |i| h2[i] = i }
    h2[swapper.new(h2, nil)] = "value"
    k.armed = true
    assert_raise(RuntimeError) { h1.__send__(op, h2) }
    h1.keys.each { |x| assert_true(h1.key?(x)) }
  end
end

if $fails.empty?
  p ["PASS"]
else
  $fails.each { |f| p f }
end
