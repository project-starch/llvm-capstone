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
assert('Hash scans that read the entry back after a callback') do
  # The same delete-and-reinsert, reached from the scans that read an entry
  # again once a callback has returned: #inspect prints the value after the
  # key printed itself, the walk behind `**rest` stores the pair after the key
  # answered #==, and #rehash moves the pair after the #eql? that asked
  # whether an earlier key is the same one. A rehash reads two entries, and
  # the #eql? can vacate either: the one it is moving, or the one it matched.
  swapper = Class.new do
    attr_accessor :armed
    def initialize(h, victim) @h, @victim, @armed = h, victim, false end
    def fire
      return false unless @armed
      @armed = false
      @h.delete(@victim || self)
      @h[:added] = :added_value
      true
    end
    def inspect; fire; "swapper" end
    def ==(other) fire; false end
    def eql?(other)
      return true if (@victim.nil? || @victim.equal?(other)) && fire
      equal?(other)
    end
    def hash; 42 end
  end

  h = {}
  20.times { |i| h[i] = i }
  k = swapper.new(h, nil)
  h[k] = "value"
  k.armed = true
  assert_raise(RuntimeError) { h.inspect }
  h.keys.each { |x| assert_true(h.key?(x)) }

  h = {}
  20.times { |i| h[i] = i }
  k = swapper.new(h, nil)
  h[k] = "value"
  k.armed = true
  assert_raise(RuntimeError) { h.__except([:absent]) }
  h.keys.each { |x| assert_true(h.key?(x)) }

  # The list (AR) shape, where the duplicate is looked for by walking the
  # entries already moved.
  h = {first: "value"}
  k = swapper.new(h, nil)
  h[k] = "dup"
  k.armed = true
  assert_raise(RuntimeError) { h.rehash }
  h.keys.each { |x| assert_true(h.key?(x)) }

  # The indexed (HT) shape. Only an entry already moved is in the new index,
  # so that is the one a callback can take away, and it is the entry the
  # duplicate is about to be written into.
  h = {}
  20.times { |i| h[i] = i }
  first = swapper.new(h, nil)
  h[first] = "value"
  k = swapper.new(h, first)
  h[k] = "dup"
  k.armed = true
  assert_raise(RuntimeError) { h.rehash }
  # What a rehash stopped part way through leaves is what it left before: the
  # entries it had not reached yet are out of the index it is rebuilding.
  assert_true(h.key?(:added))
end

if $fails.empty?
  p ["PASS"]
else
  $fails.each { |f| p f }
end
