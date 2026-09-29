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
assert('Hash lookup with entries deleted by an eql? callback') do
  # Regression for GHSA-2778-fvwg-5m8w: a delete touches the count and the
  # slot's key and nothing else, so it reallocates nothing and the reentry
  # guard, which watched the capacity and the pointers, did not see it. A
  # lookup that had already read the count then walked past the end of the
  # entry array and handed what it found there to eql? as a key. Each entry
  # below has to answer with the exception, never by reading out of bounds.
  evil = Class.new do
    def initialize(h, keys) @h, @keys = h, keys end
    def eql?(other) @keys.each { |k| @h.delete(k) }; false end
    def hash; 0 end
  end
  ar = lambda { {"k0" => 0, "k1" => 1, "k2" => 2, "k3" => 3, "k4" => 4} }
  ar_keys = ["k1", "k2", "k3", "k4"]

  # Hash#[], and the two that reach the same scan through it.
  h = ar.call
  assert_raise(RuntimeError) { h[evil.new(h, ar_keys)] }
  h3 = ar.call
  assert_raise(RuntimeError) { h3.key?(evil.new(h3, ar_keys)) }

  # A store reads the array the same way before it writes.
  h4 = ar.call
  assert_raise(RuntimeError) { h4[evil.new(h4, ar_keys)] = 9 }

  # An HT-form hash reaches it too.
  t = {}
  20.times { |i| t["h#{i}"] = i }
  assert_raise(RuntimeError) { t[evil.new(t, (1...20).map { |i| "h#{i}" })] }

  # A delete and an add together leave the count where it was, which is what
  # the guard reads, so the bound on the entry array is what has to answer for
  # this one. Whether the add also moves the array, and so is seen by the
  # guard after all, is up to the allocator, so the lookup is asked only to
  # finish: either it raises or it answers, never reads past the end.
  swapper = Class.new do
    def initialize(h) @h = h end
    def eql?(other) @h.delete("k1"); @h["zz"] = 99; false end
    def hash; 0 end
  end
  m = ar.call
  begin
    assert_nil(m[swapper.new(m)])
  rescue RuntimeError
    # the guard saw the array move; either way nothing was read out of bounds
  end
  assert_true(m.size >= 4)

  # A hash nothing touched during the lookup still answers.
  q = {"a" => 1, "b" => 2}
  assert_equal(1, q["a"])
  assert_equal(2, q.size)
  assert_equal(2, q.rehash.size)
end

if $fails.empty?
  p ["PASS"]
else
  $fails.each { |f| p f }
end
