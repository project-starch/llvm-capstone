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
assert('pattern matching - a key that moves the subject') do
  # A key's #== and #eql? run while the pattern helpers are walking the key
  # array and the subject hash, and both were held as raw storage across the
  # call. Rehashing the subject is refused by the Hash implementation's own
  # check; replacing the key array is allowed, so the walk has to keep up.
  moving = Class.new do
    attr_accessor :owner, :arr, :armed, :found
    def initialize(id); @id = id; @armed = false; @found = false; end
    def hash; @id; end
    def ==(_)
      if @armed
        @armed = false
        @arr.replace(Array.new(64) { |j| self.class.new(j + 100) }) if @arr
        2_000.times { |i| @owner[i + 10_000] = i } if @owner
      end
      @found
    end
    alias eql? ==
  end

  # `in {a: 1}` reaches __pat_values, which walks the key array. The lookup
  # keys are separate objects from the stored ones, so the hash has to ask
  # #eql? rather than settle it by identity.
  h1 = {}
  30.times { |i| h1[moving.new(i + 1)] = i }
  keys = Array.new(30) { |i| moving.new(i + 1) }
  keys.each { |k| k.arr = keys; k.found = true }
  keys[0].armed = true
  # The replacement keys are not in the hash, so the lookup fails; what
  # matters is that the walk follows the array it was given rather than the
  # buffer it started with.
  assert_false h1.__pat_values(keys)

  # `**rest` reaches __except, which walks the subject hash as well
  h2 = {a: 1}
  ks = Array.new(30) { |i| k = moving.new(i + 1); h2[k] = k; k }
  ks.each { |k| k.owner = h2 }
  ks[0].armed = true
  assert_raise(RuntimeError) { h2.__except([:a]) }
end

if $fails.empty?
  p ["PASS"]
else
  $fails.each { |f| p f }
end
