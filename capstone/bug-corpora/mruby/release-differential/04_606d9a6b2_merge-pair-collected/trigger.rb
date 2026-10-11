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
assert('Hash scans that carry a pair into a set the hash no longer holds') do
  # A scan that stores the pair it is standing on somewhere else hands it to a
  # set, and the set asks the key for its hash code before it stores anything.
  # Ruby there can take the pair out of the hash the scan is reading, and the
  # only reference left is the one the scan holds in C, which the collector
  # does not scan. The pair has to survive the set that is carrying it.
  thief = Class.new do
    attr_accessor :armed
    def initialize(h) @h, @armed = h, false end
    def hash
      if @armed
        @armed = false
        @h.delete(self)
        @h[:added] = :added_value
        GC.start
      end
      42
    end
    def eql?(other) equal?(other) end
    # The value is built here and not in the assertion so that returning drops
    # it from the arena: what keeps it alive from then on is the entry alone,
    # which is what the callback takes away.
    def self.armed_hash
      h = {}
      20.times { |i| h[i] = i }
      k = new(h)
      h[k] = "the value only that entry holds"
      k.armed = true
      [h, k]
    end
  end

  # The hash being written into holds enough entries to be indexed, which is
  # what asks a key for its hash code in the first place. `armed` goes back
  # down when the callback has run, so the assertion says the set reached it
  # and the value is not being answered from a walk that never ran Ruby.
  h2, k = thief.armed_hash
  h1 = {}
  20.times { |i| h1[i + 100] = i }
  merged = h1.merge(h2)
  assert_false(k.armed)
  assert_equal("the value only that entry holds", merged[k])

  h, k = thief.armed_hash
  excepted = h.__except([:absent])
  assert_false(k.armed)
  assert_equal("the value only that entry holds", excepted[k])
end

if $fails.empty?
  p ["PASS"]
else
  $fails.each { |f| p f }
end
