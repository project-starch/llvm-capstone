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
assert('Hash lookup with the matched entry vacated by an eql? callback') do
  # A companion to the case above, with the callback returning true. It deletes
  # the very entry the search matched and inserts another, which puts the count
  # back: the guard watches the count and the pointers and so sees nothing, and
  # the slot the search is standing on has been vacated all the same. Answering
  # from it hands #[] the value of an entry the collector no longer keeps and
  # lets #delete take it a second time, dropping the count below the entries the
  # table still holds. Each operation has to report the change, not answer from
  # the vacated slot.
  swapper = Class.new do
    def initialize(h) @h, @fired = h, false end
    def eql?(other)
      unless @fired
        @fired = true
        @h.delete(other)
        @h[:added] = :added_value
      end
      true
    end
    def hash; 42 end
  end
  # The indexed (HT) shape, and the AR shape with a leading hole: word boxing
  # reads an AR hash's first slot as the pointer the guard watches, so only a
  # hole ahead of that slot exposes the AR path the way HT is already exposed.
  ht = lambda { h = {}; 20.times { |i| h[i] = i }; h }
  ar = lambda { h = {}; 12.times { |i| h[i] = i }; h.delete(0); h }

  [ht, ar].each do |build|
    [ lambda { |h, k| h.delete(k) },
      lambda { |h, k| h[k] },
      lambda { |h, k| h[k] = :stored },
      lambda { |h, k| h.key?(k) } ].each do |op|
      h = build.call
      assert_raise(RuntimeError) { op.call(h, swapper.new(h)) }
      # What the table can find and what it iterates stay the same entries.
      h.keys.each { |k| assert_true(h.key?(k)) }
      assert_equal(h.key?(:added), h.keys.include?(:added))
    end
  end
end

if $fails.empty?
  p ["PASS"]
else
  $fails.each { |f| p f }
end
