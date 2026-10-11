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
assert("Hash#key keeps the key it answers with across the comparison") do
  # The scan hands the value to `==`, and the Ruby there can delete the pair it
  # is standing on. The key the scan would answer with is then held by the scan
  # in C, which the collector does not scan, so the collection the same Ruby
  # sets off takes it.
  thief = Class.new do
    def initialize(h, k) @h, @k = h, k end
    def ==(other)
      if @h
        h, k, @h, @k = @h, @k, nil, nil
        h.delete(k)
        h[:added] = :added_value
        GC.start
      end
      true
    end
    # Built here and not in the assertion so that returning drops the stored
    # key from the arena: what holds it from then on is the entry alone. An
    # unfrozen String key is stored as a frozen copy, which is the object the
    # entry holds and the delete below takes away.
    def self.armed_hash
      h = {}
      20.times { |i| h[i] = i }
      k = "the key only that entry holds"
      h[k] = new(h, k)
      h
    end
  end

  assert_equal("the key only that entry holds", thief.armed_hash.key(:target))
end

assert("Hash#slice keeps the value it carries across the set") do
  # Reading the pair out asks the key for its hash code, and storing it in the
  # result asks a second time. The Ruby answering that second ask can delete
  # the pair out of the receiver, and what holds the value then is the C local
  # carrying it, which the collector does not scan.
  thief = Class.new do
    attr_reader :fired
    def initialize(h) @h, @asks = h, 0 end
    def arm(n) @fire_at, @asks, @fired = n, 0, false end
    def hash
      @asks += 1
      if @asks == @fire_at
        @fired = true
        h, @h = @h, nil
        h.delete(self)
        h[:added] = :added_value
        GC.start
      end
      42
    end
    def eql?(other) equal?(other) end
    def self.armed_hash
      h = {}
      20.times { |i| h[i] = i }
      k = new(h)
      h[k] = "the value only that entry holds"
      [h, k]
    end
  end

  h, k = thief.armed_hash
  # A result asks for a hash code only once it has an index of its own, which
  # is what the keys ahead of this one are for. The second ask is the set, so
  # having fired says the value below is answered from the path being covered.
  keys = (0...17).to_a
  keys.push(k)
  k.arm(2)
  sliced = h.slice(*keys)
  assert_true(k.fired)
  assert_equal("the value only that entry holds", sliced[k])
end

assert("Hash#slice! keeps the value it removed across the set") do
  # The delete takes the pair out of the receiver, and the set that files it
  # under the same key asks that key for its hash code. Anything the Ruby
  # answering that allocates can collect the value on the way, since what
  # holds it between the two calls is the C local carrying it.
  collector = Class.new do
    attr_reader :asks
    def initialize; @asks = 0 end
    def arm; @asks = 0 end
    def hash
      @asks += 1
      GC.start
      7
    end
    def eql?(other) equal?(other) end
    def self.armed_hash
      h = {}
      20.times { |i| h[i] = i }
      k = new
      h[k] = "the value only that entry holds"
      k.arm
      [h, k]
    end
  end

  h, k = collector.armed_hash
  removed = h.slice!(0)
  # Two asks and no more: the delete and the set into the result. The scan for
  # the keys to keep asks nothing, since one key to keep is a hash in list
  # shape, which compares with `eql?` alone. A result too small to be indexed
  # would not ask either, and the value below would be answered from a walk
  # that never ran Ruby.
  assert_equal(2, k.asks)
  assert_equal("the value only that entry holds", removed[k])
end

if $fails.empty?
  p ["PASS"]
else
  $fails.each { |f| p f }
end
