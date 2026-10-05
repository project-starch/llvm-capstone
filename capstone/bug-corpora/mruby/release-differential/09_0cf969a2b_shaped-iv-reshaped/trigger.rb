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
assert('Kernel#inspect survives a callback that grows the object mid-walk (GHSA-j6fq-xj4w-877x)') do
  # inspect calls each ivar value's own #inspect, which can run arbitrary
  # Ruby. Adding a new ivar to the object being inspected from in there
  # used to free its shaped storage (shaped_iv_set growing to a wider
  # block) out from under the walk still reading the old one.
  owner_class = Class.new do
    def initialize(mutator_class)
      @first = mutator_class.new(self)
      @second = "kept"
    end

    def grow
      @third = :grown
    end
  end
  mutator_class = Class.new do
    def initialize(owner)
      @owner = owner
    end

    def inspect
      @owner.grow
      "mutated"
    end
  end

  o = owner_class.new(mutator_class)
  s = o.inspect
  assert_include s, "@first=mutated"
  assert_include s, '@second="kept"'
end

assert('Kernel#inspect survives a callback that removes an ivar mid-walk (GHSA-j6fq-xj4w-877x)') do
  # remove_instance_variable de-shapes the object (shaped storage to a
  # plain table), freeing the same shaped block a growing callback frees;
  # the walk used to keep reading through it for the remaining keys.
  owner_class = Class.new do
    def initialize(remover_class)
      @a = remover_class.new(self)
      @b = "kept"
      @c = "also kept"
    end

    def drop_b
      remove_instance_variable(:@b)
    end
  end
  remover_class = Class.new do
    def initialize(owner)
      @owner = owner
    end

    def inspect
      @owner.drop_b
      "dropped"
    end
  end

  o = owner_class.new(remover_class)
  s = o.inspect
  assert_include s, "@a=dropped"
  assert_include s, '@c="also kept"'
end

if $fails.empty?
  p ["PASS"]
else
  $fails.each { |f| p f }
end
