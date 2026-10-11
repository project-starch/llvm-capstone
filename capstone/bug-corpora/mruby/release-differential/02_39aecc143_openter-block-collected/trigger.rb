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
assert('OP_ENTER keeps the block alive while it lays out a short argument list') do
  # A call that passes fewer positional arguments than the `*rest` and post
  # parameters span has its post arguments moved to the end of that span,
  # and when nothing but the required arguments came, or they came packed in
  # one array with a keyword hash after it, that is the register the block
  # arrived in.  The empty `rest` is allocated right after, with the block
  # held in a C local and nowhere the marker looks, so a collection landing
  # on that allocation freed the `Proc`, and what the method then read as its
  # block was whatever the cell was reused for.  Enough calls for the
  # collector to land there.
  def gc_test_short_args(x, *r, y, &blk); blk; end
  def gc_test_short_args_kw(x, *r, y, **o, &blk); blk; end
  args = [1, 2]
  kw = {}
  bad = 0
  i = 0
  while i < 5000
    b = gc_test_short_args(1, 2) { :c }
    bad += 1 unless b.call == :c
    b = gc_test_short_args_kw(*args, **kw) { :c }
    bad += 1 unless b.call == :c
    i += 1
  end
  assert_equal 0, bad
end

if $fails.empty?
  p ["PASS"]
else
  $fails.each { |f| p f }
end
