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
assert('Module#method_defined? with inherit false reports only the module\'s own methods') do
  mod = Module.new do
    def mpub; end
  end
  ahead = Module.new do
    def apub; end
  end
  cls = Class.new do
    include mod
    prepend ahead
    def pub; end
    private def priv; end
  end
  sub = Class.new(cls)

  # a method the class itself defines, wherever the walk would have found it
  assert_true  cls.method_defined?(:pub, false)
  assert_true  cls.method_defined?(:pub, true)
  assert_false cls.method_defined?(:priv, false)
  assert_false cls.method_defined?(:no_such_method, false)

  # methods that come from an ancestor, an included module, a prepended
  # module or Object are not the class's own
  assert_false sub.method_defined?(:pub, false)
  assert_true  sub.method_defined?(:pub)
  assert_false cls.method_defined?(:mpub, false)
  assert_true  cls.method_defined?(:mpub)
  assert_false cls.method_defined?(:apub, false)
  assert_true  cls.method_defined?(:apub)
  assert_false cls.method_defined?(:inspect, false)
  assert_true  cls.method_defined?(:inspect)

  # a module is asked the same way, about the modules it includes
  mod2 = Module.new { include mod }
  assert_true  mod.method_defined?(:mpub, false)
  assert_false mod2.method_defined?(:mpub, false)
  assert_true  mod2.method_defined?(:mpub)

  # a visibility changed in a subclass makes the method the subclass's own
  shown = Class.new(cls) { public :priv }
  assert_true  shown.method_defined?(:priv, false)
  assert_false cls.method_defined?(:priv)

  # a method undefined or left unimplemented is not there to be found, own or
  # inherited; see "Kernel#respond_to? with an unimplemented method"
  gone = Class.new(cls) { undef_method :pub }
  assert_false gone.method_defined?(:pub, false)
  assert_false gone.method_defined?(:pub)
  assert_false TestNotImplement.method_defined?(:gone, false)

  # inherit is read for truth, as in CRuby
  assert_true  sub.method_defined?(:pub, 1)
  assert_false sub.method_defined?(:pub, nil)
  assert_raise(ArgumentError) { cls.method_defined?(:pub, false, false) }
end

if $fails.empty?
  p ["PASS"]
else
  $fails.each { |f| p f }
end
