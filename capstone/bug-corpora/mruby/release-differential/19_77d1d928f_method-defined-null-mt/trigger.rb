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
assert('Module#method_defined? reports the methods a listing reports') do
  mod = Module.new do
    def mpub; end
    private def mpriv; end
    protected def mprot; end
  end
  ahead = Module.new do
    private def ppriv; end
  end
  cls = Class.new do
    include mod
    prepend ahead
    def pub; end
    private def priv; end
    protected def prot; end
    class << self
      def spub; end
      private def spriv; end
      protected def sprot; end
    end
  end
  sub = Class.new(cls)

  # public and protected methods are matched, private ones are not
  assert_true  cls.method_defined?(:pub)
  assert_false cls.method_defined?(:priv)
  assert_true  cls.method_defined?(:prot)

  # wherever the method is found
  assert_false sub.method_defined?(:priv)
  assert_true  sub.method_defined?(:prot)
  assert_true  cls.method_defined?(:mpub)
  assert_false cls.method_defined?(:mpriv)
  assert_true  cls.method_defined?(:mprot)
  assert_false cls.method_defined?(:ppriv)

  # a singleton class reports its own methods the same way
  sclass = class << cls; self; end
  assert_true  sclass.method_defined?(:spub)
  assert_false sclass.method_defined?(:spriv)
  assert_true  sclass.method_defined?(:sprot)

  # the private methods mruby itself defines are not reported either
  assert_false cls.method_defined?(:initialize)
  assert_false cls.method_defined?(:method_missing)

  # a name with no method behind it is not put to respond_to_missing?, which
  # answers for a receiver rather than for what a module defines
  answering = Class.new do
    def respond_to_missing?(name, include_private = false)
      true
    end
  end
  assert_false answering.method_defined?(:no_such_method)
  assert_true  answering.new.respond_to?(:no_such_method)
end

if $fails.empty?
  p ["PASS"]
else
  $fails.each { |f| p f }
end
