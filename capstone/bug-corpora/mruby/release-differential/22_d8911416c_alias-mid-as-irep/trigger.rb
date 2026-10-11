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

assert 'Method#parameters and #arity on aliased methods' do
  # Regression: an alias proc carries the original method's name in body.mid
  # (not an irep), with `upper` pointing at the original proc. Both
  # mrb_proc_parameters (mruby-proc-ext) and mrb_proc_arity (core src/proc.c)
  # used to fall into their irep branch for alias procs and dereference body.mid
  # as an mrb_irep* -> SEGV / misaligned read. Both must resolve through `upper`.
  #
  # The literals below match CRuby for these positional/optional/rest/block
  # signatures (where mruby and CRuby agree), so this is independent ground
  # truth, not merely "alias == original".
  c = Class.new {
    def f0; end
    def f1(a); end
    def fopt(a, b = 1); end
    def frest(a, *b); end
    def fblk(a, &b); end
    def fmix(a, b = 1, *c, &d); end
    alias_method :a0,    :f0
    alias_method :a1,    :f1
    alias_method :aopt,  :fopt
    alias_method :arest, :frest
    alias_method :ablk,  :fblk
    alias_method :amix,  :fmix
  }
  cases = [
    # name,   parameters,                                            arity
    [:a0,    [],                                                       0],
    [:a1,    [[:req, :a]],                                             1],
    [:aopt,  [[:req, :a], [:opt, :b]],                                -2],
    [:arest, [[:req, :a], [:rest, :b]],                               -2],
    [:ablk,  [[:req, :a], [:block, :b]],                               1],
    [:amix,  [[:req, :a], [:opt, :b], [:rest, :c], [:block, :d]],     -2],
  ]
  cases.each do |name, params, arity|
    u = c.instance_method(name)
    assert_equal params, u.parameters
    assert_equal arity,  u.arity
    # the bound Method goes through the same proc paths and must agree
    b = c.new.method(name)
    assert_equal params, b.parameters
    assert_equal arity,  b.arity
  end
  # alias must equal the original it points at (parameters and arity)
  { a0: :f0, a1: :f1, aopt: :fopt, arest: :frest, ablk: :fblk, amix: :fmix }.each do |al, orig|
    assert_equal c.instance_method(orig).parameters, c.instance_method(al).parameters
    assert_equal c.instance_method(orig).arity,      c.instance_method(al).arity
  end

  # Alias-of-an-alias collapses to one proc at creation; must still resolve.
  chain = Class.new {
    def orig(x, y) end
    alias_method :a1, :orig
    alias_method :a2, :a1
  }
  assert_equal [[:req, :x], [:req, :y]], chain.instance_method(:a2).parameters
  assert_equal 2, chain.instance_method(:a2).arity

  # Aliasing a C method does NOT create an alias proc (it reuses the original
  # cfunc method), so it must behave exactly like the original and never crash.
  c2 = Class.new(String) { alias_method :up2, :upcase }
  assert_equal String.instance_method(:upcase).parameters,
               c2.instance_method(:up2).parameters
  assert_equal String.instance_method(:upcase).arity,
               c2.instance_method(:up2).arity
end

assert 'Method/UnboundMethod on C-defined (native) methods' do
  # C methods have no irep: arity/parameters come from the packed argument spec
  # (caspec) and source_location is always nil. This exercises the
  # MRB_PROC_CFUNC_P branches of mrb_proc_arity / mrb_proc_parameters and the
  # cfunc path of method_search_vm -- none of which the suite covered before.

  # source_location is nil for genuinely C-defined methods (not mrblib ones).
  assert_nil "x".method(:upcase).source_location
  assert_nil 1.method(:+).source_location
  assert_nil [].method(:push).source_location
  assert_nil String.instance_method(:upcase).source_location

  # arity: documented values for these core C methods.
  assert_equal 0,  "x".method(:upcase).arity   # no args
  assert_equal 1,  1.method(:+).arity          # one required
  assert_equal(-1, [].method(:push).arity)     # variadic (rest)
  assert_equal(-1, [].method(:first).arity)    # optional

  # parameters: always an Array of Arrays, never crashes; a no-arg C method
  # gives []. C-method parameter *kinds* are an approximation (names are absent,
  # and required args surface as :opt for non-strict procs), so we assert the
  # stable shape rather than pinning exact kind labels -- except :rest, which is
  # meaningful: a variadic C method must expose a rest parameter.
  assert_equal [], "x".method(:upcase).parameters
  [1.method(:+), [].method(:push), {}.method(:[]), [].method(:first)].each do |m|
    ps = m.parameters
    assert_true ps.is_a?(Array)
    ps.each { |p| assert_true p.is_a?(Array) }
  end
  assert_true [].method(:push).parameters.any? { |entry| entry[0] == :rest }

  # identity / metadata on a C method.
  m = "abc".method(:upcase)
  assert_equal String,  m.owner
  assert_equal :upcase, m.name
  assert_equal "abc",   m.receiver
  assert_equal "#<Method: String#upcase>", m.to_s
  assert_equal "#<UnboundMethod: String#upcase>", String.instance_method(:upcase).to_s

  # behaviour: the C function actually runs via call / [] / bind / bind_call /
  # unbind+rebind.
  assert_equal 5,  2.method(:+).call(3)
  assert_equal 5,  2.method(:+)[3]
  assert_equal 5,  Integer.instance_method(:+).bind_call(2, 3)
  assert_equal 5,  Integer.instance_method(:+).bind(2).call(3)
  assert_equal 11, 5.method(:+).unbind.bind(10).call(1)

  # eql?: equal only when bound to the SAME receiver object and same definition.
  s = "cat"
  assert_true  s.method(:upcase) == s.method(:upcase)
  assert_false s.method(:upcase) == "cat".method(:upcase)  # distinct receivers
  assert_false s.method(:upcase) == s.method(:downcase)

  # super_method resolves across C methods (Integer#to_s -> BasicObject#to_s).
  sm = 5.method(:to_s).super_method
  assert_false sm.nil?
  assert_equal :to_s, sm.name

  # binding a C UnboundMethod to an incompatible receiver still raises cleanly.
  assert_raise(TypeError) { String.instance_method(:upcase).bind(42) }
end
if $fails.empty?
  p ["PASS"]
else
  $fails.each { |f| p f }
end
