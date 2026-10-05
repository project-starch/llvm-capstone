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

# Envs on a task stack must be detached before the stack is freed

assert("closure escaping a closed task survives GC") do
  t = Task.new(name: "escaper") do
    a1 = 1; a2 = 2; a3 = 3; a4 = 4; a5 = 5; a6 = 6
    $task_escaped_proc = -> { a1 + a2 + a3 + a4 + a5 + a6 }
    Task.current.suspend
  end
  Task.pass
  assert_equal 21, $task_escaped_proc.call
  t.terminate
  t.close                 # frees the task's stack
  GC.start                # marks the escaped env; must not read freed memory
  junk = []
  i = 0
  while i < 200
    junk << "x" * 64      # reuse the freed stack region
    i += 1
  end
  GC.start
  assert_equal 21, $task_escaped_proc.call
  $task_escaped_proc = nil
end

assert("closure escaping a task whose context is reinitialized survives GC") do
  t = Task.new(name: "reinit") do
    b1 = 7; b2 = 8; b3 = 9
    $task_escaped_proc2 = -> { b1 + b2 + b3 }
    Task.current.suspend
  end
  Task.pass
  assert_equal 24, $task_escaped_proc2.call
  t.terminate
  # Reuse the task's context for another proc: the old stack is freed
  # inside mrb_task_init_context, with the escaped env still pointing at it.
  TaskTest.reinit_context(t) { 0 }
  GC.start
  junk = []
  i = 0
  while i < 200
    junk << "y" * 48
    i += 1
  end
  GC.start
  assert_equal 24, $task_escaped_proc2.call
  $task_escaped_proc2 = nil
  t.close
end

assert("closure escaping a synchronously executed proc survives GC") do
  result = TaskTest.run_sync do
    c1 = 10; c2 = 20
    $task_escaped_proc3 = -> { c1 + c2 }
    "sync-result"
  end
  # The teardown frees the temporary task's stack; both the returned
  # object and the escaped env must survive it.
  assert_equal "sync-result", result
  GC.start
  junk = []
  i = 0
  while i < 200
    junk << "z" * 48
    i += 1
  end
  GC.start
  assert_equal 30, $task_escaped_proc3.call
  assert_equal "sync-result", result
  $task_escaped_proc3 = nil
end
if $fails.empty?
  p ["PASS"]
else
  $fails.each { |f| p f }
end
