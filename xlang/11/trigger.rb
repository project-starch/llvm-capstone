# Generate deep nesting to trigger scope level overflow
for i in range(2, 500):
    eval("a#{i} = 'A' * 200")
c = "baz"

# 128 nested scopes
"a".instance_eval { "a".instance_eval {
  # ... nested scopes up to 128 levels
  puts c
} }
