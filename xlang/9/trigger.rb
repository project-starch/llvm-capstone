# Create a proc that has a large string literal in its irep pool, dynamically
p = eval("Proc.new { \"asdfasdfasdf adaf asdf asdfa sdf asdfasdfasdfa sdf\"[1..-2] }")

# Get the shared substring pointing to the dynamic irep pool string
$sub = p.call

# Nil-out the proc reference so it can be garbage collected
p = nil

# Force garbage collection to free the dynamic RProc and its dynamic mrb_irep,
# which forcefully frees the pool string!
GC.start

# Attempt to print the shared substring (reads from the freed string!)
puts "Value: #{$sub}"
