declare void @sink(i64,i64,i64,i64,i64,i64,i64, i256) addrspace(200)
define void @split_i256(i256 %v) addrspace(200) {
  call addrspace(200) void @sink(i64 0,i64 0,i64 0,i64 0,i64 0,i64 0,i64 0, i256 %v)
  ret void
}
