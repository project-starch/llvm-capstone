declare void @sink(i64,i64,i64,i64,i64,i64,i64, i256) addrspace(200)
define void @split_i256(i256 %v) addrspace(200) {
  call addrspace(200) void @sink(i64 0,i64 0,i64 0,i64 0,i64 0,i64 0,i64 0, i256 %v)
  ret void
}

; The callee side: the same split argument received indirectly.
define i64 @recv_i256(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i256 %v) addrspace(200) {
  %t = trunc i256 %v to i64
  %h = lshr i256 %v, 192
  %u = trunc i256 %h to i64
  %r = add i64 %t, %u
  ret i64 %r
}
