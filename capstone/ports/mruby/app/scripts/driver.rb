# Runs smoke.rb's checks, then each benchmark script from its file, with what
# the script prints collected and reduced to a line count and a checksum, so
# binary output (so_mandelbrot's PBM) compares as text.
dir = ARGV[0] || "/mnt/host/files"
eval(File.read("#{dir}/smoke.rb"))
$cap = +""
module Kernel
  def print(*a) = a.each { |x| $cap << x.to_s }
  def puts(*a) = (a.empty? ? $cap << "\n" : a.flatten.each { |x| s = x.to_s; $cap << s; $cap << "\n" unless s.end_with?("\n") }; nil)
  def p(*a) = a.each { |x| $cap << x.inspect << "\n" }
end
%w(bm_mandel_term bm_so_lists bm_hash_access bm_so_mandelbrot).each do |b|
  $cap = +""
  r = eval(File.read("#{dir}/#{b}.rb"))
  sum = 0; $cap.each_byte { |c| sum = (sum * 31 + c) % 1_000_000_007 }
  $stdout.write "BENCH #{b} bytes=#{$cap.bytesize} lines=#{$cap.count("\n")} sum=#{sum} ret=#{r.inspect[0, 40]}\n"
end
$stdout.write "DRIVER_DONE\n"
