# B5 (docs/plans/b0-silicon-delegated-runtime.md): mruby's own benchmarks, sized for the board, as driver.rb runs
# them -- each script read from its file and evaluated, what it prints captured and reduced to a byte count, a line
# count and a checksum, so the binary images (so_mandelbrot's PBM, ao_render's PPM) compare as one text line.
# Only those lines go to stdout, and stdout is what is compared with the native build's; the wall time of each
# benchmark goes to stderr, which is not compared.
#
# Usage: mruby b5-bench.rb <dir> [bench ...]   (dir holds the bm_*.rb files; default: every benchmark below)
#
# Each size is a textual substitution that must match exactly once, so a benchmark changed upstream fails here
# instead of silently running at its full size. bm_app_lc_fizzbuzz is left out: its size is the depth of a nested
# literal, not a parameter.
dir = ARGV[0] || "/tmp/b5"
SIZES = {
  "bm_mandel_term"   => [],
  "bm_fib"           => [["puts fib(37)", "puts fib(24)"]],
  "bm_so_lists"      => [["NUM = 300", "NUM = 3"], ["SIZE = 10000", "SIZE = 1000"]],
  "bm_so_mandelbrot" => [["size = 600 #", "size = 32 #"]],
  "bm_ao_render"     => [["Integer(ARGV[0] || 64)", "8"], ["NAO_SAMPLES = 8", "NAO_SAMPLES = 4"]],
}
benches = ARGV.size > 1 ? ARGV[1..] : SIZES.keys

$cap = +""
module Kernel
  private def print(*a) = (a.each { |x| $cap << x.to_s }; nil)
  private def puts(*a) = (a.empty? ? $cap << "\n" : a.flatten.each { |x| s = x.to_s; $cap << s; $cap << "\n" unless s.end_with?("\n") }; nil)
  private def p(*a) = (a.each { |x| $cap << x.inspect << "\n" }; a.size <= 1 ? a[0] : a)
  private def putc(c) = ($cap << (c.is_a?(Integer) ? (c & 0xff).chr : c.to_s[0]); nil)
  private def printf(*a) = ($cap << format(*a); nil)
end

# Evaluated inside a method so the benchmark's own top-level locals cannot overwrite the loop's (ao_render
# assigns one named like the loop variable).
def run_bench(src) = eval(src)

benches.each do |b|
  subs = SIZES[b] or raise "unknown benchmark #{b}"
  src = File.read("#{dir}/#{b}.rb")
  subs.each do |from, to|
    n = src.split(from, -1).size - 1
    raise "#{b}: '#{from}' occurs #{n} times, not once" unless n == 1
    src = src.sub(from, to)
  end
  $cap = +""
  t0 = Time.now
  r = run_bench(src)
  ms = ((Time.now - t0) * 1000).round
  sum = 0; $cap.each_byte { |c| sum = (sum * 31 + c) % 1_000_000_007 }
  $stdout.write "BENCH #{b} bytes=#{$cap.bytesize} lines=#{$cap.count("\n")} sum=#{sum} ret=#{r.inspect[0, 40]}\n"
  $stderr.write "TIME #{b} #{ms} ms\n"
end
$stdout.write "B5_BENCH_DONE\n"
