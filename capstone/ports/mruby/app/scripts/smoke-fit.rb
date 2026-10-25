# smoke-fit.rb: smoke.rb with M6 at 6,000 entries instead of 20,000, for the Sublet build (MRBD_SUBLET=1) on vp2's
# flat lifetime table (64,001 usable records). M6 at 20,000 needs more than 65,536 records, so smoke.rb's control
# cannot complete there and no row of a Sublet arm is a reading. 6,000 is the largest N MEASURED at or under 75% of
# 64,001 (2026-10-22, QEMU generations kit, image 58b47ccb, plain GC): N=6,000 high_water 44,304 (69.2%), 7,000 51,433
# (80.4%), 8,000 58,563 (91.5%), 10,000 fails at 64,001 (the port README's record-pressure item has the series).
# Every other line is smoke.rb's. Pass it to the release corpus as run-virtual.py --smoke.
dir = ARGV[0] || "/mnt/host/files"
puts "M1 hello"
a = (1..2000).map { |i| i * 2 }; puts "M2 array #{a.size} #{a.sum}"
h = {}; 3000.times { |i| h["k#{i}"] = i }; puts "M3 hash #{h.size} #{h['k2999']}"
puts "M4 symbols #{[:foo, :bar, :"literal_sym"].map(&:to_s).join(',')} #{:foo.object_id == :foo.object_id}"
s = ""; 500.times { |i| s << "ab#{i}" }; puts "M5 string #{s.size} #{s[0, 10]}"
objs = []; 6000.times { |i| objs << [i, "s#{i}", {i => i}] }; objs = nil; GC.start; puts "M6 gc ok"
def fib(n) = n < 2 ? n : fib(n - 1) + fib(n - 2)
puts "M7 fib #{fib(20)}"
def deep(n) = n == 0 ? 0 : 1 + deep(n - 1)
puts "M8 deep #{deep(500)}"
begin; raise ArgumentError, "x"; rescue => e; puts "M9 rescue #{e.class}"; end
puts "M10 float #{(1.5 * 3).round(2)} #{10.fdiv(4)} #{2**70}"
File.open("#{dir}/out.txt", "w") { |f| 50.times { |i| f.puts "line #{i}" } }
puts "M11 file #{File.read("#{dir}/out.txt").lines.size}"
puts "M12 dir #{Dir.entries(dir).include?('smoke.rb')}"
puts "M13 pack #{[1, 2, 3].pack('C*').bytes.inspect} #{'%05.2f' % 3.14159}"
puts "SMOKE_DONE"
