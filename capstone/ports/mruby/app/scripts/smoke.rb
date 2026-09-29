dir = ARGV[0] || "/mnt/host/files"
puts "M1 hello"
a = (1..2000).map { |i| i * 2 }; puts "M2 array #{a.size} #{a.sum}"
h = {}; 3000.times { |i| h["k#{i}"] = i }; puts "M3 hash #{h.size} #{h['k2999']}"
puts "M4 symbols #{[:foo, :bar, :"literal_sym"].map(&:to_s).join(',')} #{:foo.object_id == :foo.object_id}"
s = ""; 500.times { |i| s << "ab#{i}" }; puts "M5 string #{s.size} #{s[0, 10]}"
objs = []; 20000.times { |i| objs << [i, "s#{i}", {i => i}] }; objs = nil; GC.start; puts "M6 gc ok"
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
