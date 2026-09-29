# Exercises slot birth, slot death, compaction, growth across the AR/HT boundary,
# dup and free -- every hook the adapter installs -- and checks the answers.
ok = true
h = {}
40.times { |i| h[i] = i * 2 }                 # grows AR -> HT, past XP_SLOTS
ok &&= (h.size == 40) && (h[39] == 78)
20.times { |i| h.delete(i) }                  # slot deaths
ok &&= (h.size == 20) && h[20] == 40 && h[0].nil?
h2 = h.dup
ok &&= (h2.size == 20) && (h2[25] == 50)
h.rehash
ok &&= (h.size == 20) && (h[30] == 60)
h.keys.each { |k| ok &&= (h[k] == k * 2) }
small = {}
10.times { |i| small["k#{i}"] = i }           # stays AR
5.times { |i| small.delete("k#{i}") }
ok &&= (small.size == 5) && (small["k7"] == 7)
small["new"] = 99
ok &&= (small["new"] == 99) && (small.size == 6)
100.times { |i| t = {}; 5.times { |j| t[j] = j }; t.delete(2); ok &&= (t.size == 4) }
ok &&= ({a: 1}.merge({b: 2}) == {a: 1, b: 2})
ok &&= ([[1, 2], [3, 4]].to_h == {1 => 2, 3 => 4})
p [ok ? "PASS" : "FAIL", h.size, small.size]
