# Original reported loop trigger representing fuzzed inputs
i = 0
hash = {}
while i < 256
  hash['%d' % i] = i.to_s
  i += 1
end
