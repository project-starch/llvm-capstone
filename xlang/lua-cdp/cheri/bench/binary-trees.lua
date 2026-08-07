-- The Computer Language Benchmarks Game — binary-trees, Lua 5.x
local function BottomUpTree(depth)
  if depth > 0 then
    depth = depth - 1
    return { BottomUpTree(depth), BottomUpTree(depth) }
  else
    return { }
  end
end
local function ItemCheck(tree)
  if tree[1] then
    return 1 + ItemCheck(tree[1]) + ItemCheck(tree[2])
  else
    return 1
  end
end
local N = tonumber(arg and arg[1]) or 8
local mindepth = 4
local maxdepth = math.max(mindepth + 2, N)
do
  local sd = maxdepth + 1
  io.write(string.format("stretch tree of depth %d\t check: %d\n", sd, ItemCheck(BottomUpTree(sd))))
end
local longlived = BottomUpTree(maxdepth)
for depth = mindepth, maxdepth, 2 do
  local iterations = 2 ^ (maxdepth - depth + mindepth)
  local check = 0
  for i = 1, iterations do check = check + ItemCheck(BottomUpTree(depth)) end
  io.write(string.format("%d\t trees of depth %d\t check: %d\n", iterations, depth, check))
end
io.write(string.format("long lived tree of depth %d\t check: %d\n", maxdepth, ItemCheck(longlived)))
