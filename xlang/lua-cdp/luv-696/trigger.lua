local uv = require('luv')
local req = assert(uv.fs_scandir('.'))
local work = assert(uv.new_work(function(_entries)
  local _uv = require('luv')
  while true do if not _uv.fs_scandir_next(_entries) then break end end
end, function() end))
work:queue(req)
req = nil
collectgarbage("collect")   -- free the scandir uv_fs_t while the worker uses it
uv.run()
print("DONE")
