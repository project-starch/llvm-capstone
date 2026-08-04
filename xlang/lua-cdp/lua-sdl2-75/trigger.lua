-- lua-SDL2 #75 — cross-domain use-after-free of a native SDL_Window.
--
-- Two coupled objects:
--   1. Lua-GC handle : a Window userdata (CommonUserdata) whose __gc, when
--      mustdelete==1, calls SDL_DestroyWindow(data).
--   2. Native resource: the SDL_Window* returned by SDL_CreateWindow.
--
-- The bug (src/window.c, hitTestCallback): each time SDL invokes the hit-test,
-- the callback does commonPush(cd->L, "p", WindowName, win), which builds a
-- SECOND Window userdata over the SAME native SDL_Window with mustdelete=1.
-- When that duplicate is collected, its __gc runs SDL_DestroyWindow on the
-- window the ORIGINAL userdata + the event loop still use -> use-after-free.
-- Fix PR #77 (commit 96491c0) uses commonPushUserdata + mustdelete=0 instead.
--
-- SDL only invokes the hit-test on a real left ButtonPress inside the window
-- (X11 ProcessHitTest); a synthetic SDL event will not do, so we drive a real
-- X button event with xdotool (window is at a known position; DISPLAY comes
-- from xvfb-run). Then collectgarbage() forces the duplicate's __gc.

local SDL = require("SDL")
assert(SDL.init({ SDL.flags.Video }), "SDL video init failed")

local win = assert(SDL.createWindow{
    title = "luasdl2-75", x = 0, y = 0, width = 320, height = 240,
    flags = { SDL.window.Shown },
})

local HIT = 0
win:setHitTest(function(w, pt)
    HIT = HIT + 1
    return SDL.hitTestResult.Normal   -- do not start a WM move; the dup udata is already built
end)

local function pump() SDL.pumpEvents(); for e in SDL.pollEvent() do end end

for _ = 1, 30 do pump() end            -- let the window map
SDL.delay(200)

local deadline = SDL.getTicks() + 5000
while HIT == 0 and SDL.getTicks() < deadline do
    os.execute("xdotool mousemove 160 120 click 1 >/dev/null 2>&1")
    for _ = 1, 30 do pump(); SDL.delay(20) end
end

print("HITCOUNT " .. HIT)
if HIT == 0 then
    print("BLOCKED-NOHIT: hit-test never fired (no X button event reached the window)")
    os.exit(3)
end

print("BEFORE id=" .. win:getID())

-- The hit-test left a SECOND Window userdata over the same SDL_Window on the
-- Lua stack; hitTestCallback popped it, so it is now unreferenced garbage.
collectgarbage("collect")
collectgarbage("collect")

SDL.clearError()
local id_after = win:getID()           -- USE of the window after the dup's __gc may have freed it
local err = SDL.getError()
print("AFTER id=" .. id_after .. " err=[" .. err .. "]")

if id_after == 0 or err:find("Invalid window") then
    print("WINDOW-DESTROYED-WHILE-LIVE")   -- vulnerable: the duplicate's __gc killed a live window
else
    print("WINDOW-ALIVE")                  -- fixed: the duplicate had mustdelete=0
end
