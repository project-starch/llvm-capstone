# MPY-T27: usock: if connect() fails, socket is considered open -> not cleaned up
Source: #5272, https://github.com/micropython/micropython/issues/5272  
Upstream state: open, first seen 2019-10-28

**NOT REPRODUCIBLE HERE. The trigger cannot be expressed in this domain.**

## The defect

failed connect() leaves the socket considered open, never cleaned up.

Class `lifetime-order`, CWE-772, in `ports/esp32 usocket`. Scope `port-heap`, so a second allocator is involved.

## What Capstone does about it

`traps_unmodified` = **unclear**. A second allocator is involved and its behaviour has not been examined.

`traps_if_gc_cap_aware` = not-trapped, which is a prediction about a capability-aware
collector that does not exist yet. Not evidence.

## Measured

Not run. Needs the esp32 network stack.

## Reproducing

Not reproducible with the current setup: needs the ESP32 network stack.
