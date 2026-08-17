# MPY-T23: Sporadic GuruMeditation, presumed related to TCP/IP LWIP
Source: #12638, https://github.com/micropython/micropython/issues/12638  
Upstream state: closed, first seen 2023-10-09

**BLOCKED. Upstream status unresolved, see below.**

## The defect

sporadic fault, buffer freed by the network stack still referenced.

Class `premature-free`, CWE-416, in `ports/esp32 lwip glue`. Scope `port-heap`, so a second allocator is involved.

## What Capstone does about it

`traps_unmodified` = **unclear**. A second allocator is involved and its behaviour has not been examined.

`traps_if_gc_cap_aware` = unclear, which is a prediction about a capability-aware
collector that does not exist yet. Not evidence.

## Measured

Not run. Closed upstream as unreproducible; needs the esp32 network stack.

## Reproducing

Not reproducible with the current setup: closed upstream as unreproducible; needs the ESP32 network stack.
