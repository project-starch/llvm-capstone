# MPY-T20: esp32: gc.collect() causes panic
Source: #6988, https://github.com/micropython/micropython/issues/6988  
Upstream state: closed, first seen 2021-03-04

**BLOCKED. Upstream status unresolved, see below.**

## The defect

gc.collect() panics, collector state inconsistent with the IDF heap.

**Temporal: uncertain.** an ESP32 collector panic with no fix commit and no mechanism recorded upstream.

Class `premature-free`, CWE-416, in `ports/esp32 + py/gc.c`. Scope `port-heap`, so a second allocator is involved.

## What Capstone does about it

`traps_unmodified` = **unclear**. A second allocator is involved and its behaviour has not been examined.

`traps_if_gc_cap_aware` = unclear, which is a prediction about a capability-aware
collector that does not exist yet. Not evidence.

## Measured

Not run. Needs the esp32 idf heap alongside the collector.

## Reproducing

Not reproducible with the current setup: needs the ESP32 IDF heap alongside the collector.
