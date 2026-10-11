# wmem-repros 18-21 under the chunk port (`sublet-chunks`), 2026-10-09

Cases 0-17 had a `sublet-chunks` reading and 18-21 only the region-granular `sublet` one; column 3
of the per-bug table ("Sublet in nested") is the chunk port on every case, so these four were run.
Built from the lane's tree with `-DWM_CHUNKS=ON`, run by `shared/run-defects.py --modes sublet`.
**4 of 4 fault at the labelled probe, as pre-registered (f7adf03a1009)**: cause 7 at the write probe for
18, 20 and 21, cause 5 at the read probe for 19 -- the chunk port bounds every chunk, so the
crossing out of it faults, as it does under the region-granular hooks.
