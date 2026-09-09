# Does MPY-T09 leave a genuinely dangling memoryview, or was the resize in place?
# Stock MicroPython does not crash on issue 18168's script at pin 2e3304a, so the
# question is whether the defect is absent or merely silent. This decides it.
ba = bytearray(b"abcdefghij")
mv = memoryview(ba)
ba[:] = ba + b"X" * 256          # resize while a view is active
ba[0:1] = b"A"                   # write through the bytearray
seen_ba, seen_mv = bytes(ba[0:1]), bytes(mv[0:1])
mv[0:1] = b"Z"                   # write through the stale view
print("ba[0]=%r mv[0]=%r  after mv write: ba[0]=%r mv[0]=%r"
      % (seen_ba, seen_mv, bytes(ba[0:1]), bytes(mv[0:1])))
if seen_ba != seen_mv and bytes(ba[0:1]) == b"A":
    print("DANGLING: the view addresses an orphaned buffer and remains writable")
else:
    print("IN-PLACE: no stale view, this row needs re-examining")
