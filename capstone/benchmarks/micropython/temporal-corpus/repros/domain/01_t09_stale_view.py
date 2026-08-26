# MPY-T09 / upstream 18168, reduced to a verdict the domain can return.
# Resize a bytearray that has an active memoryview, then ask whether the view
# still addresses the SAME storage. Divergence proves the view is dangling.
ba = bytearray(b"abcdefghij")
mv = memoryview(ba)
ba[:] = ba + b"X" * 256          # resize while the view is live
ba[0:1] = b"A"                   # write through the bytearray
diverged = 1 if mv[0] != ba[0] else 0
mv[0:1] = b"Z"                   # the use-after-free WRITE
unreached = 1 if ba[0] == 0x41 else 0   # the bytearray was not touched by it
print("T09", diverged, unreached)
