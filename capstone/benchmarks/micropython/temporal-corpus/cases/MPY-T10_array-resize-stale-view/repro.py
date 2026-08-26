# MPY-T10 / upstream 18171, ADAPTED: the original uses open()+readinto() as the
# vehicle for writing through the stale view, and this domain has no filesystem.
# The write is done directly instead. The defect under test is unchanged: resizing
# an array that has an active memoryview leaves the view on orphaned storage.
from array import array
a = array('I', [0] * 4)
mv = memoryview(a)
a.extend([1] * 2048)             # resize while the view is live
a[0] = 0x41414141                # write through the array
diverged = 1 if mv[0] != a[0] else 0
mv[0] = 0x5A5A5A5A               # the use-after-free WRITE
unreached = 1 if a[0] == 0x41414141 else 0
print("T10", diverged, unreached)
