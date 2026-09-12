# Does the filesystem stack actually RUN in a capability domain, or does it merely
# link? A block device written in Python, mkfs, mount, write, read back. If this
# returns, MPY-T14/MPY-T15 are a parent-build job and nothing more.
import vfs

class RAMBD:
    SEC = 512
    def __init__(self, n):
        self.n = n
        self.data = bytearray(n * self.SEC)
    def readblocks(self, b, buf):
        buf[:] = self.data[b * self.SEC:b * self.SEC + len(buf)]
    def writeblocks(self, b, buf):
        self.data[b * self.SEC:b * self.SEC + len(buf)] = buf
    def ioctl(self, op, arg):
        if op == 4: return self.n
        if op == 5: return self.SEC
        return 0

bd = RAMBD(64)
vfs.VfsFat.mkfs(bd)
fs = vfs.VfsFat(bd)
with fs.open("t", "w") as f:
    f.write("hello")
with fs.open("t", "r") as f:
    got = f.read()
print("VFS", got, len(bd.data))
