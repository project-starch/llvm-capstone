try:
    import os, vfs

    vfs.VfsFat
except (ImportError, AttributeError):
    print("SKIP")
    raise SystemExit


class RAMBDevSparse:
    SEC_SIZE = 512

    def __init__(self, blocks):
        print(blocks)
        self.blocks = blocks
        self.data = {}

    def readblocks(self, n, buf):
        print(f"readblocks {n} {buf}")
        buf[:] = bytearray(1 + self.SEC_SIZE)

    def ioctl(self, op, arg):
        if op == 4:
            return self.SEC_SIZE

bdev = RAMBDevSparse(40)
fs = vfs.VfsFat(bdev)
