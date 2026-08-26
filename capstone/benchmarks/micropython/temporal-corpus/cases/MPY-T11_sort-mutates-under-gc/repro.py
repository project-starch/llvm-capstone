# MPY-T11 / upstream 17941. Segfaults on stock. Ordered LAST because a fault
# ends the boot and everything after it is lost.
import gc
class check:
    def __init__(self, a1, a2):
        self.arr1 = a1
        self.arr2 = a2
    def __lt__(self, other):
        self.arr1.clear()
        self.arr2.extend([777] * 5)
        gc.collect()
        self.arr2.clear()
        return True
d1 = []
d2 = []
c = [check(d1, d2) for _ in range(3)]
d1.extend(c)
d2.extend(c)
d1.sort()
print("T11 survived")
