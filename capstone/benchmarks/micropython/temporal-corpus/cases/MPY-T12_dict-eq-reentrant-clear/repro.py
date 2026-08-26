# MPY-T12 / upstream 18619. Segfaults on stock. Expected to FAULT in the domain
# too, and for a reason that is NOT temporal safety, so it is ordered late.
d2 = {}
class Bomb:
    def __bool__(self):
        d2.clear()
        self.tmp = {0: 0}
        return True
class Key:
    def __hash__(self):
        return 0
    def __eq__(self, other):
        return Bomb()
d1 = {Key(): 1}
d2[Key()] = 1
d1 == d2
print("T12 survived")
