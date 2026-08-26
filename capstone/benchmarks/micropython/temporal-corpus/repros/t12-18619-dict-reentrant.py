# MPY-T12 / upstream issue 18619, verbatim from the report.
# Stock behaviour at pin 2e3304a: SIGSEGV.
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
print("SURVIVED")
