import sys
class NotAModule:
    pass
sys.modules["fakemod"] = NotAModule()
from fakemod import *
print("T03 survived")
