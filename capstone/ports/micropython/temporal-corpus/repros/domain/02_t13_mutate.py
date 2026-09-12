# MPY-T13 / upstream 19075, verbatim. Segfaults on stock MicroPython.
import io, json
class S(io.IOBase):
    def write(self, buf):
        buf += buf
json.dump([], S())
print("T13 survived")
