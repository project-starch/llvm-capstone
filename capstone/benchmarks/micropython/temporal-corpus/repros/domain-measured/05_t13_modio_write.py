# MPY-T13 / upstream 19075, verbatim. A write() callback enlarges the buffer it
# was handed, reallocating under a caller that still holds the old pointer.
import io, json
class S(io.IOBase):
    def write(self, buf):
        buf += buf
json.dump([], S())
print("T13 survived")
