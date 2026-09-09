# Safe half of the MPY-T13 diagnosis: report WHAT json.dump hands to write(),
# without mutating it. On stock this is a bytearray, which is why `buf += buf`
# mutates in place and reallocates under the caller. If the domain reports
# something immutable, the verbatim test never exercised the defect at all.
import io, json
seen = []
class S(io.IOBase):
    def write(self, buf):
        seen.append(type(buf).__name__)
        return len(buf)
json.dump([], S())
print("T13type", seen)
