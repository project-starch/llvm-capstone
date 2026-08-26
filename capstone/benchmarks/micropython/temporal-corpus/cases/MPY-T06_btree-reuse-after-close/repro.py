# MPY-T06 / upstream 12543: reuse a btree object after close().
import btree, io, sys
f = io.BytesIO()
db = btree.open(f)
db[b"k"] = b"v"
db.close()
sys.stderr.write("T06 closed\n")
try:
    v = db[b"k"]
    sys.stderr.write("T06 read-after-close returned %r\n" % v)
except Exception as e:
    sys.stderr.write("T06 EXC %s\n" % type(e).__name__)
sys.stderr.write("T06 survived\n")
