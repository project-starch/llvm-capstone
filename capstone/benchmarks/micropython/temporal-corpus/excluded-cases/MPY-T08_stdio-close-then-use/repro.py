import sys
try:
    v1 = sys.stdout
    v1.close()
    sys.stderr.write("T08 survived close\n")
    v1.write("x")                     # use after close
    sys.stderr.write("T08 survived write-after-close\n")
except Exception as e:
    sys.stderr.write("T08 EXC " + str(type(e)) + "\n")
