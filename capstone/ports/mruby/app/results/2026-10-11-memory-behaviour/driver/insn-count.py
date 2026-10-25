#!/usr/bin/env python3
"""insn-count.py LLVM_OBJCOPY IMAGE...: CDERIVE and CREVOKE words in each image's .text.
The encodings are those build-sublet.sh gates on: opcode 0x5b, funct3 1, funct7 0x51 / 0x52."""
import struct, subprocess, sys, tempfile, hashlib
objcopy = sys.argv[1]
for image in sys.argv[2:]:
    with tempfile.NamedTemporaryFile() as t:
        subprocess.run([objcopy, '-O', 'binary', '--only-section=.text', image, t.name], check=True)
        data = open(t.name, 'rb').read()
    words = struct.unpack('<%dI' % (len(data) // 4), data[:len(data) // 4 * 4])
    count = lambda f7: sum(1 for w in words if w & 0xfe00707f == (f7 << 25) | (1 << 12) | 0x5b)
    print(hashlib.sha256(open(image, 'rb').read()).hexdigest(), f'cderive={count(0x51)} crevoke={count(0x52)}', image)
