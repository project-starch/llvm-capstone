#!/bin/bash
# usage: kit-check.sh <bench>  -- rootfs content + Image byte-diff summary, G vs P1
B=${SPEC_BOARD_DIR:?set SPEC_BOARD_DIR to the spec/board directory}
G=$B/out-nat-g-hpmnest-frcdkms-b1/$1; P=$B/out-nat-p1-hpmnest-frcdkms-b1/$1; W=$(mktemp -d); rm -rf $W; mkdir -p $W/g $W/p
(cd $W/g && cpio -idm --quiet --no-absolute-filenames < $G/image/rootfs.cpio 2>/dev/null); (cd $W/p && cpio -idm --quiet --no-absolute-filenames < $P/image/rootfs.cpio 2>/dev/null)
diff -rq --no-dereference $W/g $W/p > $W/diff.txt 2>&1; echo "rootfs diff rc=$? files=$(find $W/p -type f | wc -l)"; head -5 $W/diff.txt
python3 - $G/image/Image $P/image/Image <<'PY'
import sys,zlib
a=open(sys.argv[1],'rb').read(); b=open(sys.argv[2],'rb').read(); print('Image sizes', len(a), len(b))
o=a.find(b'\x1f\x8b\x08'); assert o==b.find(b'\x1f\x8b\x08')
def glen(x):
    d=zlib.decompressobj(16+zlib.MAX_WBITS); out=d.decompress(x[o:]); return len(x)-o-len(d.unused_data), len(out)
(la,ua),(lb,ub)=glen(a),glen(b); print(f'gzip at {o:#x}: len {la} vs {lb} (d {lb-la}), unpacked {ua} vs {ub}')
n=min(len(a),len(b)); d=[i for i in range(n) if a[i]!=b[i] and not (o<=i<o+max(la,lb))]
print('bytes differing outside the gzip stream:', len(d))
above=[i for i in d if i>=((o+max(la,lb))|0xfff)+1]
import re
print('  bytes past the stream (incl. the shifted init tail):', len(above), 'pages', sorted(set(hex(i&~0xfff) for i in above)))
PY
rm -rf $W/g $W/p
