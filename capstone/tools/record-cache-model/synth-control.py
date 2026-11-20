import struct,sys
def W(kind,op,slot,cnt=1): return struct.pack('<Q',(((kind<<2)|op)<<56)|(cnt<<32)|slot)
C,A,R=0,3,4
seq=[W(7,0,0)]                                  # T
seq+= [W(C,0,s) for s in (0,256,0,512,256)]     # RC: m m h m m (4 miss); GC set 0 tags 0,2,4: m m h m h (3 miss)
seq+= [W(A,1,1)]                                # write slot 1 (line 0): RC miss (line 0 was evicted by 512? see below)
seq+= [W(C,0,768),W(C,0,1024)]                  # two more lines in RC set 0: evicts the dirty line -> 1 writeback
seq+= [W(A,3,8),W(A,1,1024)]                    # fresh table: flush
seq+= [W(C,0,0,5)]                              # run of 5 checks after flush: RC 1 miss 4 hits; NC 5
open(sys.argv[1],'wb').write(b''.join(seq))
