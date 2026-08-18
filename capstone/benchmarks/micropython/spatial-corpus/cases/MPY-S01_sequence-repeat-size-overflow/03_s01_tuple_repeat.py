# MPY-S01 / upstream 19314, the tuple path (py/objtuple.c).
r = (1, 2, 3, 4, 5, 6, 7, 8) * (2 ** 61)
print("S01t", len(r))
