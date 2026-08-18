# MPY-S01 / upstream 19314, the list path (py/objlist.c, py/sequence.c).
# Same overflow, a different allocation site and a different element width, so
# this is not a duplicate of 01: it asks whether the outcome depends on the path.
r = [1, 2, 3, 4, 5, 6, 7, 8] * (2 ** 61)
print("S01l", len(r))
