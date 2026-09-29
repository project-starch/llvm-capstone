# Same defect as string-strip-bang-uaf, with a base past the domain's embed limit
# (RSTRING_EMBED_LEN_MAX is 27 natively but 59 at 16-byte capabilities, so the
# 45-byte original is embedded in a domain and never becomes a shared view).
base = "." + (" " * 8) + ("abcdefghijklmnopqrstuvwxyz0123456789" * 3)
p ["base.len", base.length]
view = base[1..-1]
stripped = view.lstrip!
p ["stripped.ok", stripped == ("abcdefghijklmnopqrstuvwxyz0123456789" * 3)]
p ["base.intact", base == "." + (" " * 8) + ("abcdefghijklmnopqrstuvwxyz0123456789" * 3)]
p ["PASS"] if stripped == ("abcdefghijklmnopqrstuvwxyz0123456789" * 3) && base == "." + (" " * 8) + ("abcdefghijklmnopqrstuvwxyz0123456789" * 3)
