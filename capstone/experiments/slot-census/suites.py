"""What a cell is: its suite, program, dataset and class, and which suites the figures show.

Cells are classed by what the allocator below would see:
  application  every heap object is a malloc (SPEC, Redis on libc malloc, SQLite with lookaside off, CPython with
               PYTHONMALLOC=malloc)
  hidden       the program's own allocator carves objects from blocks it mallocs (memcached's slabs,
               PostgreSQL's memory contexts, RocksDB's memtable arena, nginx's and Apache's pools, SpiderMonkey's
               garbage-collected heap in Firefox, SQLite's lookaside, CPython's pymalloc):
               the census counts the blocks, so the row is the count the allocator below sees with the program
               unmodified, not the number of objects
  stress       allocator stress tests (mimalloc-bench), reported apart"""
import re
from collections import namedtuple

Point = namedtuple("Point", "program value dataset")             # one mark of a figure column
Pair = namedtuple("Pair", "suite program dataset peak rate")     # a run with both a census peak and a rate


def classify(pass_name, provenance, cell):
    """(suite, program, dataset, class) for one cell"""
    if m := re.match(r"sqlite-(lookaside|nolookaside)-(\d+)$", cell):
        on = m[1] == "lookaside"
        return ("SQLite 3.22.0 speedtest1", "speedtest1, lookaside " + ("on" if on else "off"), f"size {m[2]}",
                "hidden" if on else "application")
    if m := re.match(r"ycsb-(redis|memcached)-(\d+)$", cell):
        if m[1] == "redis":
            return "Redis 7.2.16 (MALLOC=libc), YCSB", "redis-server", f"{int(m[2]):,} records", "application"
        return "memcached 1.6.45, YCSB", "memcached", f"{int(m[2]):,} records", "hidden"
    if cell in ("nginx", "httpd"):
        return ({"nginx": "nginx 1.20.2, wrk", "httpd": "Apache httpd 2.4.58, ab"}[cell], cell, "published setup",
                "hidden")
    if m := re.match(r"speedometer-([\d.]+)$", cell):
        return f"Firefox Nightly, Speedometer {m[1]}", "firefox", f"Speedometer {m[1]}", "hidden"
    if m := re.match(r"rocksdb-(\d+)$", cell):
        return "RocksDB db_bench", "db_bench", f"{int(m[1]):,} keys", "hidden"
    if m := re.match(r"parsec-(\w+)-(\w+)$", cell):
        return "PARSEC 3.0", m[2], m[1], "application"
    if m := re.match(r"postgres-(\d+)$", cell):
        return "PostgreSQL 17.6 pgbench", "postgres", f"scale {m[1]}", "hidden"
    if m := re.match(r"py-(pymalloc|malloc)-(.+)$", cell):
        return ("pyperformance (CPython 3.13.7)", m[2], f"PYTHONMALLOC={m[1]}",
                "hidden" if m[1] == "pymalloc" else "application")
    if m := re.match(r"octane-(.+)$", cell):
        return "Octane on QuickJS 2026-06-04", m[1], "Octane", "application"
    if m := re.match(r"mb-(.+)$", cell):
        return "mimalloc-bench", m[1], "bench.sh arguments", "stress"
    if re.match(r"\d{3}\.\w+_r$", cell):
        size = re.search(r"^size (\w+)$", provenance, re.M)
        return "SPEC CPU2017 rate", cell, size[1] if size else "?", "application"
    return "unclassified", cell, pass_name, "unclassified"


def dataset_order(dataset):
    """test < train < ref, else the number in the label"""
    if dataset in ("test", "train", "ref"):
        return ("test", "train", "ref").index(dataset), ""
    m = re.search(r"[\d,]+", dataset)
    return int(m[0].replace(",", "")) if m else 0, dataset


# The suites the figures show, one column each, named by their benchmark (user, 2026-10-27): label, input shown in
# the caption, unit, which rows belong to it.
COLUMNS = (
    ("SPEC CPU2017", "ref", "programs",
     lambda r: r["suite"] == "SPEC CPU2017 rate" and r["dataset"] == "ref"),
    ("PARSEC", None, "programs", lambda r: r["suite"] == "PARSEC 3.0"),
    ("pyperformance", "CPython 3.13, PYTHONMALLOC=malloc", "scripts",
     lambda r: r["suite"].startswith("pyperformance") and r["class"] == "application"),
    ("Octane", "QuickJS", "scripts", lambda r: r["suite"].startswith("Octane")),
    ("Firefox", "Nightly, Speedometer 2.1 and 3.1 (its largest process each), a lower bound: SpiderMonkey's heap is "
     "not counted", "runs",
     lambda r: r["suite"].startswith("Firefox")),
    ("speedtest1", "SQLite 3.22.0, lookaside off", "sizes",
     lambda r: r["suite"].startswith("SQLite") and r["class"] == "application"),
    ("Redis", "libc malloc, YCSB load + A + C, 10^5 to 1.6x10^7 records", "sizes",
     lambda r: r["suite"].startswith("Redis") and "YCSB" in r["suite"]),
)


def columns(rows, measure):
    """the figure columns of rows, each mark a run's measure (the key "peak" or "rate"), sorted by it"""
    cols = []
    for label, shown, unit, belongs in COLUMNS:
        sel = [r for r in rows if belongs(r)]
        if label == "PARSEC":   # each program at its largest measured input: native where intact, else simlarge
            native = {r["program"] for r in sel if r["dataset"] == "native"}
            sel = [r for r in sel if r["dataset"] == "native" or r["program"] not in native]
            fallback = sorted({r["program"] for r in sel if r["dataset"] != "native"})
            shown = "native" + (f" (simlarge for {', '.join(fallback)}: native input lost)" if fallback else "")
        best = {}   # a benchmark rerun in a second pass (the network ones run alone) counts once, at its maximum
        for r in sel:
            key = (r["program"], r["dataset"])
            if key not in best or r[measure] > best[key][measure]:
                best[key] = r
        cols.append({"label": label, "shown": shown, "unit": unit,
                     "points": sorted((Point(r["program"], r[measure], r["dataset"]) for r in best.values()),
                                      key=lambda p: p.value)})
    return cols


def pairs(peak_cols, rate_cols):
    """every run with both a census peak and an allocation rate (per 1,000 instructions), in column order"""
    rate = {(c["label"], p.program, p.dataset): p.value for c in rate_cols for p in c["points"]}
    return [Pair(c["label"], p.program, p.dataset, p.value, rate[(c["label"], p.program, p.dataset)])
            for c in peak_cols for p in c["points"] if (c["label"], p.program, p.dataset) in rate]
