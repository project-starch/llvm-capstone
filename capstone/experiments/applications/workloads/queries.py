"""Generate a PostgreSQL single-user session with checked transient batches."""
import argparse
p = argparse.ArgumentParser()
p.add_argument('size', type=int)
p.add_argument('batches', type=int)
a = p.parse_args()
print('SET statement_timeout = 0;')
for epoch in range(a.batches):
    rows = a.size * (4 if epoch == a.batches // 2 else 1)
    print('BEGIN;')
    print('CREATE TEMP TABLE transient AS SELECT i, repeat(md5(i::text), 3) AS value FROM generate_series(0, %d) AS i;' % (rows-1))
    print('CREATE INDEX transient_value ON transient(value);')
    print("SELECT 'EXP-OK postgres ' || count(*)::text || ' ' || sum(i)::text AS oracle FROM transient;")
    # Native context counters are an independent application-level ledger.
    print("SELECT sum(total_bytes) AS context_reserved, sum(used_bytes) AS context_used FROM pg_backend_memory_contexts;")
    print('DROP TABLE transient; COMMIT;')
    print("SELECT sum(total_bytes) AS context_reserved, sum(used_bytes) AS context_used FROM pg_backend_memory_contexts;")
