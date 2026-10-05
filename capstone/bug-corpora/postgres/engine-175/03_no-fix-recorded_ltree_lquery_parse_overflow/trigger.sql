CREATE EXTENSION ltree;
SELECT (repeat('x', 1000) || repeat('|' || repeat('x', 1000), 65))::lquery;
