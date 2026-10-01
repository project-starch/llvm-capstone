SELECT count(*), group_concat(name, ',') FROM t;
PRAGMA integrity_check;
