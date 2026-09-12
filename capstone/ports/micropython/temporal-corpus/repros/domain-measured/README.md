The exact six scripts that produced the six measured script-driven rows, plus the
sanity control, and nothing else. `repros/domain/` also holds diagnostic probes
(`01_t13_type.py`, `02_t13_mutate.py`); baking those changes the test indices and
therefore which row a result belongs to, which is why this directory exists
separately rather than each run.sh pointing at `repros/domain/`.
