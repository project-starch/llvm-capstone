# Control and probe run records

The result lines of every negative-control and attribution-probe run, so the
per-case claims in `../README.md` can be checked against something in the tree
rather than against aggregates alone. The review of #214 asked for this.

Three files per run, and nothing else:

- `verdicts.tsv` -- one row per case. The Capstone runner writes
  `case arm verdict rc cause last pc address untagged_value`; the CheriBSD
  runner writes `case arm rc signal si_code last`, where a fault is a non-empty
  `si_code`.
- `control-kind.tsv` -- `variant` or `stub` per case, which is what separates a
  strong control from a weak one.
- `run.meta` -- the arm, the capacity or binary, the image hash, the case
  timeout, the `ONLY` subset, any `TRIGGER=` override, and the run's own
  verdict on itself (`negative_control`, `negative_control_detections`,
  `negative_control_not_run`, `aborted`).

Per-case logs are not here. SCHEMA rule 6: results are summaries, never
captures.

A run whose name carries `negctl` is a negative control; `probe` is a
diagnostic run with `TRIGGER=` set, which replaces the trigger with one script
from the case directory and records the override in `run.meta`.

## Four runs have no record here, and that is rule 4

A run that produced no result line produced no measurement, so there is nothing
to commit for it. Named so the gaps in the list are not a mystery:

| run | why |
|---|---|
| `spatial-negctl-20261010-061310` | aborted before the first case |
| `sublet-negctl-20261010-061311` | the same |
| `spatial-negctl-20261010-093428` | boot refused: another Capstone VM held the lock |
| `boundary-negctl-20261010-131714` | refused by the guest-binary gate, which was right -- the default `CHERI_PYTHON` pointed at the wrong build |
