"""The result contract for bug-corpus runs: one Observation per case and arm, one judge, one bundle.

WHY THIS EXISTS. Each corpus runner used to decide on its own what a row means, in its own words
(`PASS:fault`, `detected`, `silent`, `differential`, a SIGPROT exit code), and the per-bug table
then had to translate every vocabulary back into "caught" or "missed" -- defaulting anything it
did not recognise to "missed", which is the no-data-reads-as-zero failure. A runner now only
REPORTS what it saw, as an Observation. Whether that is a catch is decided here, once, by rules
that have their own tests (tools/test_verdicts.py).

THE THREE VERDICTS, and what each one needs:

  CAUGHT      the case reached its defective access, the arm faulted, and the fault is tied to
              the defect -- by the labelled probe, by the function the case declares, or by a
              paired control that ran the same statement without the defect and completed.
  MISSED      the case reached its defective access, ran to completion without a fault, and
              every control this run declared for the arm behaved as the arm says it must. A
              silence on an arm whose controls did not run says nothing about the arm.
  NO-READING  anything else, always with one reason from REASONS. Never counted as either.

The judge is a pure function of the Observation and the arm's declaration (tools/arms.json), so
a reading can be re-judged from its bundle without re-running anything.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
import subprocess
from pathlib import Path

CAUGHT, MISSED, NO_READING = "CAUGHT", "MISSED", "NO-READING"
VERDICTS = (CAUGHT, MISSED, NO_READING)

# Every way a run can fail to be a reading. Closed: an unknown reason is an error, not a row.
REASONS = {
    "infra": "the run produced no usable output: boot, launcher or timeout",
    "build-failed": "there is no image for this case on this arm",
    "not-reached": "the case never showed it reached its defective access",
    "setup-fault": "the fault came before the case reached its defective access",
    "control-failed": "a control in this run did not behave as the arm declares",
    "unattributed": "a fault, but nothing ties it to the defect",
    "out-of-denominator": "the case declares it cannot run on this arm",
    "inconclusive": "the case reached its access and then neither completed nor faulted",
}

# How a fault is tied to the defect, strongest first. `control` is a separate run of the same
# image on the same input minus the defect; it is what a case without a labelled instruction or
# a single faulting function relies on.
ATTRIBUTIONS = ("probe", "function", "control")

ARMS_FILE = Path(__file__).resolve().parent / "arms.json"


@dataclass
class Fault:
    cause: int
    pc: int
    symbol: str = None          # function the pc lies in, from the image's own symbols
    offset: int = None


@dataclass
class Control:
    """One control run in the same invocation as the cases, against the same build. What it
    SHOULD do is not recorded here: that is the configuration's, in arms.json, so an adapter
    cannot declare its own controls passed."""
    name: str
    observed: str               # "fault", "complete" or "none" (did not run / no output)
    evidence: str = ""


@dataclass
class Observation:
    """What one run of one case on one arm showed. Facts only; no verdict."""
    case: str                   # the case directory name
    arm: str
    infra: str = None           # a REASONS key when the run itself failed; set by the adapter
    reached: bool = False       # the case's own marker before the defective access was seen
    reach_evidence: str = ""
    completed: bool = False     # ran to its normal end
    fault: Fault = None
    attribution: str = None     # an ATTRIBUTIONS entry when the fault is tied to the defect
    attribution_evidence: str = ""
    controls: list = field(default_factory=list)
    image_sha256: str = None
    notes: str = ""

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, d):
        d = dict(d)
        d["fault"] = Fault(**d["fault"]) if d.get("fault") else None
        d["controls"] = [Control(**c) for c in d.get("controls", [])]
        return cls(**d)


def load_arms(path=ARMS_FILE):
    return json.loads(Path(path).read_text())["arms"]


def judge(obs, arm_spec):
    """(verdict, reason, evidence). reason is a REASONS key for NO-READING, else None.

    arm_spec is the configuration's entry in arms.json.

    The order is the rule: a run that is not a reading must never fall through to a verdict.
    """
    if obs.infra:
        if obs.infra not in REASONS:
            raise ValueError(f"unknown NO-READING reason {obs.infra!r}")
        return NO_READING, obs.infra, obs.notes
    declared = arm_spec["controls"]
    unknown = [c.name for c in obs.controls if c.name not in declared]
    if unknown:
        raise ValueError(f"controls {unknown} are not declared for this configuration")
    bad = [c for c in obs.controls if c.observed != declared[c.name]]
    if bad:
        return NO_READING, "control-failed", "; ".join(
            f"{c.name}: expected {declared[c.name]}, observed {c.observed}" for c in bad)
    if obs.fault is not None:
        if not obs.reached:
            return NO_READING, "setup-fault", (
                f"cause={obs.fault.cause} in {obs.fault.symbol or '?'} before the case's marker")
        if obs.attribution not in ATTRIBUTIONS:
            return NO_READING, "unattributed", (
                f"cause={obs.fault.cause} pc={obs.fault.pc:#x} in {obs.fault.symbol or '?'}"
                f"; {obs.attribution_evidence or 'no probe, declared function or control'}")
        return CAUGHT, None, (f"cause={obs.fault.cause} pc={obs.fault.pc:#x} in "
                              f"{obs.fault.symbol or '?'}; by {obs.attribution}: "
                              f"{obs.attribution_evidence}")
    if not obs.reached:
        return NO_READING, "not-reached", obs.notes or "no marker before the defective access"
    if not obs.completed:
        return NO_READING, "inconclusive", obs.notes or "reached, then neither completed nor faulted"
    # A silence is a reading only on an arm that proved, in this run, that it is what it says.
    required = set(arm_spec.get("controls_for_missed", []))
    present = {c.name for c in obs.controls}
    if not required <= present:
        return NO_READING, "control-failed", (
            "silent, but the arm's controls did not run: missing " + ", ".join(sorted(required - present)))
    return MISSED, None, f"reached ({obs.reach_evidence}) and completed with no fault" + (
        f"; {obs.notes}" if obs.notes else "")


# ---- shared pieces every adapter needs -----------------------------------------------------

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# The host's line for an application-domain fault, as capstone-exec prints it.
FAULT_LINE = re.compile(r"domain fault cause=(\d+) pc=(0x[0-9a-f]+) address=(0x[0-9a-f]+).*?code=(0x[0-9a-f]+)-")


class Symbols:
    """Function symbols of one ELF image, and its link base, for pc attribution."""

    def __init__(self, llvm_bin, image):
        out = subprocess.run([str(Path(llvm_bin) / "llvm-nm"), "--print-size", "-n", "--defined-only",
                              str(image)], capture_output=True, text=True, check=True).stdout
        self.funcs = []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[2] in "tTwW":
                self.funcs.append((int(parts[0], 16), int(parts[1], 16), parts[3]))
        hdr = subprocess.run([str(Path(llvm_bin) / "llvm-readelf"), "-l", str(image)],
                             capture_output=True, text=True, check=True).stdout
        loads = [int(m.group(1), 16) for m in re.finditer(r"^\s*LOAD\s+\S+\s+(0x[0-9a-f]+)", hdr, re.M)]
        if not loads or not self.funcs:
            raise RuntimeError(f"{image}: no LOAD segment or no function symbols")
        self.base = min(loads)

    def lookup(self, runtime_pc, code_start):
        pc = runtime_pc - code_start + self.base
        for addr, size, name in self.funcs:
            if addr <= pc < addr + max(size, 1):
                return name, pc - addr
        return None, None


def domain_fault(text, symbols=None):
    """The application-domain fault in `text`, as a Fault, or None."""
    m = FAULT_LINE.search(text or "")
    if not m:
        return None
    cause, pc, code = int(m.group(1)), int(m.group(2), 16), int(m.group(4), 16)
    fn, off = symbols.lookup(pc, code) if symbols else (None, None)
    return Fault(cause=cause, pc=pc, symbol=fn, offset=off)


# ---- the bundle ----------------------------------------------------------------------------

# What every bundle's inputs.json must say about the platform, so a reader can tell which build
# and which machine produced a cell. A value may be "unrecorded" -- said, not omitted.
PLATFORM_KEYS = ("compiler", "qemu", "firmware", "kernel", "runner")


def write_bundle(out, corpus, arm, rows, inputs):
    """results/<stamp>-<arm>/: verdicts.jsonl (one judged Observation per line) + inputs.json.

    rows: [(Observation, (verdict, reason, evidence))]. The judged verdict is stored beside the
    Observation so a reader does not need this module, and re-judging can check it.
    """
    missing = [k for k in PLATFORM_KEYS if k not in inputs.get("platform", {})]
    if missing:
        raise ValueError(f"inputs.platform must say (or say 'unrecorded' for): {missing}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "verdicts.jsonl").open("w") as f:
        for obs, (verdict, reason, evidence) in rows:
            f.write(json.dumps({"observation": obs.to_json(), "verdict": verdict,
                                "reason": reason, "evidence": evidence}, sort_keys=True) + "\n")
    tally = {}
    for _, (verdict, reason, _) in rows:
        key = verdict if verdict != NO_READING else f"{verdict}:{reason}"
        tally[key] = tally.get(key, 0) + 1
    record = dict(inputs, contract="verdicts-v1", corpus=corpus, arm=arm, cases=len(rows),
                  tally=dict(sorted(tally.items())))
    (out / "inputs.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def read_bundle(path):
    """[(Observation, verdict, reason, evidence)] from a bundle directory."""
    rows = []
    for line in (Path(path) / "verdicts.jsonl").read_text().splitlines():
        d = json.loads(line)
        rows.append((Observation.from_json(d["observation"]), d["verdict"], d["reason"], d["evidence"]))
    return rows
