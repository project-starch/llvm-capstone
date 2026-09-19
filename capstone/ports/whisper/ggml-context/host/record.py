"""Record a native Whisper transcription only after stock output agrees."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import wave

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path)
p.add_argument("--native-build", required=True, type=Path)
p.add_argument("--model", required=True, type=Path)
p.add_argument("--audio", required=True, type=Path)
p.add_argument("--repeat", type=int, default=1)
a = p.parse_args()


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


if (
    digest(a.model)
    != "921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f"
):
    p.error("tiny.en model checksum mismatch")
if not 1 <= a.repeat <= 10:
    p.error("repeat must be 1..10")
a.output.mkdir(parents=True, exist_ok=False)
audio = a.output / "input.wav"
with wave.open(str(a.audio), "rb") as source:
    params = source.getparams()
    frames = source.readframes(source.getnframes())
with wave.open(str(audio), "wb") as dest:
    dest.setparams(params)
    dest.writeframes(frames * a.repeat)
trace = a.output / "trace.bin.partial"
hashes = {}
for arm in ("stock", "recorded"):
    exe = a.native_build / f"whisper-{arm}/bin/whisper-cli"
    env = dict(os.environ)
    env.pop("WG_RECORD_PATH", None)
    if arm == "recorded":
        env["WG_RECORD_PATH"] = str(trace.resolve())
    with (a.output / f"{arm}.log").open("wb") as log:
        subprocess.run(
            [
                str(exe),
                "-m",
                str(a.model),
                "-f",
                str(audio),
                "-t",
                "1",
                "-np",
                "-otxt",
                "-of",
                str(a.output / arm),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    hashes[arm] = digest(exe)
stock = (a.output / "stock.txt").read_bytes()
if not stock.strip() or stock != (a.output / "recorded.txt").read_bytes():
    raise SystemExit("stock/recorded transcript mismatch")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common/host"))
from port_trace import inspect_trace

inspection = inspect_trace(trace, expected_format="whisper.ggml-context", replay=True)
count = inspection["trace"]["records"]
trace.rename(a.output / "trace.bin")
(a.output / "manifest.json").write_text(
    json.dumps(
        {
            "model_sha256": digest(a.model),
            "audio_sha256": digest(audio),
            "source_audio_sha256": digest(a.audio),
            "repeat": a.repeat,
            "threads": 1,
            "trace_sha256": digest(a.output / "trace.bin"),
            "transcript_sha256": hashlib.sha256(stock).hexdigest(),
            "executables_sha256": hashes,
            "events": count,
            "trace": inspection["trace"],
        },
        indent=2,
    )
    + "\n"
)
print(f"PASS stock/recorded transcript; {count} allocator events")
