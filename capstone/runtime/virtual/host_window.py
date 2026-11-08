"""Host-side timing of a trusted guest workload, excluding boot and validation.

Serial delivery adds endpoint jitter. CPU time is process-wide Linux /proc time,
with tick resolution; neither number is a simulated target cycle count.
"""
import os
from pathlib import Path
import time


def sample(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return time.monotonic_ns(), int(fields[11]) + int(fields[12])


class HostWindow:
    def __init__(self, pid, sampler=sample):
        self.pid, self.sampler = pid, sampler
        self.pending = b''
        self.begin = self.end = None

    def feed(self, data):
        lines = (self.pending + data).split(b'\n')
        self.pending = lines.pop()[-65536:]
        markers = [line.rstrip(b'\r') for line in lines
                   if line.rstrip(b'\r') in (b'HOST_WINDOW_BEGIN', b'HOST_WINDOW_END')]
        if len(markers) > 1:
            raise ValueError('Timing markers coalesced; increase workload duration')
        for marker in markers:
            if marker == b'HOST_WINDOW_BEGIN':
                if self.begin is not None:
                    raise ValueError('Duplicate timing begin')
                self.begin = self.sampler(self.pid)
            else:
                if self.begin is None or self.end is not None:
                    raise ValueError('Missing begin or duplicate end')
                self.end = self.sampler(self.pid)

    def result(self):
        if self.begin is None or self.end is None:
            raise ValueError('Incomplete host timing window')
        wall = (self.end[0] - self.begin[0]) / 1e9
        ticks = self.end[1] - self.begin[1]
        if wall <= 0 or ticks < 0:
            raise ValueError('Invalid host timing samples')
        hz = os.sysconf('SC_CLK_TCK')
        return dict(wall_seconds=wall, cpu_seconds=ticks/hz, cpu_tick_hz=hz,
                    scope='Guest serial window; host time, not guest cycles')
