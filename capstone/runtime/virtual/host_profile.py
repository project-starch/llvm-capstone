"""Optional host perf sampling, deliberately separate from clean timing runs."""
import json
from pathlib import Path
import shutil
import signal
import subprocess


class HostProfile:
    def __init__(self, perf, work, pid):
        self.perf, self.work, self.pid = perf, work, pid
        self.process = self.log = None

    def start(self):
        cmd = [str(self.perf), 'record', '--clockid', 'mono', '-e', 'cycles:u',
               '-F', '199', '-m', '128', '--call-graph', 'dwarf,8192', '-p', str(self.pid),
               '-o', str(self.work/'host-perf.data')]
        (self.work/'host-perf-command.json').write_text(json.dumps(cmd, indent=2)+'\n')
        self.log = (self.work/'host-perf.log').open('wb')
        try:
            self.process = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        except BaseException:
            self.log.close()
            raise

    def stop(self):
        if self.process is None:
            return
        process, self.process = self.process, None
        try:
            interrupted = process.poll() is None
            if interrupted:
                process.send_signal(signal.SIGINT)
            try:
                rc = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise RuntimeError('Host perf did not finish after SIGINT')
            # perf flushes its data and then re-raises our SIGINT on some versions.
            if rc != 0 and not (interrupted and rc == -signal.SIGINT):
                raise RuntimeError(f'Host perf failed: {rc}; see host-perf.log')
        finally:
            self.log.close()

    def save_map(self):
        source = Path(f'/tmp/perf-{self.pid}.map')
        if source.exists():
            shutil.copy2(source, self.work/source.name)
