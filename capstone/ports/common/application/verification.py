"""Shared port regression transport; application bytes stay in separate files."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent


class VM:
    def __init__(self, state, parent):
        self.state = Path(state)
        self.share = Path(json.loads((self.state / 'config.json').read_text())['share'])
        Path(parent).mkdir(parents=True, exist_ok=True)
        self.output = Path(tempfile.mkdtemp(prefix='delegated-', dir=parent))
        self.data = Path(tempfile.mkdtemp(prefix='port-input-', dir=self.share))

    def stage(self, source):
        source = Path(source)
        target = self.data / source.name
        shutil.copyfile(source, target)
        return '/mnt/host/' + str(target.relative_to(self.share))

    def run(self, name, image, arguments=(), environment=(), user=None):
        result = self.output / (name + '.json')
        command = [sys.executable, str(HERE / 'run.py'), '--state', str(self.state),
                   '--result', str(result)]
        if user:
            command += ['--user', user]
        for item in environment:
            command += ['-e', item]
        command += [str(image), *arguments]
        with (self.output / (name + '.stdout')).open('wb') as out, (self.output / (name + '.stderr')).open('wb') as err:
            subprocess.run(command, stdout=out, stderr=err, check=False)
        record = json.loads(result.read_text())
        return record

    def close(self):
        shutil.rmtree(self.data)
