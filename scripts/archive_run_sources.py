"""Freeze the small source files used by a run, never datasets/checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def archive_sources(run):
    destination = run / 'source_snapshot.zip'
    if destination.exists():
        return
    manifest = json.loads((run / 'controlled_manifest.json').read_text())
    source = Path(manifest['source'])
    files = {}
    for name, expected in manifest['source_sha256'].items():
        path = source / name
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Source changed before snapshot: {path}')
        files['source/' + name] = content
    for name in ('run_controlled.py', 'archive_run_sources.py'):
        wrapper = Path(__file__).with_name(name)
        files['runner/' + name] = wrapper.read_bytes()
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    (run / 'source_snapshot_manifest.json').write_text(json.dumps(
        {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    archive_sources(parser.parse_args().run)
