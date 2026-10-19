"""Copy only project source/config into a new immutable experiment snapshot."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', type=Path)
    parser.add_argument('--data-path', type=Path,
                        default=Path('data/colmap4d/flame_steak'),
                        help='Repository-relative dataset adapter to link into the snapshot')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    target = args.target.resolve()
    if target.exists():
        raise FileExistsError(target)
    if args.data_path.is_absolute() or '..' in args.data_path.parts:
        raise ValueError('data-path must be repository-relative without parent traversal')
    dataset = (repo / args.data_path).resolve(strict=True)
    files = subprocess.check_output(['git', 'ls-files', '--cached', '--others',
        '--exclude-standard'], cwd=repo, text=True).splitlines()
    selected = [name for name in files if name in {'main.py', 'AGENTS.md'} or
                name.startswith(('pipeline/', 'config/', 'scripts/', 'tests/',
                                 'extensions/'))]
    hashes = {}
    for name in selected:
        source, destination = repo / name, target / name
        if not source.is_file() or source.is_symlink():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    link = target / args.data_path
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(dataset, target_is_directory=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    manifest = dict(source=str(repo), commit=commit, hashes=hashes,
                    dataset_target=str(dataset), dataset_link=str(args.data_path),
                    copied_data=False, copied_outputs=False)
    (target / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(snapshot=str(target), files=len(hashes), commit=commit)))


if __name__ == '__main__':
    main()
