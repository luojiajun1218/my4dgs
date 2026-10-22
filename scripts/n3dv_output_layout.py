"""Canonical per-scene outputs with aliases for already running launchers."""

import os
import ctypes
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def scene_output(batch, scene, timestamp, recipe='half_final480w'):
    for value in (scene, timestamp, recipe):
        if not value or Path(value).name != value or value in ('.', '..'):
            raise ValueError('Scene output labels must be plain path components')
    alias = batch / scene
    parent = batch.parent if batch.parent.name == 'n3dv' else ROOT / 'output/n3dv'
    parent.mkdir(parents=True, exist_ok=True)
    if alias.is_symlink():
        target = alias.resolve(strict=True)
        if target.parent != parent or not target.name.startswith(scene + '_'):
            raise ValueError('Unexpected existing scene output alias: ' + str(alias))
        return target
    target = parent / f'{scene}_{timestamp}_{recipe}'
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    moved = alias.exists()
    if moved:
        # Exchange the directory and its future alias in one Linux rename.
        # Writers retain a valid path throughout, so training need not pause.
        target.symlink_to(os.path.relpath(target, alias.parent), target_is_directory=True)
        try:
            rename = ctypes.CDLL(None, use_errno=True).renameat2
            rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                               ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            if rename(-100, os.fsencode(alias), -100, os.fsencode(target), 2):
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
        except BaseException:
            target.unlink()
            raise
        return target
    else:
        target.mkdir()
    try:
        alias.symlink_to(os.path.relpath(target, alias.parent), target_is_directory=True)
    except BaseException:
        target.rmdir()
        raise
    return target
