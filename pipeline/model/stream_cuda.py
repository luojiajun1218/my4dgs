"""Source-isolated build of the forward-only streaming update kernels."""
import functools
import hashlib
from pathlib import Path


@functools.lru_cache(maxsize=1)
def backend():
    from torch.utils.cpp_extension import load
    root=(Path(__file__).resolve().parents[2]/'extensions/stream_update').resolve()
    source=[root/'update.cpp',root/'update.cu']
    fingerprint=hashlib.sha256(str(root).encode())
    for path in source:
        fingerprint.update(path.read_bytes())
    return load(name='my4dgs_stream_'+fingerprint.hexdigest()[:16],
        sources=[str(path) for path in source],extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3','--fmad=false'],verbose=False)
