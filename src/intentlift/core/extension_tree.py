"""Lock, identify and atomically install complete compiler extension trees."""
from contextlib import contextmanager
import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

from .errors import LiftError, require


@contextmanager
def extension_lock(root, exclusive=False):
    # Lock the stable parent inode. Swapping the extension directory must not
    # change the inode that competing readers and writers use as their lock.
    descriptor = os.open(Path(root).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        os.close(descriptor)


def fingerprint(root):
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and '.git' not in path.parts:
            name = os.fsencode(path.relative_to(root))
            content = path.read_bytes()
            digest.update(len(name).to_bytes(8, 'big'))
            digest.update(name)
            digest.update(len(content).to_bytes(8, 'big'))
            digest.update(content)
    return digest.hexdigest()


def snapshot_extensions(root, destination):
    with extension_lock(root):
        shutil.copytree(root, destination, ignore=shutil.ignore_patterns('__pycache__'))
        return fingerprint(destination)


def exchange_directories(first, second):
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    require(rename is not None, 'atomic directory exchange requires renameat2', 'infrastructure_failure')
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    at_fdcwd, rename_exchange = -100, 2
    if rename(at_fdcwd, os.fsencode(first), at_fdcwd, os.fsencode(second), rename_exchange):
        raise LiftError('infrastructure_failure', 'atomic extension exchange failed: ' +
                        os.strerror(ctypes.get_errno()))


def publish_extensions(root, candidate, baseline, validated):
    root, candidate = Path(root), Path(candidate)
    with extension_lock(root, exclusive=True):
        require(fingerprint(root) == baseline, 'official compiler changed during candidate iteration',
                'boundary_required')
        require(fingerprint(candidate) == validated, 'candidate changed after compilation',
                'boundary_required')
        if validated == baseline:
            return validated
        prefix = f'.{root.name}-promotion-'
        # An interrupted writer may leave only its temporary container. Its
        # official directory is always either wholly old or wholly new.
        marker = 'intentlift-promotion'
        for abandoned in root.parent.glob(prefix + '*'):
            if not abandoned.is_symlink() and (abandoned / marker).is_file():
                shutil.rmtree(abandoned)
        with tempfile.TemporaryDirectory(prefix=prefix, dir=root.parent) as staging:
            (Path(staging) / marker).touch()
            prepared = Path(staging) / 'tree'
            shutil.copytree(candidate, prepared, ignore=shutil.ignore_patterns('__pycache__'))
            require(fingerprint(prepared) == validated, 'staged publication is not the validated compiler',
                    'boundary_required')
            exchange_directories(prepared, root)
        return validated
