"""Atomic directory ownership for jobs sharing pfs3 across different machines."""
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import socket
import time
import uuid


@contextmanager
def directory_lock(path, *, wait=False):
    """Use filesystem namespace exclusion, independently of advisory flock.

    Never expire a remote owner's lock by elapsed time: evaluation and training
    can legitimately take hours. Owner metadata permits checking a stale job.
    """
    path = Path(path)
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            try:
                owner = json.loads((path/'owner.json').read_text())
            except (OSError, ValueError):
                owner = {'status':'owner is being recorded; lock remains held'}
            message = f'Shared job lock held: {path}; owner={owner}'
            if not wait:
                raise BlockingIOError(errno.EAGAIN, message) from None
            print(message, flush=True)
            time.sleep(30)
    owner = dict(hostname=socket.gethostname(), pid=os.getpid(), token=uuid.uuid4().hex,
                 created_at=datetime.now(timezone.utc).isoformat())
    owner_file = path/'owner.json'
    try:
        owner_file.write_text(json.dumps(owner, indent=2)+'\n')
        yield owner
    finally:
        # Only release this invocation's directory, never a replacement owner.
        try:
            current = json.loads(owner_file.read_text())
        except (OSError, ValueError):
            current = None
        if current == owner:
            owner_file.unlink()
            path.rmdir()
