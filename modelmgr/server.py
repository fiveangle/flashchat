"""Server process awareness for management operations.

Destructive/manage operations must refuse to touch a model the running
server is serving. Mirrors the bash guards: pid file liveness plus a
port-listening fallback.
"""
from __future__ import annotations

import os
import socket

from . import configfile, paths


def pid_file_path() -> str:
    return os.path.join(paths.config_dir(), "server.pid")


def is_server_running() -> bool:
    pid_file = pid_file_path()
    if not os.path.isfile(pid_file):
        return False
    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        return False


def is_port_listening(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=0.3):
            return True
    except OSError:
        return False


def serving_model_id() -> str | None:
    """The resolved model id the running server is serving, if any."""
    values = configfile.load()
    if not values:
        return None
    if is_server_running():
        return values.get("MODEL")
    port = values.get("SERVER_PORT", "8000")
    host = values.get("SERVER_HOST", "127.0.0.1")
    try:
        if is_port_listening(int(port), "127.0.0.1" if host in ("localhost", "0.0.0.0") else host):
            return values.get("MODEL")
    except ValueError:
        pass
    return None


def guard_model_not_serving(resolved_ids) -> str | None:
    """Returns an error message if the server is serving one of the ids."""
    serving = serving_model_id()
    if serving and serving in set(resolved_ids):
        return (f"the server is currently running model '{serving}' — "
                f"stop it first with './flashchat serve --stop'")
    return None
