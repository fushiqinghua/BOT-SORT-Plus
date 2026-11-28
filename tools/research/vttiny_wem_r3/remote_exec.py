"""Run the W R3 archival workflow on an explicitly configured project host."""

import os
from pathlib import PurePosixPath
import shlex
import subprocess


def remote(script: str) -> bytes:
    target = os.environ.get("VT_WEM_SSH_TARGET", "")
    project_value = os.environ.get("VT_WEM_REVIEW_PROJECT_ROOT", "")
    executable = os.environ.get("VT_WEM_REMOTE_PYTHON", "python3")
    project = PurePosixPath(project_value)
    if not target or target.startswith("-") or any(c.isspace() for c in target):
        raise SystemExit("Set VT_WEM_SSH_TARGET to the SSH host or alias used for this project.")
    if not project.is_absolute() or ".." in project.parts:
        raise SystemExit("Set VT_WEM_REVIEW_PROJECT_ROOT to the absolute remote project directory.")
    root = project / "integrations" / "b_gmc_evidence_overlap_v1"
    prefix = f"from pathlib import Path\nimport os, sys, json, subprocess\nr = Path({str(root)!r})\n"
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", target,
               f"{shlex.quote(executable)} -Bu -"]
    result = subprocess.run(command, input=(prefix + script).encode("utf-8"), check=True, stdout=subprocess.PIPE)
    return result.stdout
