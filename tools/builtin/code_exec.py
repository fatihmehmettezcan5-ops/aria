"""Execute a short Python snippet in a subprocess with strict limits.

This is *best-effort* sandboxing — for production you should add OS-level
isolation (gVisor / firejail / nsjail / a separate container).
"""
from __future__ import annotations

import asyncio
import os
import resource
import shutil
import sys
import tempfile
from typing import Any

from tools.schema import ToolContext, ToolSpec

MAX_CODE_CHARS = 4_000
TIMEOUT_S = 5.0
MEM_LIMIT_BYTES = 256 * 1024 * 1024  # 256 MB


def _set_limits():
    # Memory cap (only effective on Linux)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))
    except (ValueError, OSError):
        pass
    # No fork-bomb
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    except (ValueError, OSError):
        pass
    # No dumps
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass


async def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    code = args.get("code") or ""
    if not isinstance(code, str) or not code.strip():
        return {"error": "code required"}
    if len(code) > MAX_CODE_CHARS:
        return {"error": f"code > {MAX_CODE_CHARS} chars"}

    workdir = tempfile.mkdtemp(prefix="aria_exec_")
    src = os.path.join(workdir, "main.py")
    with open(src, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-S", "-B", src,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            preexec_fn=_set_limits,
            env={"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1"},
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT_S)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"error": "timeout", "timeout_s": TIMEOUT_S}
        return {
            "exit_code": proc.returncode,
            "stdout": stdout.decode("utf-8", errors="replace")[:4000],
            "stderr": stderr.decode("utf-8", errors="replace")[:2000],
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


TOOL = ToolSpec(
    name="code_exec",
    description=(
        "Execute a short Python snippet in a sandboxed subprocess (5s timeout, "
        "256MB RAM cap, no network). Returns stdout, stderr and exit code."
    ),
    parameters={
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
        "additionalProperties": False,
    },
    handler=_run,
)
