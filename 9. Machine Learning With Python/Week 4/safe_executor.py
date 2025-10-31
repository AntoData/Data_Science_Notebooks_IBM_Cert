#!/usr/bin/env python3
"""
Interactive safe executor for Python scripts.

- Asks which .py file to run (list from a folder or paste a path).
- Runs it in a child process.
- Monitors RAM and runtime; terminates the child if limits are exceeded.
- Optionally caps BLAS/OpenMP threads to keep your machine responsive.

Requires: psutil  (pip install psutil)
"""

import os
import sys
import time
import shlex
import psutil
import subprocess
from pathlib import Path
from typing import List, Optional

# ---------- helpers ----------

def list_py(dir_path: Path) -> List[Path]:
    return sorted(p for p in dir_path.glob("*.py") if p.is_file())

def pick_script() -> Path:
    # Ask for a directory; default to current dir
    raw = input(f"Directory to list .py files [default: {Path.cwd()}]: ").strip()
    dir_path = Path(raw).expanduser().resolve() if raw else Path.cwd()

    scripts = list_py(dir_path)
    if scripts:
        print("\nSelect a script to run:")
        for i, p in enumerate(scripts, 1):
            print(f"  [{i}] {p.name}")
        print("  [P] Paste/enter a full path")
        while True:
            choice = input(f"Enter 1..{len(scripts)} or P: ").strip()
            if choice.lower() == "p":
                path = Path(input("Full path to .py: ").strip()).expanduser().resolve()
                if path.exists() and path.suffix == ".py":
                    return path
                print("Path invalid or not a .py file; try again.")
                continue
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(scripts):
                    return scripts[idx]
            print("Invalid selection; try again.")
    else:
        print(f"No .py files found in {dir_path}.")
        path = Path(input("Paste a full path to a .py file: ").strip()).expanduser().resolve()
        if not (path.exists() and path.suffix == ".py"):
            print("Given path is invalid or not a .py file.")
            sys.exit(2)
        return path

def tree_rss_gb(proc: psutil.Process) -> float:
    """Resident memory (GB) of proc + its children."""
    rss = 0
    try:
        rss += proc.memory_info().rss
        for c in proc.children(recursive=True):
            try:
                rss += c.memory_info().rss
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    return rss / (1024 ** 3)

def run_with_guard(
    cmd: List[str],
    mem_limit_gb: float = 8.0,
    time_limit_s: Optional[int] = None,
    poll_s: float = 0.5,
    threads: Optional[int] = None,
    env_extra: Optional[dict] = None,
) -> int:
    """Run command in a guarded subprocess. Return exit code."""
    env = os.environ.copy()
    if threads:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[k] = str(threads)
    if env_extra:
        env.update(env_extra)

    print(f"\n[EXEC] {' '.join(cmd)}")
    print(f"[GUARD] mem ≤ {mem_limit_gb} GB"
          + (f", time ≤ {time_limit_s}s" if time_limit_s else "")
          + f", poll={poll_s}s, threads={threads or 'default'}")

    p = subprocess.Popen(cmd, env=env)  # inherit stdout/stderr
    proc = psutil.Process(p.pid)
    t0 = time.time()

    try:
        while True:
            rc = p.poll()
            if rc is not None:
                return rc

            rss_gb = tree_rss_gb(proc)
            sys_mem = psutil.virtual_memory().percent
            if rss_gb > mem_limit_gb or sys_mem > 92:
                print(f"[GUARD] Limit exceeded (child {rss_gb:.2f} GB, system {sys_mem:.0f}% used). Terminating…")
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("[GUARD] Forcing kill…")
                    p.kill()
                return p.returncode if p.returncode is not None else 1

            if time_limit_s and (time.time() - t0 > time_limit_s):
                print("[GUARD] Time limit exceeded. Terminating…")
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("[GUARD] Forcing kill…")
                    p.kill()
                return p.returncode if p.returncode is not None else 1

            time.sleep(poll_s)
    finally:
        # make sure any grandchildren are gone
        try:
            for c in proc.children(recursive=True):
                try:
                    c.terminate()
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            pass

# ---------- main ----------

def main():
    script = pick_script()

    args_str = input("Arguments to pass to the script (optional, e.g. --k 6): ").strip()
    try:
        script_args = shlex.split(args_str) if args_str else []
    except ValueError as e:
        print(f"Could not parse arguments: {e}")
        sys.exit(2)

    def _float_in(prompt: str, default: Optional[float]):
        s = input(f"{prompt} [{'' if default is None else default}]: ").strip()
        if not s:
            return default
        try:
            return float(s)
        except ValueError:
            print("Not a number; using default.")
            return default

    def _int_in(prompt: str, default: Optional[int]):
        s = input(f"{prompt} [{'' if default is None else default}]: ").strip()
        if not s:
            return default
        try:
            return int(s)
        except ValueError:
            print("Not an integer; using default.")
            return default

    mem_limit_gb = _float_in("Memory cap in GB", 8.0)
    time_limit_s = _int_in("Time limit in seconds (empty = no limit)", None)
    threads = _int_in("Max threads for BLAS/OpenMP (empty = default)", None)

    cmd = [sys.executable, str(script)] + script_args
    rc = run_with_guard(cmd, mem_limit_gb=mem_limit_gb, time_limit_s=time_limit_s, threads=threads)
    print(f"\n[EXIT] Return code: {rc}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)
