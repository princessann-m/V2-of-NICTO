"""Sandboxed coding studio that actually runs Python tests."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap


class CodingStudio:
    def run_tests(self, project_files: dict) -> dict:
        if not project_files:
            return {"tests_run": 0, "tests_passed": 0, "success": False, "error": "no files"}
        with tempfile.TemporaryDirectory() as tmpdir:
            for filename, content in project_files.items():
                path = os.path.join(tmpdir, filename)
                with open(path, "w") as f:
                    f.write(content)
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "--tb=short", tmpdir],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                passed = result.stdout.count(" passed")
                failed = result.stdout.count(" failed")
                return {
                    "tests_run": passed + failed,
                    "tests_passed": passed,
                    "success": result.returncode == 0,
                    "stdout": result.stdout[-2000:],
                    "stderr": result.stderr[-2000:],
                }
            except subprocess.TimeoutExpired:
                return {"tests_run": 0, "tests_passed": 0, "success": False, "error": "timeout"}
            except Exception as e:
                return {"tests_run": 0, "tests_passed": 0, "success": False, "error": str(e)}
