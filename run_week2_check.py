import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "day8_step_by_step.py",
    "day9_step_by_step.py",
    "day10_step_by_step.py",
    "day11_step_by_step.py",
    "day12_step_by_step.py",
    "day13_step_by_step.py",
    "day14_step_by_step.py",
]


def run_script(base_dir, script):
    print("=" * 72)
    print(f"CHECK {script}")
    print("=" * 72)
    result = subprocess.run(
        [sys.executable, str(base_dir / script)],
        cwd=str(base_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    print(result.stdout[-1800:])
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed with exit code {result.returncode}")


def main():
    base_dir = Path(__file__).parent
    print("python:", sys.executable)
    print("course_dir:", base_dir)
    for script in SCRIPTS:
        run_script(base_dir, script)
    print("=" * 72)
    print("WEEK 2 CHECK PASSED")
    print("=" * 72)
    print("Day 8-14 project scripts all run successfully.")


if __name__ == "__main__":
    main()

