import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "day1_step_by_step.py",
    "day2_step_by_step.py",
    "day3_step_by_step.py",
    "day4_step_by_step.py",
    "day5_step_by_step.py",
    "day6_step_by_step.py",
    "day7_step_by_step.py",
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
    print("WEEK 1 CHECK PASSED")
    print("=" * 72)
    print("Day 1-7 teaching scripts all run successfully.")


if __name__ == "__main__":
    main()

