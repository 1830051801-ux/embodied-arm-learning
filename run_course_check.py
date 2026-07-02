import subprocess
import sys
from pathlib import Path


LESSONS = [
    "day1_step_by_step.py",
    "day2_step_by_step.py",
    "day3_step_by_step.py",
    "day4_step_by_step.py",
    "day5_step_by_step.py",
    "day6_step_by_step.py",
    "day7_step_by_step.py",
    "day8_step_by_step.py",
    "day9_step_by_step.py",
    "day10_step_by_step.py",
    "day11_step_by_step.py",
    "day12_step_by_step.py",
    "day13_step_by_step.py",
    "day14_step_by_step.py",
    "day15_step_by_step.py",
    "day16_step_by_step.py",
    "day17_step_by_step.py",
    "day18_step_by_step.py",
    "day19_step_by_step.py",
    "day20_step_by_step.py",
    "day21_step_by_step.py",
    "lesson_01_train_line.py",
    "lesson_02_train_robot_action.py",
    "lesson_03_use_robot_policy.py",
    "lesson_04_fake_yolo_to_policy.py",
    "lesson_05_policy_to_ik.py",
    "lesson_07_dry_run_robot_command.py",
    "lesson_08_evaluate_policy.py",
    "lesson_10_yolo_json_to_robot_command.py",
]


def run_lesson(base_dir, lesson):
    print("=" * 72)
    print(f"RUN {lesson}")
    print("=" * 72)
    result = subprocess.run(
        [sys.executable, str(base_dir / lesson)],
        cwd=str(base_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    print(result.stdout[-2500:])
    if result.returncode != 0:
        raise RuntimeError(f"{lesson} failed with exit code {result.returncode}")


def main():
    base_dir = Path(__file__).parent
    print("python:", sys.executable)
    print("course_dir:", base_dir)

    for lesson in LESSONS:
        run_lesson(base_dir, lesson)

    print("=" * 72)
    print("COURSE CHECK PASSED")
    print("=" * 72)
    print("All checked lessons can run on this machine.")


if __name__ == "__main__":
    main()
