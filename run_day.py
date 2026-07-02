import argparse
import subprocess
import sys
from pathlib import Path


DAYS = {
    0: ("Why PyTorch for your robot arm", "day0_notes.md", "day0_step_by_step.py"),
    1: ("PyTorch minimal training loop", "day1_notes.md", "day1_step_by_step.py"),
    2: ("Robot state to action", "day2_notes.md", "day2_step_by_step.py"),
    3: ("YOLO output to PyTorch input", "day3_notes.md", "day3_step_by_step.py"),
    4: ("IK target position to joint angles", "day4_notes.md", "day4_step_by_step.py"),
    5: ("Grasp CSV data to model", "day5_notes.md", "day5_step_by_step.py"),
    6: ("Evaluate whether policy learned", "day6_notes.md", "day6_step_by_step.py"),
    7: ("Dry-run robot command", "day7_notes.md", "day7_step_by_step.py"),
    8: ("Camera calibration pixel to robot", "day8_notes.md", "day8_step_by_step.py"),
    9: ("YOLO adapter to standard JSON", "day9_notes.md", "day9_step_by_step.py"),
    10: ("Serial dry-run template", "day10_notes.md", "day10_step_by_step.py"),
    11: ("Grasp episode logging", "day11_notes.md", "day11_step_by_step.py"),
    12: ("Train policy from grasp log", "day12_notes.md", "day12_step_by_step.py"),
    13: ("Rule baseline vs learned policy", "day13_notes.md", "day13_step_by_step.py"),
    14: ("Interview project summary", "day14_notes.md", "day14_step_by_step.py"),
    15: ("Closed-loop policy rollout", "day15_notes.md", "day15_step_by_step.py"),
    16: ("Action chunking policy", "day16_notes.md", "day16_step_by_step.py"),
    17: ("Temporal context from observation history", "day17_notes.md", "day17_step_by_step.py"),
    18: ("Diffusion Policy denoising intuition", "day18_notes.md", "day18_step_by_step.py"),
    19: ("Language-conditioned policy VLA intuition", "day19_notes.md", "day19_step_by_step.py"),
    20: ("Vision input policy", "day20_notes.md", "day20_step_by_step.py"),
    21: ("Mini VLA multimodal fusion", "day21_notes.md", "day21_step_by_step.py"),
}


def list_days():
    for day, (title, notes, script) in DAYS.items():
        print(f"Day {day:2d}: {title}")
        print(f"        notes:  {notes}")
        print(f"        script: {script}")


def run_day(day):
    if day not in DAYS:
        raise SystemExit(f"Unknown day: {day}. Use --list to see valid days.")

    base_dir = Path(__file__).parent
    title, notes, script = DAYS[day]
    notes_path = base_dir / notes
    script_path = base_dir / script

    print("=" * 72)
    print(f"Day {day}: {title}")
    print("=" * 72)
    print("Read notes first:")
    print(notes_path)
    print()
    print("Running:")
    print(script_path)
    print()
    sys.stdout.flush()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(base_dir),
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List all available days.")
    parser.add_argument("--day", type=int, help="Run a specific day, from 0 to 21.")
    args = parser.parse_args()

    if args.list:
        list_days()
        return
    if args.day is not None:
        run_day(args.day)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
