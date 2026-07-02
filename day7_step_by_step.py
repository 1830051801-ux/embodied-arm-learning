import math


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def clamp(value, low, high):
    return max(low, min(high, value))


def make_command(q1_deg, q2_deg, gripper_width_m):
    q1_safe = clamp(q1_deg, -90.0, 90.0)
    q2_safe = clamp(q2_deg, 0.0, 170.0)
    gripper_mm = gripper_width_m * 1000.0
    gripper_safe = clamp(gripper_mm, 20.0, 100.0)
    command = f"J {q1_safe:.2f} {q2_safe:.2f} G {gripper_safe:.1f}"
    return {
        "q1_raw": q1_deg,
        "q2_raw": q2_deg,
        "gripper_raw_mm": gripper_mm,
        "q1_safe": q1_safe,
        "q2_safe": q2_safe,
        "gripper_safe_mm": gripper_safe,
        "command": command,
    }


def main():
    print_header("Step 0: What is dry-run?")
    print("Dry-run means: build the command string, print it, but do not send it.")
    print("This is required before connecting a real motor controller.")

    print_header("Step 1: Pretend these came from PyTorch + IK")
    move_dx = 0.1234
    move_dy = -0.0441
    gripper_width = 0.0344
    q1_deg = -52.39
    q2_deg = 163.53
    print(f"policy output move_dx={move_dx:.4f}, move_dy={move_dy:.4f}, gripper_width={gripper_width:.4f} m")
    print(f"IK output q1={q1_deg:.2f} deg, q2={q2_deg:.2f} deg")
    print("Meaning: PyTorch decided action, IK converted target to joint angles.")

    print_header("Step 2: Apply safety limits")
    result = make_command(q1_deg, q2_deg, gripper_width)
    print(f"q1 raw={result['q1_raw']:.2f}, safe={result['q1_safe']:.2f}, limit=[-90, 90]")
    print(f"q2 raw={result['q2_raw']:.2f}, safe={result['q2_safe']:.2f}, limit=[0, 170]")
    print(
        f"gripper raw={result['gripper_raw_mm']:.1f} mm, "
        f"safe={result['gripper_safe_mm']:.1f} mm, limit=[20, 100]"
    )
    print("Meaning: never send unchecked model/IK output directly to motors.")

    print_header("Step 3: Build teaching command")
    print("teaching format: J q1 q2 G gripper_mm")
    print("dry_run_robot_command:", result["command"])
    print("Meaning: this is only a sample protocol. Replace it with your real controller format.")

    print_header("Step 4: Show what happens when values are unsafe")
    unsafe = make_command(q1_deg=-140.0, q2_deg=190.0, gripper_width_m=0.150)
    print("raw unsafe values: q1=-140 deg, q2=190 deg, gripper=150 mm")
    print(
        "clamped values: "
        f"q1={unsafe['q1_safe']:.2f}, "
        f"q2={unsafe['q2_safe']:.2f}, "
        f"gripper={unsafe['gripper_safe_mm']:.1f} mm"
    )
    print("dry_run_robot_command:", unsafe["command"])

    print_header("Step 5: Real hardware checklist")
    checklist = [
        "COM port, for example COM3",
        "baudrate, for example 115200",
        "real command protocol",
        "joint limits for every motor",
        "gripper command format",
        "emergency stop or power-off method",
        "low-speed small-angle test first",
    ]
    for i, item in enumerate(checklist, start=1):
        print(f"{i}. {item}")

    print_header("Step 6: Meaning for your project")
    print("Today we stop at printing commands.")
    print("Only after confirming the hardware protocol should this become serial write.")


if __name__ == "__main__":
    main()

