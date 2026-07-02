import math


LINK_1 = 0.35
LINK_2 = 0.25


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def forward_kinematics(q1_rad, q2_rad):
    x = LINK_1 * math.cos(q1_rad) + LINK_2 * math.cos(q1_rad + q2_rad)
    y = LINK_1 * math.sin(q1_rad) + LINK_2 * math.sin(q1_rad + q2_rad)
    return x, y


def inverse_kinematics(x, y):
    distance_sq = x * x + y * y
    cos_q2 = (distance_sq - LINK_1 * LINK_1 - LINK_2 * LINK_2) / (2 * LINK_1 * LINK_2)

    if cos_q2 < -1.0 or cos_q2 > 1.0:
        raise ValueError("Target is outside reachable workspace.")

    q2 = math.acos(cos_q2)
    q1 = math.atan2(y, x) - math.atan2(LINK_2 * math.sin(q2), LINK_1 + LINK_2 * math.cos(q2))
    return q1, q2


def main():
    print_header("Step 0: Why do we need IK?")
    print("PyTorch may output move_dx/move_dy.")
    print("Motors need joint angles.")
    print("IK converts target x/y into joint angles q1/q2.")

    print_header("Step 1: Define a simple 2-link arm")
    print(f"link_1 length: {LINK_1} m")
    print(f"link_2 length: {LINK_2} m")
    print(f"max reach: {LINK_1 + LINK_2} m")

    print_header("Step 2: FK example - joint angles to hand position")
    q1_deg = 20.0
    q2_deg = 80.0
    q1_rad = math.radians(q1_deg)
    q2_rad = math.radians(q2_deg)
    hand_xy = forward_kinematics(q1_rad, q2_rad)
    print(f"input joint angles: q1={q1_deg:.2f} deg, q2={q2_deg:.2f} deg")
    print(f"FK output hand_xy=({hand_xy[0]:.4f}, {hand_xy[1]:.4f})")
    print("Meaning: if motors are at these angles, the hand is at this x/y.")

    print_header("Step 3: IK example - target position to joint angles")
    target_xy = (0.30, 0.10)
    q1, q2 = inverse_kinematics(target_xy[0], target_xy[1])
    print(f"target_xy=({target_xy[0]:.4f}, {target_xy[1]:.4f})")
    print(f"IK output q1={math.degrees(q1):.2f} deg, q2={math.degrees(q2):.2f} deg")
    print("Meaning: these are the joint angles needed to reach the target.")

    print_header("Step 4: Check IK with FK")
    check_xy = forward_kinematics(q1, q2)
    error = math.dist(target_xy, check_xy)
    print(f"FK check_xy=({check_xy[0]:.4f}, {check_xy[1]:.4f})")
    print(f"target_xy=  ({target_xy[0]:.4f}, {target_xy[1]:.4f})")
    print(f"error={error:.8f} m")
    print("Meaning: FK check proves the IK result reaches the target.")

    print_header("Step 5: Add PyTorch policy output conceptually")
    current_hand_xy = (0.00, 0.00)
    move_dx = 0.12
    move_dy = -0.04
    next_hand_xy = (current_hand_xy[0] + move_dx, current_hand_xy[1] + move_dy)
    q1_next, q2_next = inverse_kinematics(next_hand_xy[0], next_hand_xy[1])
    print(f"current_hand_xy=({current_hand_xy[0]:.4f}, {current_hand_xy[1]:.4f})")
    print(f"policy output move_dx={move_dx:.4f}, move_dy={move_dy:.4f}")
    print(f"next_hand_xy=({next_hand_xy[0]:.4f}, {next_hand_xy[1]:.4f})")
    print(f"IK for next hand target: q1={math.degrees(q1_next):.2f} deg, q2={math.degrees(q2_next):.2f} deg")

    print_header("Step 6: Meaning for your real robot")
    print("YOLO and calibration find the object.")
    print("PyTorch decides the next action.")
    print("IK converts the target action into joint angles.")
    print("The control board receives joint-angle commands.")


if __name__ == "__main__":
    main()

