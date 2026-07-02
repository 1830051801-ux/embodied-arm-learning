from dataclasses import dataclass


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


@dataclass
class SerialConfig:
    port: str = "COM3"
    baudrate: int = 115200
    newline: str = "\n"
    dry_run: bool = True


class RobotSerialDriver:
    def __init__(self, config):
        self.config = config

    def send(self, command):
        full_command = command + self.config.newline
        if self.config.dry_run:
            print(f"DRY_RUN_SEND port={self.config.port} baudrate={self.config.baudrate}: {full_command!r}")
            return

        # Real serial code intentionally omitted from this teaching lesson.
        # Add pyserial only after confirming the real controller protocol.
        raise RuntimeError("Real serial sending is disabled in this lesson.")

    def move_joints(self, q1_deg, q2_deg, gripper_mm):
        command = f"J {q1_deg:.2f} {q2_deg:.2f} G {gripper_mm:.1f}"
        self.send(command)

    def home(self):
        self.send("HOME")

    def open_gripper(self):
        self.send("G OPEN")

    def close_gripper(self):
        self.send("G CLOSE")


def main():
    print_header("Step 0: What are we doing?")
    print("We are building the serial command layer.")
    print("Today is dry-run only: no COM port is opened, no command is sent.")

    print_header("Step 1: Define serial config")
    config = SerialConfig(port="COM3", baudrate=115200, dry_run=True)
    print(config)
    print("Meaning: dry_run=True means print only.")

    print_header("Step 2: Create driver")
    driver = RobotSerialDriver(config)
    print("driver created")

    print_header("Step 3: Send basic dry-run commands")
    driver.home()
    driver.open_gripper()
    driver.close_gripper()

    print_header("Step 4: Send PyTorch+IK dry-run move")
    q1_deg = -52.39
    q2_deg = 163.53
    gripper_mm = 34.4
    driver.move_joints(q1_deg, q2_deg, gripper_mm)

    print_header("Step 5: What must be confirmed before real sending?")
    checklist = [
        "Actual COM port shown by Windows Device Manager",
        "Baudrate required by the control board",
        "Exact command syntax",
        "Whether command needs newline, carriage return, or checksum",
        "Safe joint angle limits",
        "Emergency stop method",
    ]
    for i, item in enumerate(checklist, start=1):
        print(f"{i}. {item}")

    print_header("Step 6: Meaning")
    print("This script is the boundary between AI decision and hardware command.")
    print("Keep it dry-run until the real controller protocol is confirmed.")


if __name__ == "__main__":
    main()

