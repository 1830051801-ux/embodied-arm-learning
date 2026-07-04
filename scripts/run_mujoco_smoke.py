from __future__ import annotations

from embodied_arm.mujoco_sim import check_mujoco_runtime
from embodied_arm.sim_env import ArmManipulationSim


if __name__ == "__main__":
    status = check_mujoco_runtime()
    env = ArmManipulationSim()
    obs = env.reset()
    step = env.step([0.1, 0.02, 0.06])
    print({"runtime": status, "initial": obs, "step": step})
