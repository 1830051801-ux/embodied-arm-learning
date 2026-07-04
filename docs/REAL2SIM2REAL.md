# Real2Sim2Real Notes

This project is structured so real robot data can gradually replace synthetic samples.

## Real to Sim

1. Log real detection output and robot state.
2. Convert detector pixels to robot-frame coordinates.
3. Store each interaction in the grasp CSV schema.
4. Replay the row through `ArmManipulationSim` and compare predicted motion to logged motion.

## Sim

The MuJoCo scene is compact but keeps the main elements needed for tabletop grasp validation:

- 6-DoF robot arm with base, shoulder, elbow, wrist and gripper controls
- table collision body
- target object and distractor object
- fixed camera and position actuators

`embodied_arm.domain_randomization` samples object position, camera scale, detection noise and friction. `scripts/generate_synthetic_data.py` uses those samples to produce extra grasp rows.

## Sim to Real

Before any real hardware command:

1. Evaluate the policy on held-out logs.
2. Run closed-loop dry-run output.
3. Check IK joint limits.
4. Compare the generated command against the STM32/Raspberry Pi coordinate protocol used by the real arm project.

The current default output is a dry-run joint/gripper command, not a direct motor command.
