"""Constraint-diffusion digital twin for the XiaoU six-axis arm.

The pipeline creates counterfactual visual observations from the checked-in
POE arm model, learns a conditional denoising trajectory prior, then projects
sampled trajectories through IK, joint limits, and the diagnostic scene. It is
strictly offline: no ROS hardware, CAN interface, serial port, or physical arm
command is opened by this tool.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROBOT_AI = ROOT / "robot_ai"
if str(ROBOT_AI) not in sys.path:
    sys.path.insert(0, str(ROBOT_AI))

from arm_control import JointLimits, fk_space, ik_space_multistart, load_default_model
from arm_control.scene_review import DiagnosticScene, load_diagnostic_scene, review_tcp_positions


TRAJECTORY_STEPS = 32
JOINT_COUNT = 6
GRASP_INDEX = 10
PLACE_INDEX = 23
ASSUMED_JOINT_LIMIT_RAD = 1.20
CONTEXT_DIM = 21
ARCHITECTURE_NAME = "embodied_action_chunk_transformer_diffusion"
CONTEXT_TOKEN_NAMES = (
    "visual_grasp_pose",
    "visual_place_goal",
    "observation_uncertainty",
    "learned_task_token",
)


def require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover - depends on the local CUDA runtime.
        raise RuntimeError(
            "PyTorch is required. Run this tool with D:\\EmbodiedAI\\mujoco-venv\\Scripts\\python.exe."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def rotation_z(angle_rad: float) -> np.ndarray:
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def pose_to_context(grasp_pose: np.ndarray, place_pose: np.ndarray, confidence: float, xy_sigma_m: float, yaw_sigma_rad: float) -> np.ndarray:
    grasp_rotation6d = grasp_pose[:3, :2].reshape(-1)
    place_rotation6d = place_pose[:3, :2].reshape(-1)
    context = np.concatenate(
        [
            grasp_pose[:3, 3] / 0.5,
            grasp_rotation6d,
            place_pose[:3, 3] / 0.5,
            place_rotation6d,
            np.array([confidence, xy_sigma_m / 0.01, yaw_sigma_rad / math.radians(10.0)], dtype=np.float64),
        ]
    )
    if context.shape != (CONTEXT_DIM,) or not np.isfinite(context).all():
        raise ValueError("invalid diffusion-policy context")
    return context.astype(np.float32)


def quintic_blend(tau: np.ndarray) -> np.ndarray:
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def build_expert_trajectory(home: np.ndarray, grasp: np.ndarray, place: np.ndarray) -> np.ndarray:
    """Build a smooth home -> grasp -> place -> home trajectory."""
    if home.shape != (JOINT_COUNT,) or grasp.shape != (JOINT_COUNT,) or place.shape != (JOINT_COUNT,):
        raise ValueError("every trajectory anchor must contain six joint angles")
    anchors = [(0, home), (GRASP_INDEX, grasp), (PLACE_INDEX, place), (TRAJECTORY_STEPS - 1, home)]
    trajectory = np.zeros((TRAJECTORY_STEPS, JOINT_COUNT), dtype=np.float64)
    for (start_index, start), (end_index, end) in zip(anchors, anchors[1:]):
        tau = np.linspace(0.0, 1.0, end_index - start_index + 1, dtype=np.float64)
        blend = quintic_blend(tau)[:, None]
        trajectory[start_index : end_index + 1] = start + blend * (end - start)
    return trajectory


def trajectory_tcp_positions(model, trajectory: np.ndarray) -> np.ndarray:
    return np.stack(
        [fk_space(model.home_grasp_tcp, model.screw_axes, angles)[:3, 3] for angles in trajectory],
        axis=0,
    )


def pose_is_in_diagnostic_workspace(pose: np.ndarray) -> bool:
    x_m, y_m, z_m = pose[:3, 3]
    return 0.14 <= x_m <= 0.38 and -0.25 <= y_m <= 0.18 and 0.08 <= z_m <= 0.48


def perturb_pose(pose: np.ndarray, rng: np.random.Generator, xy_sigma_m: float, z_sigma_m: float, yaw_sigma_rad: float) -> np.ndarray:
    noisy = pose.copy()
    noisy[:3, 3] += rng.normal([0.0, 0.0, 0.0], [xy_sigma_m, xy_sigma_m, z_sigma_m])
    noisy[:3, :3] = rotation_z(float(rng.normal(0.0, yaw_sigma_rad))) @ noisy[:3, :3]
    return noisy


def rigidify_pose(pose: np.ndarray) -> np.ndarray:
    """Restore a valid SE(3) pose after float32 dataset serialization."""
    result = np.asarray(pose, dtype=np.float64).copy()
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError("pose must be a finite 4x4 matrix")
    left, _, right_t = np.linalg.svd(result[:3, :3])
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    result[:3, :3] = rotation
    result[3] = [0.0, 0.0, 0.0, 1.0]
    return result


@dataclass(frozen=True)
class DatasetSettings:
    count: int
    seed: int
    xy_sigma_m: float
    z_sigma_m: float
    yaw_sigma_deg: float

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("dataset count must be positive")
        values = (self.xy_sigma_m, self.z_sigma_m, self.yaw_sigma_deg)
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("noise settings must be finite and non-negative")


def generate_dataset(output_path: Path, settings: DatasetSettings, scene: DiagnosticScene) -> dict[str, Any]:
    rng = np.random.default_rng(settings.seed)
    model = load_default_model()
    home = np.zeros(JOINT_COUNT, dtype=np.float64)
    trajectories: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    clean_grasp_poses: list[np.ndarray] = []
    clean_place_poses: list[np.ndarray] = []
    observed_grasp_poses: list[np.ndarray] = []
    observed_place_poses: list[np.ndarray] = []
    attempts = 0
    max_attempts = settings.count * 200
    yaw_sigma_rad = math.radians(settings.yaw_sigma_deg)

    while len(trajectories) < settings.count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"only generated {len(trajectories)}/{settings.count} safe episodes after {attempts} attempts")
        # Keep the synthetic expert data inside a conservative portion of the
        # unmeasured joint envelope. Larger excursions are evaluated later as
        # distribution shift rather than silently treated as calibrated motion.
        grasp = rng.uniform(-0.42, 0.42, size=JOINT_COUNT)
        place = rng.uniform(-0.42, 0.42, size=JOINT_COUNT)
        trajectory = build_expert_trajectory(home, grasp, place)
        if np.max(np.abs(trajectory)) > ASSUMED_JOINT_LIMIT_RAD:
            continue
        clean_grasp = fk_space(model.home_grasp_tcp, model.screw_axes, grasp)
        clean_place = fk_space(model.home_grasp_tcp, model.screw_axes, place)
        if not pose_is_in_diagnostic_workspace(clean_grasp) or not pose_is_in_diagnostic_workspace(clean_place):
            continue
        review = review_tcp_positions(trajectory_tcp_positions(model, trajectory), scene)
        if not review.safe:
            continue
        observed_grasp = perturb_pose(clean_grasp, rng, settings.xy_sigma_m, settings.z_sigma_m, yaw_sigma_rad)
        observed_place = perturb_pose(clean_place, rng, settings.xy_sigma_m, settings.z_sigma_m, yaw_sigma_rad)
        confidence = float(rng.uniform(0.80, 0.99))
        trajectories.append((trajectory / ASSUMED_JOINT_LIMIT_RAD).astype(np.float32))
        contexts.append(pose_to_context(observed_grasp, observed_place, confidence, settings.xy_sigma_m, yaw_sigma_rad))
        clean_grasp_poses.append(clean_grasp.astype(np.float32))
        clean_place_poses.append(clean_place.astype(np.float32))
        observed_grasp_poses.append(observed_grasp.astype(np.float32))
        observed_place_poses.append(observed_place.astype(np.float32))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        trajectories=np.stack(trajectories),
        contexts=np.stack(contexts),
        clean_grasp_poses=np.stack(clean_grasp_poses),
        clean_place_poses=np.stack(clean_place_poses),
        observed_grasp_poses=np.stack(observed_grasp_poses),
        observed_place_poses=np.stack(observed_place_poses),
    )
    metadata = {
        "episodes": len(trajectories),
        "attempts": attempts,
        "acceptance_rate": len(trajectories) / attempts,
        "trajectory_steps": TRAJECTORY_STEPS,
        "joint_count": JOINT_COUNT,
        "assumed_joint_limit_rad": ASSUMED_JOINT_LIMIT_RAD,
        "noise": {
            "xy_sigma_m": settings.xy_sigma_m,
            "z_sigma_m": settings.z_sigma_m,
            "yaw_sigma_deg": settings.yaw_sigma_deg,
        },
        "scene": "offline_diagnostic_scene.json",
        "real_motion_enabled": False,
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_denoiser(hidden_dim: int, diffusion_steps: int):
    torch, nn, _, _ = require_torch()

    if hidden_dim < 16 or hidden_dim % 4 != 0:
        raise ValueError("hidden_dim must be at least 16 and divisible by four")

    class EmbodiedActionChunkTransformer(nn.Module):
        """ACT-style chunk decoder conditioned on visual-goal-uncertainty tokens.

        The model represents a complete 32-step joint action chunk rather than
        choosing a single grasp point.  A context encoder fuses observed grasp
        pose, place goal, and perception uncertainty with a learned task token.
        A Transformer decoder then cross-attends from six-axis action tokens to
        that context while estimating diffusion noise or a trajectory proposal.
        """

        def __init__(self) -> None:
            super().__init__()
            self.visual_grasp_projection = nn.Sequential(
                nn.Linear(9, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.visual_goal_projection = nn.Sequential(
                nn.Linear(9, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.uncertainty_projection = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.task_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.context_type_embedding = nn.Parameter(torch.zeros(1, len(CONTEXT_TOKEN_NAMES), hidden_dim))

            context_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context_encoder = nn.TransformerEncoder(context_layer, num_layers=2, enable_nested_tensor=False)

            self.state_projection = nn.Linear(JOINT_COUNT, hidden_dim)
            self.time_projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.action_position_embedding = nn.Parameter(torch.zeros(1, TRAJECTORY_STEPS, hidden_dim))
            self.action_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.prior_query = nn.Parameter(torch.zeros(1, TRAJECTORY_STEPS, hidden_dim))
            action_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.action_decoder = nn.TransformerDecoder(action_layer, num_layers=3)
            self.output_projection = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, JOINT_COUNT),
            )
            self._reset_parameters()

        def _reset_parameters(self) -> None:
            nn.init.normal_(self.task_token, mean=0.0, std=0.02)
            nn.init.normal_(self.context_type_embedding, mean=0.0, std=0.02)
            nn.init.normal_(self.action_position_embedding, mean=0.0, std=0.02)
            nn.init.normal_(self.action_type_embedding, mean=0.0, std=0.02)
            nn.init.normal_(self.prior_query, mean=0.0, std=0.02)

        def encode_context(self, context):
            if context.ndim != 2 or context.shape[1] != CONTEXT_DIM:
                raise ValueError(f"context must have shape (batch, {CONTEXT_DIM})")
            grasp_token = self.visual_grasp_projection(context[:, 0:9])
            goal_token = self.visual_goal_projection(context[:, 9:18])
            uncertainty_token = self.uncertainty_projection(context[:, 18:21])
            task_token = self.task_token.expand(context.shape[0], -1, -1).squeeze(1)
            tokens = torch.stack((grasp_token, goal_token, uncertainty_token, task_token), dim=1)
            return self.context_encoder(tokens + self.context_type_embedding)

        def decode_action_chunk(self, action_tokens, context_tokens):
            return self.output_projection(self.action_decoder(action_tokens, context_tokens))

        def forward(self, noisy_trajectory, time_index, context):
            if noisy_trajectory.ndim != 3 or noisy_trajectory.shape[1:] != (TRAJECTORY_STEPS, JOINT_COUNT):
                raise ValueError(f"noisy_trajectory must have shape (batch, {TRAJECTORY_STEPS}, {JOINT_COUNT})")
            half = hidden_dim // 2
            frequencies = torch.exp(
                -math.log(10000.0) * torch.arange(half, device=noisy_trajectory.device, dtype=torch.float32) / max(half - 1, 1)
            )
            phase = time_index.float().unsqueeze(1) * frequencies.unsqueeze(0)
            time_embedding = torch.cat([phase.sin(), phase.cos()], dim=1)
            if time_embedding.shape[1] < hidden_dim:
                time_embedding = torch.nn.functional.pad(time_embedding, (0, hidden_dim - time_embedding.shape[1]))
            action_tokens = self.state_projection(noisy_trajectory)
            action_tokens = action_tokens + self.time_projection(time_embedding).unsqueeze(1)
            action_tokens = action_tokens + self.action_position_embedding + self.action_type_embedding
            return self.decode_action_chunk(action_tokens, self.encode_context(context))

        def predict_prior(self, context):
            action_queries = self.prior_query.expand(context.shape[0], -1, -1)
            action_queries = action_queries + self.action_position_embedding + self.action_type_embedding
            return self.decode_action_chunk(action_queries, self.encode_context(context))

    return EmbodiedActionChunkTransformer()


def diffusion_schedule(torch, steps: int, device):
    betas = torch.linspace(1e-4, 0.02, steps, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return {"betas": betas, "alphas": alphas, "alpha_bars": alpha_bars}


def train_model(dataset_path: Path, checkpoint_path: Path, epochs: int, batch_size: int, hidden_dim: int, diffusion_steps: int, seed: int) -> dict[str, Any]:
    torch, _, DataLoader, TensorDataset = require_torch()
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(dataset_path)
    trajectories = torch.tensor(data["trajectories"], dtype=torch.float32)
    contexts = torch.tensor(data["contexts"], dtype=torch.float32)
    context_np = data["contexts"].astype(np.float64)
    context_mean = context_np.mean(axis=0)
    # Some context fields describe declared sensor noise and are intentionally
    # constant inside one training split. A scale floor prevents a harmless
    # metadata change from creating a numerical divide-by-near-zero OOD score.
    context_std = np.maximum(context_np.std(axis=0), 0.05)
    support_distances = np.linalg.norm((context_np - context_mean) / context_std, axis=1)
    support_threshold = float(np.quantile(support_distances, 0.995))
    loader = DataLoader(TensorDataset(trajectories, contexts), batch_size=batch_size, shuffle=True, drop_last=False)
    model = build_denoiser(hidden_dim, diffusion_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    schedule = diffusion_schedule(torch, diffusion_steps, device)
    losses: list[float] = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        for x0, context in loader:
            x0 = x0.to(device)
            context = context.to(device)
            time_index = torch.randint(0, diffusion_steps, (x0.shape[0],), device=device)
            alpha_bar = schedule["alpha_bars"][time_index].view(-1, 1, 1)
            noise = torch.randn_like(x0)
            noisy = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise
            predicted_noise = model(noisy, time_index, context)
            prior = model.predict_prior(context)
            noise_loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            prior_loss = torch.nn.functional.mse_loss(prior, x0)
            loss = noise_loss + 0.5 * prior_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
        if epoch == 1 or epoch % max(1, epochs // 8) == 0 or epoch == epochs:
            print(f"epoch={epoch:03d} loss={losses[-1]:.6f}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_dim": hidden_dim,
            "diffusion_steps": diffusion_steps,
            "trajectory_steps": TRAJECTORY_STEPS,
            "context_dim": CONTEXT_DIM,
            "architecture": ARCHITECTURE_NAME,
            "context_tokens": list(CONTEXT_TOKEN_NAMES),
            "action_chunk_length": TRAJECTORY_STEPS,
            "joint_limit_rad": ASSUMED_JOINT_LIMIT_RAD,
            "context_mean": context_mean.astype(np.float32),
            "context_std": context_std.astype(np.float32),
            "context_support_threshold": support_threshold,
            "losses": losses,
            "device": str(device),
        },
        checkpoint_path,
    )
    return {
        "device": str(device),
        "epochs": epochs,
        "final_loss": losses[-1],
        "initial_loss": losses[0],
        "dataset_episodes": int(trajectories.shape[0]),
        "context_support_threshold": support_threshold,
        "checkpoint": str(checkpoint_path),
    }


def sample_trajectories(model, context, diffusion_steps: int, sample_count: int, device):
    torch, _, _, _ = require_torch()
    model.eval()
    samples = []
    with torch.no_grad():
        for _ in range(sample_count):
            repeated_context = context.unsqueeze(0).to(device)
            trajectory = model.predict_prior(repeated_context) + 0.35 * torch.randn(
                (1, TRAJECTORY_STEPS, JOINT_COUNT), dtype=torch.float32, device=device
            )
            for step in range(diffusion_steps - 1, -1, -1):
                time_index = torch.full((1,), step, dtype=torch.long, device=device)
                beta = 1e-4 + (0.02 - 1e-4) * step / max(diffusion_steps - 1, 1)
                alpha = 1.0 - beta
                alpha_bar = float(np.prod([1.0 - (1e-4 + (0.02 - 1e-4) * index / max(diffusion_steps - 1, 1)) for index in range(step + 1)]))
                predicted_noise = model(trajectory, time_index, repeated_context)
                predicted_x0 = (trajectory - math.sqrt(1.0 - alpha_bar) * predicted_noise) / math.sqrt(alpha_bar)
                if step == 0:
                    trajectory = predicted_x0
                else:
                    previous_alpha_bar = float(np.prod([1.0 - (1e-4 + (0.02 - 1e-4) * index / max(diffusion_steps - 1, 1)) for index in range(step)]))
                    trajectory = math.sqrt(previous_alpha_bar) * predicted_x0 + math.sqrt(1.0 - previous_alpha_bar) * predicted_noise
            samples.append(trajectory.squeeze(0).cpu().numpy())
    return np.stack(samples, axis=0)


def project_with_constraints(prediction: np.ndarray, observed_grasp_pose: np.ndarray, observed_place_pose: np.ndarray, scene: DiagnosticScene):
    model = load_default_model()
    limits = JointLimits(
        position_min=np.full(JOINT_COUNT, -ASSUMED_JOINT_LIMIT_RAD),
        position_max=np.full(JOINT_COUNT, ASSUMED_JOINT_LIMIT_RAD),
        velocity_max=np.full(JOINT_COUNT, 0.4),
        acceleration_max=np.full(JOINT_COUNT, 0.8),
    )
    anchors: list[np.ndarray] = []
    for target_pose, seed in ((rigidify_pose(observed_grasp_pose), prediction[GRASP_INDEX]), (rigidify_pose(observed_place_pose), prediction[PLACE_INDEX])):
        bounded_seed = np.clip(seed, limits.position_min, limits.position_max)
        offsets = (
            np.zeros(JOINT_COUNT, dtype=np.float64),
            np.array([0.0, -0.35, 0.35, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.35, -0.35, 0.0, 0.0, 0.0]),
            np.array([0.0, -0.70, 0.70, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.70, -0.70, 0.0, 0.0, 0.0]),
        )
        candidate_seeds = [bounded_seed]
        candidate_seeds.extend(
            np.clip(bounded_seed + offset, limits.position_min, limits.position_max)
            for offset in offsets[1:]
        )
        candidate_seeds.append(np.zeros(JOINT_COUNT, dtype=np.float64))
        result = ik_space_multistart(
            model.home_grasp_tcp,
            model.screw_axes,
            target_pose,
            candidate_seeds,
            preferred_angles=bounded_seed,
            joint_lower=limits.position_min,
            joint_upper=limits.position_max,
            orientation_tolerance_rad=2e-4,
            position_tolerance_m=2e-4,
            max_iterations=250,
            max_step_rad=0.15,
        )
        if not result.converged:
            return None, f"IK projection failed at {'grasp' if len(anchors) == 0 else 'place'}"
        anchors.append(result.joint_angles)
    trajectory = build_expert_trajectory(np.zeros(JOINT_COUNT), anchors[0], anchors[1])
    review = review_tcp_positions(trajectory_tcp_positions(model, trajectory), scene)
    if not review.safe:
        return None, "projected trajectory violates diagnostic scene clearance"
    return trajectory, None


def pose_position_error(model, joint_angles: np.ndarray, target_pose: np.ndarray) -> float:
    actual = fk_space(model.home_grasp_tcp, model.screw_axes, joint_angles)
    return float(np.linalg.norm(actual[:3, 3] - target_pose[:3, 3]))


def evaluate_model(checkpoint_path: Path, dataset_path: Path, output_path: Path, samples_per_context: int, abstain_dispersion_rad: float, seed: int) -> dict[str, Any]:
    torch, _, _, _ = require_torch()
    if samples_per_context < 1 or not math.isfinite(abstain_dispersion_rad) or abstain_dispersion_rad <= 0.0:
        raise ValueError("invalid evaluation settings")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != ARCHITECTURE_NAME:
        found = checkpoint.get("architecture", "legacy_checkpoint_without_architecture")
        raise ValueError(
            f"checkpoint architecture is {found!r}; retrain with {ARCHITECTURE_NAME!r} before evaluation"
        )
    model = build_denoiser(int(checkpoint["hidden_dim"]), int(checkpoint["diffusion_steps"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    data = np.load(dataset_path)
    contexts = data["contexts"]
    clean_grasp = data["clean_grasp_poses"]
    clean_place = data["clean_place_poses"]
    observed_grasp = data["observed_grasp_poses"]
    observed_place = data["observed_place_poses"]
    scene = load_diagnostic_scene()
    arm = load_default_model()
    context_mean = np.asarray(checkpoint["context_mean"], dtype=np.float64)
    context_std = np.asarray(checkpoint["context_std"], dtype=np.float64)
    support_threshold = float(checkpoint["context_support_threshold"])

    raw_errors: list[float] = []
    shielded_errors: list[float] = []
    raw_scene_safe = 0
    shielded_scene_safe = 0
    abstentions = 0
    ood_abstentions = 0
    dispersion_abstentions = 0
    projection_failures: list[str] = []
    examples: list[dict[str, object]] = []
    for index, context_np in enumerate(contexts):
        context = torch.tensor(context_np, dtype=torch.float32)
        samples = sample_trajectories(model, context, int(checkpoint["diffusion_steps"]), samples_per_context, device)
        dispersion = float(np.mean(np.linalg.norm(samples[:, GRASP_INDEX] - samples[:, GRASP_INDEX].mean(axis=0), axis=1)))
        prediction = np.mean(samples, axis=0) * ASSUMED_JOINT_LIMIT_RAD
        prediction = np.clip(prediction, -ASSUMED_JOINT_LIMIT_RAD, ASSUMED_JOINT_LIMIT_RAD)
        raw_errors.append(pose_position_error(arm, prediction[GRASP_INDEX], clean_grasp[index]))
        raw_review = review_tcp_positions(trajectory_tcp_positions(arm, prediction), scene)
        raw_scene_safe += int(raw_review.safe)
        support_distance = float(np.linalg.norm((context_np.astype(np.float64) - context_mean) / context_std))
        if support_distance > support_threshold:
            abstentions += 1
            ood_abstentions += 1
            if len(examples) < 8:
                examples.append(
                    {
                        "index": index,
                        "support_distance": support_distance,
                        "dispersion_rad": dispersion,
                        "decision": "abstain",
                        "reason": "context outside calibrated training support",
                    }
                )
            continue
        if dispersion > abstain_dispersion_rad:
            abstentions += 1
            dispersion_abstentions += 1
            if len(examples) < 8:
                examples.append({"index": index, "dispersion_rad": dispersion, "decision": "abstain", "reason": "sample dispersion above threshold"})
            continue
        projected, failure = project_with_constraints(prediction, observed_grasp[index], observed_place[index], scene)
        if projected is None:
            projection_failures.append(failure or "unknown")
            if len(examples) < 8:
                examples.append({"index": index, "dispersion_rad": dispersion, "decision": "projection_failed", "reason": failure})
            continue
        shielded_errors.append(pose_position_error(arm, projected[GRASP_INDEX], clean_grasp[index]))
        shielded_scene_safe += 1
        if len(examples) < 8:
            examples.append(
                {
                    "index": index,
                    "dispersion_rad": dispersion,
                    "decision": "projected_safe",
                    "raw_grasp_error_m": raw_errors[-1],
                    "shielded_grasp_error_m": shielded_errors[-1],
                }
            )
    report = {
        "evaluation": "constraint_diffusion_digital_twin",
        "architecture": ARCHITECTURE_NAME,
        "context_tokens": list(CONTEXT_TOKEN_NAMES),
        "action_chunk_length": TRAJECTORY_STEPS,
        "real_motion_enabled": False,
        "episodes": int(len(contexts)),
        "seed": seed,
        "samples_per_context": samples_per_context,
        "abstain_dispersion_rad": abstain_dispersion_rad,
        "context_support_threshold": support_threshold,
        "metrics": {
            "raw_mean_grasp_position_error_m": float(np.mean(raw_errors)),
            "raw_scene_safe_rate": raw_scene_safe / len(contexts),
            "shielded_mean_grasp_position_error_m": float(np.mean(shielded_errors)) if shielded_errors else None,
            "shielded_scene_safe_rate": shielded_scene_safe / len(contexts),
            "projection_success_rate": shielded_scene_safe / max(1, len(contexts) - abstentions),
            "abstention_rate": abstentions / len(contexts),
            "ood_abstention_rate": ood_abstentions / len(contexts),
            "dispersion_abstention_rate": dispersion_abstentions / len(contexts),
        },
        "projection_failure_counts": {key: projection_failures.count(key) for key in sorted(set(projection_failures))},
        "examples": examples,
        "limitations": [
            "training data is generated from the checked-in POE kinematic model and a synthetic perception noise model",
            "constraint projection is a planning safety shield, not a measured real-arm control law",
            "the diagnostic scene checks TCP clearance only and does not replace full-link collision checking",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="generate counterfactual expert trajectories")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--count", type=int, default=512)
    generate.add_argument("--seed", type=int, default=20260815)
    generate.add_argument("--xy-sigma-m", type=float, default=0.003)
    generate.add_argument("--z-sigma-m", type=float, default=0.002)
    generate.add_argument("--yaw-sigma-deg", type=float, default=2.0)

    train = subparsers.add_parser("train", help="train the conditional diffusion trajectory prior")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--hidden-dim", type=int, default=96)
    train.add_argument("--diffusion-steps", type=int, default=32)
    train.add_argument("--seed", type=int, default=20260815)

    evaluate = subparsers.add_parser("evaluate", help="evaluate raw samples against constraint-projected trajectories")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--samples-per-context", type=int, default=3)
    evaluate.add_argument("--abstain-dispersion-rad", type=float, default=0.45)
    evaluate.add_argument("--seed", type=int, default=20260815)

    args = parser.parse_args()
    if args.command == "generate":
        result = generate_dataset(
            args.output,
            DatasetSettings(args.count, args.seed, args.xy_sigma_m, args.z_sigma_m, args.yaw_sigma_deg),
            load_diagnostic_scene(),
        )
    elif args.command == "train":
        result = train_model(args.dataset, args.checkpoint, args.epochs, args.batch_size, args.hidden_dim, args.diffusion_steps, args.seed)
    else:
        result = evaluate_model(args.checkpoint, args.dataset, args.output, args.samples_per_context, args.abstain_dispersion_rad, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
