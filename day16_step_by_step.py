import torch
from torch import nn


CHUNK_SIZE = 3
ACTION_DIM = 2


class ChunkPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, CHUNK_SIZE * ACTION_DIM),
        )

    def forward(self, obs):
        out = self.net(obs)
        return out.reshape(-1, CHUNK_SIZE, ACTION_DIM)


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def make_dataset(num_samples=5000):
    torch.manual_seed(501)
    target_xy = torch.empty(num_samples, 2).uniform_(-0.5, 0.5)
    hand_xy = torch.empty(num_samples, 2).uniform_(-0.5, 0.5)
    obs = torch.cat([target_xy, hand_xy], dim=1)

    # Expert chunk:
    # Plan three smooth moves that reduce the remaining distance.
    chunks = []
    current = hand_xy.clone()
    for _ in range(CHUNK_SIZE):
        move = 0.30 * (target_xy - current)
        chunks.append(move)
        current = current + move
    action_chunk = torch.stack(chunks, dim=1)
    return obs, action_chunk


def main():
    print_header("Step 0: Single action vs action chunk")
    print("Single-step policy output: [move_dx, move_dy]")
    print("Chunk policy output: [[dx0, dy0], [dx1, dy1], [dx2, dy2]]")

    print_header("Step 1: Create training data")
    obs, action_chunk = make_dataset()
    print("obs shape:", tuple(obs.shape))
    print("action_chunk shape:", tuple(action_chunk.shape))
    print("one obs [target_x, target_y, hand_x, hand_y]:", obs[0].tolist())
    print("one action chunk:", action_chunk[0].tolist())

    print_header("Step 2: Train chunk policy")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    obs = obs.to(device)
    action_chunk = action_chunk.to(device)
    model = ChunkPolicy().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(601):
        pred = model(obs)
        loss = loss_fn(pred, action_chunk)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch in [0, 10, 50, 100, 300, 600]:
            print(f"epoch={epoch:3d} loss={loss.item():.8f}")

    print_header("Step 3: Roll out one predicted action chunk")
    target_xy = torch.tensor([0.35, -0.20], dtype=torch.float32)
    hand_xy = torch.tensor([0.00, 0.00], dtype=torch.float32)
    test_obs = torch.cat([target_xy, hand_xy]).reshape(1, 4).to(device)
    with torch.no_grad():
        chunk = model(test_obs).cpu()[0]

    print(f"target_xy=({target_xy[0]:.3f}, {target_xy[1]:.3f})")
    print(f"start hand_xy=({hand_xy[0]:.3f}, {hand_xy[1]:.3f})")
    for i, move in enumerate(chunk):
        hand_xy = hand_xy + move
        distance = torch.linalg.norm(target_xy - hand_xy).item()
        print(
            f"chunk_step={i} "
            f"move=({move[0].item():+.4f}, {move[1].item():+.4f}) "
            f"hand=({hand_xy[0].item():+.4f}, {hand_xy[1].item():+.4f}) "
            f"distance={distance:.5f}"
        )

    print_header("Step 4: Meaning")
    print("This model predicts a short action sequence from one observation.")
    print("ACT uses the same high-level idea, but with transformer models and real robot observations.")


if __name__ == "__main__":
    main()

