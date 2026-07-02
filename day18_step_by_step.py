import torch
from torch import nn


CHUNK_SIZE = 3
ACTION_DIM = 2
OBS_DIM = 4
NOISE_STD = 0.08


class DenoisePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = OBS_DIM + CHUNK_SIZE * ACTION_DIM
        output_dim = CHUNK_SIZE * ACTION_DIM
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, obs, noisy_chunk):
        flat_noisy = noisy_chunk.reshape(noisy_chunk.shape[0], -1)
        x = torch.cat([obs, flat_noisy], dim=1)
        clean_flat = self.net(x)
        return clean_flat.reshape(-1, CHUNK_SIZE, ACTION_DIM)


def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def make_clean_dataset(num_samples=6000, seed=701):
    torch.manual_seed(seed)
    target_xy = torch.empty(num_samples, 2).uniform_(-0.5, 0.5)
    hand_xy = torch.empty(num_samples, 2).uniform_(-0.5, 0.5)
    obs = torch.cat([target_xy, hand_xy], dim=1)

    chunks = []
    current = hand_xy.clone()
    for _ in range(CHUNK_SIZE):
        move = 0.30 * (target_xy - current)
        chunks.append(move)
        current = current + move
    clean_chunk = torch.stack(chunks, dim=1)
    return obs, clean_chunk


def add_noise(clean_chunk):
    return clean_chunk + torch.randn_like(clean_chunk) * NOISE_STD


def chunk_mse(a, b):
    return torch.mean((a - b) ** 2).item()


def main():
    print_header("Step 0: Diffusion Policy intuition")
    print("Day 16: directly predict an action chunk.")
    print("Day 18: learn to denoise a noisy action chunk into a clean action chunk.")

    print_header("Step 1: Create clean expert action chunks")
    obs, clean_chunk = make_clean_dataset()
    noisy_chunk = add_noise(clean_chunk)
    print("obs shape:", tuple(obs.shape))
    print("clean_chunk shape:", tuple(clean_chunk.shape))
    print("noisy_chunk shape:", tuple(noisy_chunk.shape))
    print("one clean chunk:", clean_chunk[0].tolist())
    print("one noisy chunk:", noisy_chunk[0].tolist())
    print(f"initial noisy_vs_clean_mse={chunk_mse(noisy_chunk, clean_chunk):.8f}")

    print_header("Step 2: Train denoise model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    obs = obs.to(device)
    clean_chunk = clean_chunk.to(device)
    noisy_chunk = noisy_chunk.to(device)

    model = DenoisePolicy().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(701):
        pred_clean = model(obs, noisy_chunk)
        loss = loss_fn(pred_clean, clean_chunk)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch in [0, 10, 50, 100, 300, 700]:
            print(f"epoch={epoch:3d} denoise_loss={loss.item():.8f}")

    print_header("Step 3: Test denoising on new action chunks")
    test_obs, test_clean = make_clean_dataset(num_samples=1000, seed=702)
    test_noisy = add_noise(test_clean)
    test_obs = test_obs.to(device)
    test_clean = test_clean.to(device)
    test_noisy = test_noisy.to(device)

    with torch.no_grad():
        test_pred = model(test_obs, test_noisy)

    noisy_mse = chunk_mse(test_noisy.cpu(), test_clean.cpu())
    denoised_mse = chunk_mse(test_pred.cpu(), test_clean.cpu())
    print(f"noisy_vs_clean_mse={noisy_mse:.8f}")
    print(f"denoised_vs_clean_mse={denoised_mse:.8f}")

    print_header("Step 4: Inspect one denoised chunk")
    idx = 0
    print("clean chunk:")
    print(test_clean[idx].cpu().tolist())
    print("noisy chunk:")
    print(test_noisy[idx].cpu().tolist())
    print("denoised prediction:")
    print(test_pred[idx].cpu().tolist())

    print_header("Step 5: Meaning")
    print("This is not full Diffusion Policy yet.")
    print("It is the core idea: condition on state and turn noisy action sequences into cleaner action sequences.")
    print("Real DP repeats denoising over multiple noise levels and often conditions on image observations.")


if __name__ == "__main__":
    main()

