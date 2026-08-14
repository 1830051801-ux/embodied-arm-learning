param(
    [string]$PythonPath = "D:\EmbodiedAI\mujoco-venv\Scripts\python.exe",
    [int]$TrainCount = 512,
    [int]$TestCount = 128,
    [int]$Epochs = 350
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python runtime not found: $PythonPath. Use a Python environment with PyTorch, NumPy, and Matplotlib."
}

Set-Location -LiteralPath $root
$runtime = Join-Path $root "runtime\constraint_diffusion"
$checkpoint = Join-Path $runtime "embodied_action_chunk_transformer_policy.pt"
$train = Join-Path $runtime "train.npz"
$test = Join-Path $runtime "test.npz"
$shift = Join-Path $runtime "test_shift.npz"
$nominalReport = Join-Path $runtime "evaluation_nominal.json"
$shiftReport = Join-Path $runtime "evaluation_shift.json"

& $PythonPath tools\constraint_diffusion_twin.py generate --output $train --count $TrainCount --seed 20260815
& $PythonPath tools\constraint_diffusion_twin.py generate --output $test --count $TestCount --seed 20260816
& $PythonPath tools\constraint_diffusion_twin.py generate --output $shift --count $TestCount --seed 20260817 --xy-sigma-m 0.008 --z-sigma-m 0.004 --yaw-sigma-deg 5.0
& $PythonPath tools\constraint_diffusion_twin.py train --dataset $train --checkpoint $checkpoint --epochs $Epochs --batch-size 64 --hidden-dim 96 --diffusion-steps 16 --seed 20260815
& $PythonPath tools\constraint_diffusion_twin.py evaluate --checkpoint $checkpoint --dataset $test --output $nominalReport --samples-per-context 3 --abstain-dispersion-rad 0.45 --seed 20260815
& $PythonPath tools\constraint_diffusion_twin.py evaluate --checkpoint $checkpoint --dataset $shift --output $shiftReport --samples-per-context 3 --abstain-dispersion-rad 0.45 --seed 20260815
& $PythonPath tools\render_constraint_diffusion_dashboard.py --checkpoint $checkpoint --nominal $nominalReport --shift $shiftReport --dataset $test --output assets\constraint_diffusion_dashboard.png
