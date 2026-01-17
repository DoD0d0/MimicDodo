# MimicDodo

<p align="center">
  <img src="whole_body_tracking/dodo_backflip.gif" alt="Dodo Robot Backflip" width="100%"/>
  <br>
  <em>Figure 1: Dodo robot performing a backflip trained via RSL-RL</em>
</p>

A unified workspace for Dodo robot motion generation and training, combining trajectory optimization with reinforcement learning-based whole-body tracking.

This repository integrates two complementary approaches to robot motion generation:
- **Trajectory Creation**: Motion planning using Pinocchio and SE(3) trajectory optimization
- **Whole Body Tracking**: Deep learning-based motion imitation using Isaac Lab and RSL-RL

🧩 **Prerequisites**

- Ubuntu 22.04 (Linux x64) or Windows 11 (x64)
- RAM: 32GB
- GPU VRAM: 16GB (recommended for training)
- Python: 3.10 or 3.11 (depending on Isaac Sim version)
- NVIDIA GPU with compatible driver
- Conda (recommended)
- Isaac Sim 4.5.0 or later (for whole_body_tracking)

## 📁 Project Structure

The MimicDodo repository is organized into three main components:

```
MimicDodo/
├── assets/
│   └── dodo/                          # Centralized robot assets
│       ├── urdf/
│       │   └── dodo.urdf             # Robot URDF (shared by all projects)
│       ├── meshes/                   # Robot mesh files
│       └── config/                   # Robot configuration
│
├── trajectory_creation/              # Motion planning & trajectory generation
│   ├── pinocchio/                   # Pinocchio-based recording scripts
│   │   ├── record_walk.py
│   │   ├── record_jump.py
│   │   ├── record_kangaroo.py
│   │   └── record_stand.py
│   ├── se3_trajopt/                 # SE(3) trajectory optimization
│   │   └── src/
│   │       ├── examples/
│   │       │   └── agile_exps/
│   │       │       └── dodo_flip.py  # Backflip trajectory optimization
│   │       └── robots/
│   │           └── dodobot_v3/
│   │               └── DodoWrapper.py
│   └── blender/                     # Blender animation workflow (⚠️ Under Development)
│       └── Dodo_Walk.blend          # Blender file with skeleton rigging
│
└── whole_body_tracking/             # RL-based motion imitation
    ├── scripts/                     # Training & evaluation scripts
    │   ├── csv_to_npz_dodo.py      # Convert CSV to NPZ format
    │   ├── replay_npz_dodo.py      # Replay motion data
    │   └── rsl_rl/
    │       ├── train.py             # Training entry point
    │       └── play.py              # Policy evaluation/visualization
    └── source/
        └── whole_body_tracking/    # Isaac Lab extension source
```

### 🔄 Asset Centralization

All robot assets (URDF, meshes, configuration) are centralized in `assets/dodo/` and shared across both trajectory creation and training projects. This ensures consistency and simplifies asset management.

---

## 🚀 Installation

### Environment Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd MimicDodo
```

2. **Set up environment for trajectory creation (Pinocchio/SE3 TrajOpt):**
```bash
conda create -n trajopt python=3.13
conda activate trajopt
conda install -c conda-forge pinocchio meshcat-python cyipopt matplotlib numpy
```

3. **Set up environment for whole body tracking (Isaac Lab):**
```bash
cd whole_body_tracking
conda create -n beyondmimic python=3.10
conda activate beyondmimic
pip install --upgrade pip
pip install "isaacsim[all,extscache]==4.5.0.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
./isaaclab.sh --install rsl_rl
pip install wandb
python -m pip install -e source/whole_body_tracking
```

---

## 🎬 Trajectory Creation

The trajectory creation module provides multiple approaches for generating robot motion:

> ⚠️ **Note**: A Blender-based animation workflow is available in `trajectory_creation/blender/` but is **not production-ready** due to known issues with rotation axes, collision geometry, and CSV export. Use Pinocchio scripts or SE3 TrajOpt for reliable trajectory generation.

### 📐 Pinocchio Recording Scripts

Simple motion recording using forward/inverse kinematics. Ideal for generating basic locomotion patterns.

**Setup:**
```bash
conda activate trajopt
export PYTHONPATH="/home/simonkruelle/DataDome/Simons_Dokumente/Studium/Robotics, Cognition, Intelligence/DODO Alive/se3_trajopt/src"
```

**Available Scripts:**
- `record_walk.py` - Walking trajectory
- `record_jump.py` - Jumping motion
- `record_kangaroo.py` - Hopping motion
- `record_stand.py` - Static standing pose

**Run from project root:**
```bash
python trajectory_creation/pinocchio/record_walk.py
python trajectory_creation/pinocchio/record_jump.py
python trajectory_creation/pinocchio/record_kangaroo.py
python trajectory_creation/pinocchio/record_stand.py
```

**Output:** CSV files saved to `whole_body_tracking/scripts/`

### 🎯 SE(3) Trajectory Optimization

Advanced trajectory optimization in SE(3) tangent space. Generates dynamic motions like backflips using nonlinear optimization.

**Setup:**
```bash
conda activate trajopt
export PYTHONPATH="/home/simonkruelle/DataDome/Simons_Dokumente/Studium/Robotics, Cognition, Intelligence/DODO Alive/se3_trajopt/src"
```

**Run backflip optimization:**
```bash
python trajectory_creation/se3_trajopt/src/examples/agile_exps/dodo_flip.py --vis
```

**Output:** CSV file saved to `scripts/dodo_backflip.csv` (project root)

**Key Features:**
- Whole-body dynamics in SE(3) tangent space
- Contact constraints and terrain modeling
- Uses IPOPT for optimization
- Supports visualization with meshcat (`--vis` flag)

---

## 🧠 Whole Body Tracking (RL Training)

Reinforcement learning-based motion imitation using Isaac Lab and RSL-RL (PPO). Trains policies to reproduce reference motions with robustness and generalization.

### ⚙️ Environment Setup

All training commands must be executed from the `whole_body_tracking/` directory with the `beyondmimic` conda environment activated:

```bash
cd whole_body_tracking
conda activate beyondmimic
```

### 📊 Motion Preprocessing

Convert CSV trajectory files to NPZ format for training:

```bash
python scripts/csv_to_npz_dodo.py \
  --input_file scripts/dodo_walk.csv \
  --input_fps 100 \
  --output_name dodo_walk \
  --output_fps 100
```

**Parameters:**
- `--input_file`: Path to CSV trajectory file
- `--input_fps`: FPS of input CSV (100 Hz for Pinocchio scripts)
- `--output_name`: Name for output NPZ file
- `--output_fps`: Target FPS for training (typically 100 Hz)

**Output:** `dodo_walk.npz` saved in `whole_body_tracking/` directory

### 🎬 Motion Replay (Debugging)

Test motion data before training:

```bash
python scripts/replay_npz_dodo.py \
  --registry_name simon-kruelle-technical-university-of-munich/csv_to_npz/dodo_walk
```

This visualizes the reference motion in Isaac Sim to verify it loads correctly.

### 🏋️ Policy Training

Train a policy to imitate the reference motion:

```bash
WANDB_DIR=/tmp/wandb_cache python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-Dodo-v0 \
  --resume=True \
  --load_run=2026-01-17_12-19-27 \
  --checkpoint=model_12499.pt \
  --motion_file dodo_backflip.npz \
  --wandb_project whole_body_tracking \
  --headless
```

**Key Arguments:**
- `--task`: Environment variant (`Tracking-Flat-Dodo-v0` for flat terrain)
- `--motion_file`: NPZ file containing reference motion (e.g., `dodo_backflip.npz`)
- `--wandb_project`: Weights & Biases project name
- `--headless`: Run without GUI (recommended for training)
- `--resume`: Resume training from checkpoint
- `--load_run`: Run ID to resume from
- `--checkpoint`: Checkpoint file to load

**Training Output:**
- Logs: `whole_body_tracking/logs/rsl_rl/dodo_flat/<run_id>/`
- Checkpoints: Saved as `model_*.pt` files
- Metrics: Logged to WandB (if configured)

### 🎮 Policy Evaluation

Play a trained policy to visualize results:

```bash
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-Dodo-v0 \
  --num_envs=1 \
  --load_run "2026-01-17_12-19-27" \
  --checkpoint "model_12499.pt" \
  --motion_file dodo_backflip.npz \
  --video \
  --video_length 1000 \
  --disable_fabric
```

**Key Arguments:**
- `--task`: Environment variant (must match training task)
- `--num_envs`: Number of parallel environments (1 for visualization)
- `--load_run`: Run ID containing trained model
- `--checkpoint`: Checkpoint file to load
- `--motion_file`: Reference motion file
- `--video`: Enable video recording
- `--video_length`: Number of frames to record
- `--disable_fabric`: Disable Fabric rendering (for compatibility)

---

## 🤖 Robot Configuration (Dodo)

The Dodo robot is a bipedal robot with 8 actuated joints:

**Joints:**
- `left_joint_1` through `left_joint_4` (left leg)
- `right_joint_1` through `right_joint_4` (right leg)

**Asset Location:**
- **URDF**: `assets/dodo/urdf/dodo.urdf` (centralized, used by all projects)
- **Meshes**: `assets/dodo/meshes/`
- **Configuration**: `assets/dodo/config/`

All projects (Pinocchio, SE3 TrajOpt, and Isaac Lab) reference this centralized asset location automatically.

---

## 📋 Quick Reference: Command Cheat Sheet

### Trajectory Creation

```bash
# Activate environment
conda activate trajopt
export PYTHONPATH="/home/simonkruelle/DataDome/Simons_Dokumente/Studium/Robotics, Cognition, Intelligence/DODO Alive/se3_trajopt/src"

# Run Pinocchio scripts (from project root)
python trajectory_creation/pinocchio/record_walk.py

# Run SE3 trajectory optimization
python trajectory_creation/se3_trajopt/src/examples/agile_exps/dodo_flip.py --vis
```

### Whole Body Tracking

```bash
# Navigate to whole_body_tracking and activate environment
cd whole_body_tracking
conda activate beyondmimic

# Convert CSV to NPZ
python scripts/csv_to_npz_dodo.py --input_file scripts/dodo_walk.csv --input_fps 100 --output_name dodo_walk --output_fps 100

# Replay motion
python scripts/replay_npz_dodo.py --registry_name simon-kruelle-technical-university-of-munich/csv_to_npz/dodo_walk

# Train policy
WANDB_DIR=/tmp/wandb_cache python scripts/rsl_rl/train.py --task=Tracking-Flat-Dodo-v0 --resume=True --load_run=2026-01-17_12-19-27 --checkpoint=model_12499.pt --motion_file dodo_backflip.npz --wandb_project whole_body_tracking --headless

# Play trained policy
python scripts/rsl_rl/play.py --task=Tracking-Flat-Dodo-v0 --num_envs=1 --load_run "2026-01-17_12-19-27" --checkpoint "model_12499.pt" --motion_file dodo_backflip.npz --video --video_length 1000 --disable_fabric
```

---

## 🔄 Workflow: From Motion Planning to Policy Deployment

1. **Generate Reference Motion** (Trajectory Creation)
   - Option A: Use Pinocchio scripts for simple motions (walk, jump, stand)
   - Option B: Use SE3 TrajOpt for complex motions (backflip, handstand)

2. **Preprocess Motion Data** (Whole Body Tracking)
   - Convert CSV → NPZ format
   - Verify with replay script

3. **Train Policy** (Whole Body Tracking)
   - Train RL policy to imitate reference motion
   - Monitor training via WandB

4. **Evaluate & Deploy** (Whole Body Tracking)
   - Visualize trained policy with play script
   - Export policy for deployment

---

## 📚 Additional Documentation

- **Pinocchio Scripts**: See `trajectory_creation/pinocchio/README.md`
- **SE3 TrajOpt**: See `trajectory_creation/se3_trajopt/README.md`
- **Blender Animation** (⚠️ Under Development): See `trajectory_creation/blender/README.md`
- **Whole Body Tracking**: See `whole_body_tracking/README.md`

---

## 🛠️ Troubleshooting

### Path Issues

If you encounter path errors, ensure:
- URDF path is correctly calculated (all scripts use relative paths from script location)
- Mesh directory points to `assets/` (parent of `dodo/` package) for Pinocchio scripts
- Working directory is correct when running scripts

### Environment Issues

- **Trajectory Creation**: Ensure `PYTHONPATH` is set correctly
- **Whole Body Tracking**: Ensure you're in `whole_body_tracking/` directory with `beyondmimic` conda environment active

### Asset Loading

All projects now use the centralized `assets/dodo/urdf/dodo.urdf`. If you see errors about missing URDF files, verify:
1. The URDF exists at `assets/dodo/urdf/dodo.urdf`
2. Mesh files exist at `assets/dodo/meshes/`
3. The path calculation in scripts resolves correctly

---

## 📝 Notes

- Training is typically executed in headless mode on a workstation or server
- Motion data (CSV/NPZ files) should be placed in `whole_body_tracking/scripts/` or project root `scripts/`
- All robot assets are centralized in `assets/dodo/` for consistency
- The workflow supports both simple motions (Pinocchio) and complex dynamic motions (SE3 TrajOpt)

---

## 📄 License

See individual project directories for license information.

