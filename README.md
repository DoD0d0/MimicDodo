# MimicDodo

<p align="center">
  <img src="whole_body_tracking/dodo_backflip.gif" alt="Dodo Robot Backflip" width="100%"/>
  <br>
  <em>Figure 1: Dodo robot performing a backflip trained via RSL-RL</em>
</p>

A unified workspace for Dodo robot motion generation and training, combining trajectory optimization with reinforcement learning-based whole-body tracking.

🧩 **Prerequisites**

- Ubuntu 22.04 (Linux x64)
- RAM: 32GB
- GPU VRAM: 16GB (recommended for training)
- NVIDIA GPU with compatible driver
- Conda (recommended)
- Isaac Sim 4.5.0 or later (for whole_body_tracking)

---

## 📁 Project Structure
```
MimicDodo/
├── assets/
│   ├── dodo/                          # Original Dodo robot
│   │   ├── urdf/dodo.urdf
│   │   ├── meshes/
│   │   └── config/
│   └── dodo_daimao/                   # DodoDaimao robot
│       ├── urdf/dodo_daimao.urdf      # Fixed: hip_left mass, primitive collisions
│       └── meshes/
│
├── trajectory_creation/
│   ├── pinocchio/                     # Simple FK/IK recording scripts
│   └── se3_trajopt/src/
│       ├── examples/agile_exps/
│       │   ├── dodo_backflip.py
│       │   ├── dodo_daimao_backflip.py
│       │   └── dodo_daimao_walk.py
│       └── robots/
│           ├── dodobot_v3/DodoWrapper.py
│           └── dodo_daimao/DodoDaimaoWrapper.py
│
└── whole_body_tracking/
    ├── dodo_daimao_backflip.npz       # Generated — do not commit
    ├── scripts/
    │   ├── csv_to_npz_dodo.py         # Dodo converter (requires Isaac Sim)
    │   ├── csv_to_npz_dodo_daimao.py  # DodoDaimao converter (Pinocchio, fast)
    │   ├── replay_npz_dodo.py
    │   └── rsl_rl/
    │       ├── train.py
    │       └── play.py
    └── source/whole_body_tracking/whole_body_tracking/
        ├── robots/
        │   ├── dodo.py
        │   └── dodo_daimao.py
        └── tasks/tracking/
            ├── dodo/
            │   ├── __init__.py
            │   ├── flat_env_cfg.py
            │   └── agents/rsl_rl_ppo_cfg.py
            └── dodo_daimao/
                ├── __init__.py
                ├── flat_env_cfg.py
                └── agents/
                    ├── __init__.py
                    └── rsl_rl_ppo_cfg.py
```

---

## 🚀 Installation

### Trajectory Creation (Pinocchio / SE3 TrajOpt)
```bash
conda create -n trajopt python=3.13
conda activate trajopt
conda install -c conda-forge pinocchio meshcat-python cyipopt matplotlib numpy
```

### Whole Body Tracking (Isaac Lab)
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

## 🎬 DodoDaimao Backflip — Full Workflow

All commands run from the **MimicDodo project root** unless noted.

### Step 1 — Generate backflip trajectory
```bash
conda activate trajopt
export PYTHONPATH="$(pwd)/trajectory_creation/se3_trajopt/src"

# With visualizer (meshcat — open http://127.0.0.1:7000/static/ in browser)
python trajectory_creation/se3_trajopt/src/examples/agile_exps/dodo_daimao_backflip.py --vis

# Headless export only
python trajectory_creation/se3_trajopt/src/examples/agile_exps/dodo_daimao_backflip.py

# Output: whole_body_tracking/scripts/dodo_daimao_backflip.csv
```

### Step 2 — Convert CSV → NPZ (fast, no Isaac Sim needed)
```bash
# Still in trajopt env — finishes in <5 seconds
python whole_body_tracking/scripts/csv_to_npz_dodo_daimao.py \
  --input_file  whole_body_tracking/scripts/dodo_daimao_backflip.csv \
  --input_fps   100 \
  --output_name dodo_daimao_backflip \
  --output_fps  100

# Output: whole_body_tracking/dodo_daimao_backflip.npz
```

### Step 3 — Upload NPZ to wandb
```bash
cd whole_body_tracking
conda activate beyondmimic

python -c "
import wandb
name = 'dodo_daimao_backflip'
run  = wandb.init(project='csv_to_npz', name=name)
art  = run.log_artifact('dodo_daimao_backflip.npz', name=name, type='motions')
run.link_artifact(art, target_path=f'wandb-registry-motions/{name}')
print('Uploaded: motions/' + name)
run.finish()
"
```

### Step 4 — Replay to verify (recommended before training)
```bash
# From whole_body_tracking/
python scripts/replay_npz_dodo.py \
  --registry_name simon-kruelle-technical-university-of-munich/csv_to_npz/dodo_daimao_backflip
```

Check that the motion looks correct in Isaac Sim before spending GPU time training.

### Step 5 — Train
```bash
# From whole_body_tracking/
WANDB_DIR=/tmp/wandb_cache python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-DodoDaimao-v0 \
  --motion_file dodo_daimao_backflip.npz \
  --wandb_project whole_body_tracking \
  --headless
```

To resume from a checkpoint:
```bash
WANDB_DIR=/tmp/wandb_cache python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-DodoDaimao-v0 \
  --motion_file dodo_daimao_backflip.npz \
  --wandb_project whole_body_tracking \
  --resume=True \
  --load_run=<your-run-id> \
  --checkpoint=model_XXXXX.pt \
  --headless
```

### Step 6 — Evaluate trained policy
```bash
# From whole_body_tracking/
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-DodoDaimao-v0 \
  --num_envs=1 \
  --load_run "<your-run-id>" \
  --checkpoint "model_XXXXX.pt" \
  --motion_file dodo_daimao_backflip.npz \
  --video \
  --video_length 1000 \
  --disable_fabric
```

---

## 🎬 Original Dodo Backflip — Full Workflow

### Step 1 — Generate trajectory
```bash
conda activate trajopt
export PYTHONPATH="$(pwd)/trajectory_creation/se3_trajopt/src"
python trajectory_creation/se3_trajopt/src/examples/agile_exps/dodo_backflip.py --vis
```

### Step 2 — Convert CSV → NPZ (requires Isaac Sim)
```bash
cd whole_body_tracking
conda activate beyondmimic
python scripts/csv_to_npz_dodo.py \
  --input_file scripts/dodo_backflip.csv \
  --input_fps 100 \
  --output_name dodo_backflip \
  --output_fps 100
```

### Step 3 — Train
```bash
WANDB_DIR=/tmp/wandb_cache python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-Dodo-v0 \
  --motion_file dodo_backflip.npz \
  --wandb_project whole_body_tracking \
  --headless
```

### Step 4 — Evaluate
```bash
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-Dodo-v0 \
  --num_envs=1 \
  --load_run "<your-run-id>" \
  --checkpoint "model_XXXXX.pt" \
  --motion_file dodo_backflip.npz \
  --video \
  --video_length 1000 \
  --disable_fabric
```

---

## 🎬 Pinocchio Recording Scripts (Simple Motions)
```bash
conda activate trajopt
export PYTHONPATH="$(pwd)/trajectory_creation/se3_trajopt/src"

python trajectory_creation/pinocchio/record_walk.py
python trajectory_creation/pinocchio/record_jump.py
python trajectory_creation/pinocchio/record_kangaroo.py
python trajectory_creation/pinocchio/record_stand.py
```

Output: CSV files saved to `whole_body_tracking/scripts/`

---

## 🤖 Robot Configurations

### Dodo (original)
- **URDF**: `assets/dodo/urdf/dodo.urdf`
- **Joints**: `left_joint_1–4`, `right_joint_1–4`
- **Isaac Lab task**: `Tracking-Flat-Dodo-v0`
- **CSV converter**: `scripts/csv_to_npz_dodo.py` (requires Isaac Sim)

### DodoDaimao
- **URDF**: `assets/dodo_daimao/urdf/dodo_daimao.urdf`
- **Joints**: `hip_left/right`, `upper_leg_left/right`, `lower_leg_left/right`, `foot_left/right`
- **Isaac Lab task**: `Tracking-Flat-DodoDaimao-v0`
- **CSV converter**: `scripts/csv_to_npz_dodo_daimao.py` (Pinocchio-based, fast)
- **Key fixes applied**:
  - `hip_left` mass corrected 0 → 0.4286 kg (was causing asymmetric dynamics)
  - All collision geometries replaced with primitives (box for body and feet, none for legs/hips) — matches dodo.urdf pattern and is required for Isaac Sim performance
  - Two contact points per foot (toe + heel) added in wrapper — required for yaw stability
  - Standing height corrected to `q[2] = 0.435902` (feet exactly at z=0)

---

## 📋 Quick Reference Cheat Sheet

### DodoDaimao
```bash
# From MimicDodo root
conda activate trajopt
export PYTHONPATH="$(pwd)/trajectory_creation/se3_trajopt/src"
python trajectory_creation/se3_trajopt/src/examples/agile_exps/dodo_daimao_backflip.py
python whole_body_tracking/scripts/csv_to_npz_dodo_daimao.py --input_file whole_body_tracking/scripts/dodo_daimao_backflip.csv --input_fps 100 --output_name dodo_daimao_backflip --output_fps 100

cd whole_body_tracking && conda activate beyondmimic
python -c "import wandb; run=wandb.init(project='csv_to_npz',name='dodo_daimao_backflip'); art=run.log_artifact('dodo_daimao_backflip.npz',name='dodo_daimao_backflip',type='motions'); run.link_artifact(art,target_path='wandb-registry-motions/dodo_daimao_backflip'); run.finish()"
python scripts/replay_npz_dodo.py --registry_name simon-kruelle-technical-university-of-munich/csv_to_npz/dodo_daimao_backflip
WANDB_DIR=/tmp/wandb_cache python scripts/rsl_rl/train.py --task=Tracking-Flat-DodoDaimao-v0 --motion_file dodo_daimao_backflip.npz --wandb_project whole_body_tracking --headless
```

### Dodo
```bash
conda activate trajopt && export PYTHONPATH="$(pwd)/trajectory_creation/se3_trajopt/src"
python trajectory_creation/se3_trajopt/src/examples/agile_exps/dodo_backflip.py

cd whole_body_tracking && conda activate beyondmimic
python scripts/csv_to_npz_dodo.py --input_file scripts/dodo_backflip.csv --input_fps 100 --output_name dodo_backflip --output_fps 100
WANDB_DIR=/tmp/wandb_cache python scripts/rsl_rl/train.py --task=Tracking-Flat-Dodo-v0 --motion_file dodo_backflip.npz --wandb_project whole_body_tracking --headless
```

---

## 🛠️ Troubleshooting

### Isaac Sim loads very slowly / freezes on URDF import
All `<collision>` bodies in the URDF must use primitive shapes (`<box>`, `<cylinder>`, `<sphere>`), not `<mesh>`. Full mesh collision causes Isaac Sim's convex decomposition to run on every triangle. After editing the URDF, clear the USD cache so Isaac Sim regenerates it:
```bash
find ~/.local/share/ov -name "*dodo_daimao*" -delete 2>/dev/null
find /tmp -name "*dodo_daimao*" -delete 2>/dev/null
```
Also ensure `force_usd_conversion=True` in `dodo_daimao.py`.

### csv_to_npz is slow (2–3 min)
Use `csv_to_npz_dodo_daimao.py` instead of `csv_to_npz_dodo.py`. The dodo_daimao version uses Pinocchio FK and runs in the `trajopt` conda env in under 5 seconds — no Isaac Sim boot required.

### Yaw rotation / spinning during walking or backflip
The `DodoDaimaoWrapper` must register **two** contact frames per foot (toe + heel). A single point contact has no yaw resistance and the solver freely spins the body. Check `DodoDaimaoWrapper.__init__` for the `_add_foot_frame` calls.

### Wrong foot height / feet below ground at start
`go_neutral()` uses hardcoded `q[2] = 0.435902`. If you change the neutral joint angles, recompute with:
```python
robot = DodoDaimao()
q = robot.go_neutral()
robot.fk_all(q)
fid = robot.model.getFrameId("left_toe")
foot_z = robot.data.oMf[fid].translation[2]
print(foot_z)  # should be ~0.0
# If not: set q[2] = 0.435902 - foot_z and hardcode the result
```

### `k1 == k2` assertion error in backflip script
The flight phase must produce at least 1 timestep: `flight_duration / DT >= 1`. Current setting: `0.45s / 0.05s = 9 steps`. If you change `DT`, scale the flight duration accordingly.

### `hip_left` zero mass (asymmetric dynamics)
Already fixed in the current `dodo_daimao.urdf`. If you regenerate the URDF from SolidWorks, check that `hip_left` has `<mass value="0.428645410212592"/>` — the exporter originally set it to 0.

---

## 📄 License

See individual project directories for license information.