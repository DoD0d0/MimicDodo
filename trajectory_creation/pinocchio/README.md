# Pinocchio Trajectory Recording Scripts

This folder contains scripts for recording robot trajectories using Pinocchio. These scripts generate motion data that can be used for training and trajectory optimization.

## Robot Configuration

All scripts use the centralized Dodo robot URDF and meshes from:
- **URDF**: `assets/dodo/urdf/dodo.urdf`
- **Meshes**: `assets/dodo/meshes/`

The scripts automatically calculate the correct paths relative to their location, so they work regardless of where they are executed from.

## Available Scripts

- **`record_walk.py`** - Records a walking trajectory
- **`record_jump.py`** - Records a jumping motion
- **`record_kangaroo.py`** - Records kangaroo-style hopping motion
- **`record_stand.py`** - Generates a static standing pose

## Usage

Run any script from the project root directory:

```bash
python trajectory_creation/pinocchio/record_walk.py
python trajectory_creation/pinocchio/record_jump.py
python trajectory_creation/pinocchio/record_kangaroo.py
python trajectory_creation/pinocchio/record_stand.py
```

## Output

The scripts save trajectory data to CSV files in:
- `whole_body_tracking/scripts/`

Each script generates a CSV file with the robot's state (position, orientation, joint angles) at each time step, ready for use in training pipelines.

## Requirements

- Pinocchio
- NumPy
- Meshcat (for visualization)

