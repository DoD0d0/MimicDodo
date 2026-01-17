# Blender Animation Workflow for Dodo Robot

This directory contains a Blender-based animation setup for the Dodo robot, providing an alternative workflow for creating and exporting robot trajectories.

## ⚠️ Status: **Not Ready for Production Use**

**This Blender animation workflow is currently under development and contains several known issues. It should not be used for generating production trajectories at this time.**

## 📋 Overview

The Blender animation setup provides a visual, keyframe-based approach to animating the Dodo robot. It includes:

- **Skeleton Model**: A rigged skeleton based on the Dodo URDF structure
- **Inverse Kinematics (IK) Controllers**: Well-designed foot controllers for intuitive animation
- **Trajectory Export**: Scripts to export animated motions to CSV format

## 🎬 Blender Setup

**Required Version:** Blender 4.5.4 LTS

The main Blender file (`Dodo_Walk.blend`) contains:
- Dodo robot model with skeleton rigging
- Animation keyframes
- Custom Python scripts for trajectory export

### Main Script

**`dodo_trajectory.py`**: Main script for exporting joint angles and robot state to CSV format

The Blender file includes additional scripts for:
- Joint angle calculations
- Coordinate transformations
- CSV export formatting

## 🔧 Features

### Strengths

✅ **Intuitive Animation Interface**: Visual keyframe editing in Blender  
✅ **IK Foot Controllers**: Well-designed inverse kinematics for feet positioning  
✅ **Interactive Preview**: Real-time visualization of robot motion

### Known Issues

❌ **Rotation Axes**: Not properly aligned with current URDF model (uses legacy axes)  
❌ **Collision Body**: Outdated, does not match current URDF geometry  
❌ **CSV Export**: Buggy joint angle calculation and export functionality  
❌ **Joint Angle Calculation**: Incorrect transformations in export script

## 🚧 Development Status

This workflow is actively being debugged and improved. The following components need attention:

1. **Rotation Axis Alignment**: Update to match current `assets/dodo/urdf/dodo.urdf` conventions
2. **Collision Geometry**: Synchronize with latest URDF collision models
3. **Export Script Fixes**: Correct joint angle calculations and CSV formatting
4. **Validation**: Verify exported trajectories match URDF joint conventions

## 💡 Use Cases

Once completed, this workflow will be useful for:

- **Artistic Motion Design**: Create stylized motions with visual feedback
- **Manual Keyframe Animation**: Fine-tune specific motion phases
- **Motion Exploration**: Rapidly prototype new motion ideas
- **Visual Iteration**: Preview motions before committing to optimization

## 🔄 Alternative Workflows

While this Blender workflow is under development, consider using:

1. **Pinocchio Scripts** (`trajectory_creation/pinocchio/`): Reliable, script-based motion generation
2. **SE(3) Trajectory Optimization** (`trajectory_creation/se3_trajopt/`): Physics-based motion optimization

Both alternatives are production-ready and fully functional.

## 📝 Notes

- The Blender file may reference old URDF paths - ensure you update them to point to `assets/dodo/urdf/dodo.urdf`
- Export scripts may need adjustments for the centralized asset structure
- Coordinate systems may differ between Blender and the simulation environments

## 🐛 Bug Reports

If you attempt to use this workflow and encounter issues, please document:
- Blender version
- Error messages
- Steps to reproduce
- Expected vs. actual behavior

---

**Last Updated:** 2025-01-XX  
**Status:** Under Development - Do Not Use for Production

