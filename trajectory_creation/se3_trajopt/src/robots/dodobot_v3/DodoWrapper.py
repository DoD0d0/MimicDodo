"""Dodo robot wrapper for SE(3) trajectory optimization.

Loads URDF and meshes from MimicDodo/assets/dodo/.
"""
import os
import numpy as np
import pinocchio as pin


class Dodo:
    def __init__(self):
        # Path to MimicDodo/assets (from robots/dodobot_v3/ -> 5 levels up to MimicDodo)
        wrapper_dir = os.path.dirname(os.path.realpath(__file__))
        mimic_root = os.path.abspath(os.path.join(wrapper_dir, "../../../../../"))
        asset_dir = os.path.join(mimic_root, "assets", "dodo")
        urdf_path = os.path.join(asset_dir, "urdf", "dodo.urdf")
        package_dirs = os.path.join(mimic_root, "assets")

        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path,
            root_joint=pin.JointModelFreeFlyer(),
            package_dirs=[package_dirs],
        )

        self._add_foot_frame("left_toe", "left_joint_4", [0.045, 0, -0.043])
        self._add_foot_frame("left_heel", "left_joint_4", [-0.045, 0, -0.043])
        self._add_foot_frame("right_toe", "right_joint_4", [0.045, 0, -0.043])
        self._add_foot_frame("right_heel", "right_joint_4", [-0.045, 0, -0.043])

        self.data = self.model.createData()

        self.left_foot_frames = ["left_toe", "left_heel"]
        self.right_foot_frames = ["right_toe", "right_heel"]
        self.left_gripper_frames = []
        self.right_gripper_frames = []

    def _add_foot_frame(self, frame_name, parent_joint_name, offset_xyz):
        parent_id = self.model.getJointId(parent_joint_name)
        parent_frame = self.model.getFrameId(parent_joint_name)
        placement = pin.SE3(np.eye(3), np.array(offset_xyz))
        frame = pin.Frame(frame_name, parent_id, parent_frame, placement, pin.FrameType.OP_FRAME)
        self.model.addFrame(frame)

    def fk_all(self, q, v=None):
        if v is not None:
            pin.forwardKinematics(self.model, self.data, q, v)
        else:
            pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def go_neutral(self):
        q = pin.neutral(self.model)
        q[2] = 0.52
        q[7:11] = [0.0, 0.4, -0.7, 0.3]   # Right leg
        q[11:15] = [0.0, 0.4, -0.7, 0.3]  # Left leg
        self.fk_all(q)
        return q
