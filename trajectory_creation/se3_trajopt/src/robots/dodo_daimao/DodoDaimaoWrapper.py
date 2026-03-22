"""DodoDaimao robot wrapper for SE(3) trajectory optimization."""
import os
import numpy as np
import pinocchio as pin


class DodoDaimao:
    def __init__(self):
        wrapper_dir  = os.path.dirname(os.path.realpath(__file__))
        mimic_root   = os.path.abspath(os.path.join(wrapper_dir, "../../../../../"))
        asset_dir    = os.path.join(mimic_root, "assets", "dodo_daimao")
        urdf_path    = os.path.join(asset_dir, "urdf", "dodo_daimao.urdf")
        package_dirs = os.path.join(mimic_root, "assets")

        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path,
            root_joint=pin.JointModelFreeFlyer(),
            package_dirs=[package_dirs],
        )

        # TWO contact points per foot — toe and heel.
        # Single point has zero yaw resistance; the solver freely spins
        # the body around the stance foot. Two points create a line contact
        # that constrains yaw, matching how DodoWrapper works.
        #
        # Offsets relative to foot_left / foot_right joint frames.
        # Z = -0.04880 (sole height from foot joint, from URDF foot_sole origin)
        # X = ±0.04m (toe forward, heel backward — matches foot length)
        self._add_foot_frame("left_toe",   "foot_left",  [ 0.04,  0.0, -0.04880])
        self._add_foot_frame("left_heel",  "foot_left",  [-0.04,  0.0, -0.04880])
        self._add_foot_frame("right_toe",  "foot_right", [ 0.04,  0.0, -0.04880])
        self._add_foot_frame("right_heel", "foot_right", [-0.04,  0.0, -0.04880])

        self.data = self.model.createData()

        self.left_foot_frames  = ["left_toe",  "left_heel"]
        self.right_foot_frames = ["right_toe", "right_heel"]
        self.left_gripper_frames  = []
        self.right_gripper_frames = []

    def _add_foot_frame(self, frame_name, parent_joint_name, offset_xyz):
        parent_id    = self.model.getJointId(parent_joint_name)
        parent_frame = self.model.getFrameId(parent_joint_name, pin.FrameType.JOINT)
        placement    = pin.SE3(np.eye(3), np.array(offset_xyz))
        frame = pin.Frame(
            frame_name, parent_id, parent_frame,
            placement, pin.FrameType.OP_FRAME
        )
        self.model.addFrame(frame)

    def fk_all(self, q, v=None):
        if v is not None:
            pin.forwardKinematics(self.model, self.data, q, v)
        else:
            pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def go_neutral(self):
        q = pin.neutral(self.model)
        # Corrected height: feet exactly at z=0
        # 0.52 - 0.084098 = 0.435902
        q[2] = 0.435902
        # Joint order: 7=hip_left, 8=upper_leg_left, 9=lower_leg_left,
        #   10=foot_left, 11=hip_right, 12=upper_leg_right,
        #   13=lower_leg_right, 14=foot_right
        q[7:11]  = [0.0,  0.4, -0.7, 0.3]   # left leg
        q[11:15] = [0.0,  0.4, -0.7, 0.3]   # right leg
        self.fk_all(q)
        return q