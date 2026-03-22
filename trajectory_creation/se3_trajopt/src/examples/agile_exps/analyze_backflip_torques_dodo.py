"""
Analyse joint torques including contact reaction forces via RNEA.
"""
import sys, os
sys.path.insert(0, os.path.abspath("trajectory_creation/se3_trajopt/src"))

import numpy as np
import json
import glob
import pinocchio as pin
from robots.dodobot_v3.DodoWrapper import Dodo

# ── Config — swap these for dodo_daimao ───────────────────────────────────
ROBOT_NAME    = "dodo"
SOLUTION_DIR  = "trajopt_solutions_batch/dodo_flip"
JOINT_NAMES   = [
    "left_joint_1",  "left_joint_2",  "left_joint_3",  "left_joint_4",
    "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4",
]
EFFORT_LIMITS = {n: 3 for n in JOINT_NAMES}
VEL_LIMITS    = {n: 5 for n in JOINT_NAMES}
SHORT_NAMES   = {
    "left_joint_1":  r"\texttt{hip\_l}",
    "left_joint_2":  r"\texttt{thigh\_l}",
    "left_joint_3":  r"\texttt{knee\_l}",
    "left_joint_4":  r"\texttt{ankle\_l}",
    "right_joint_1": r"\texttt{hip\_r}",
    "right_joint_2": r"\texttt{thigh\_r}",
    "right_joint_3": r"\texttt{knee\_r}",
    "right_joint_4": r"\texttt{ankle\_r}",
}

# ── Load ──────────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(SOLUTION_DIR, "*.json")))
assert files, f"No files in {SOLUTION_DIR}"
print(f"Loading: {files[-1]}")
with open(files[-1]) as f:
    result = json.load(f)

nodes = result.get("nodes") or result["solution"]["nodes"]
K     = len(nodes)
dts   = [nodes[k]["dt"] for k in range(K)]
qs    = [np.array(nodes[k]["q"]) for k in range(K)]
vs    = [np.array(nodes[k]["v"]) for k in range(K)]
accs  = [np.array(nodes[k]["a"]) for k in range(K)]
DT    = dts[0]
print(f"K={K}, DT={DT}, q shape={qs[0].shape}")
print(f"First node forces keys: {list(nodes[0].get('forces', {}).keys())[:5]}")

# ── Load robot ─────────────────────────────────────────────────────────────
robot = Dodo()
model = robot.model
data  = robot.data

joint_ids = [model.getJointId(n) for n in JOINT_NAMES]
q_ids     = [model.joints[j].idx_q for j in joint_ids]
v_ids     = [model.joints[j].idx_v for j in joint_ids]

nj    = len(JOINT_NAMES)
q_arr = np.array([[q[idx] for idx in q_ids] for q in qs])
v_arr = np.gradient(q_arr, DT, axis=0)
a_arr = np.gradient(v_arr, DT, axis=0)

# ── Print raw forces structure for first grounded node ────────────────────
print("\n── Forces structure (node 0) ──")
for fname, fval in list(nodes[0].get("forces", {}).items())[:3]:
    print(f"  '{fname}': {type(fval)} = {fval}")

# ── Build external force vector from contact forces ───────────────────────
def build_fext(node_data, model):
    """Convert stored contact forces to pinocchio fext vector."""
    fext = [pin.Force.Zero() for _ in range(model.njoints)]
    forces_dict = node_data.get("forces", {})

    for frame_name, force_val in forces_dict.items():
        # force_val might be [fx,fy,fz] or [[fx,fy,fz],[tx,ty,tz]] or a scalar
        try:
            if isinstance(force_val, (int, float)):
                continue  # scalar, skip
            fv = np.array(force_val, dtype=float).flatten()
            if len(fv) < 3:
                continue
            f_world = fv[:3]

            # Find which joint this frame belongs to
            if not model.existFrame(frame_name):
                continue
            fid  = model.getFrameId(frame_name)
            frame = model.frames[fid]
            jid  = frame.parent

            # Get frame placement in world
            pin.forwardKinematics(model, data, qs[0])
            pin.updateFramePlacements(model, data)
            oMf = data.oMf[fid]

            # Force in joint frame
            f_local = oMf.rotation.T @ f_world
            fext[jid] += pin.Force(f_local, np.zeros(3))
        except Exception as e:
            pass  # skip malformed entries

    return fext

# ── RNEA with external forces ──────────────────────────────────────────────
tau_arr      = np.zeros((K, nj))
tau_arr_noFext = np.zeros((K, nj))

for k in range(K):
    q_full = qs[k].copy()
    v_full = vs[k].copy() if vs[k].shape[0] == model.nv else np.zeros(model.nv)
    a_full = accs[k].copy() if accs[k].shape[0] == model.nv else np.zeros(model.nv)

    # Without external forces (for comparison)
    tau_no = pin.rnea(model, data, q_full, v_full, a_full)
    for i, vid in enumerate(v_ids):
        tau_arr_noFext[k, i] = tau_no[vid]

    # With external forces
    fext = build_fext(nodes[k], model)
    tau_with = pin.rnea(model, data, q_full, v_full, a_full, fext)
    for i, vid in enumerate(v_ids):
        tau_arr[k, i] = tau_with[vid]

# ── Statistics ─────────────────────────────────────────────────────────────
def print_table(tau, label):
    rows = []
    print(f"\n{'='*84}  [{label}]")
    print(f"{'Joint':<22} {'τ_max':>8} {'τ_rms':>8} {'ω_max':>8}  {'τ_lim':>6}  {'τ%':>6}")
    print(f"{'':22} {'[Nm]':>8} {'[Nm]':>8} {'[r/s]':>8}  {'[Nm]':>6}")
    print("-"*70)
    for i, name in enumerate(JOINT_NAMES):
        tau_max = np.max(np.abs(tau[:, i]))
        tau_rms = np.sqrt(np.mean(tau[:, i]**2))
        v_max   = np.max(np.abs(v_arr[:, i]))
        elim    = EFFORT_LIMITS[name]
        util_t  = tau_max / elim * 100
        rows.append((name, tau_max, tau_rms, v_max, elim, util_t))
        print(f"{name:<22} {tau_max:>8.2f} {tau_rms:>8.2f} {v_max:>8.2f}  {elim:>6}  {util_t:>5.1f}%")
    print("="*70)
    return rows

rows_no  = print_table(tau_arr_noFext, "WITHOUT contact forces")
rows_yes = print_table(tau_arr,        "WITH contact forces")

# ── Flight detection ───────────────────────────────────────────────────────
flight_nodes = [k for k in range(K)
                if not nodes[k].get("forces") or
                all((np.linalg.norm(v) < 1e-3 if isinstance(v, list) else abs(v) < 1e-3)
                    for v in nodes[k]["forces"].values())]
k1 = min(flight_nodes) if flight_nodes else 20
k2 = max(flight_nodes) if flight_nodes else 28
print(f"\nFlight: nodes {k1}–{k2}  (t={sum(dts[:k1]):.2f}s–{sum(dts[:k2]):.2f}s)")

print("\n── Peak torque phase (WITH contact forces) ──")
for i, name in enumerate(JOINT_NAMES):
    k_peak = np.argmax(np.abs(tau_arr[:, i]))
    t_peak = sum(dts[:k_peak])
    phase  = "FLIGHT" if k1 <= k_peak <= k2 else "GROUND"
    print(f"  {name:<22}  τ_peak={tau_arr[k_peak,i]:+7.3f} Nm  at t={t_peak:.3f}s  ({phase})")

# ── LaTeX (WITH contact forces) ────────────────────────────────────────────
print("\n── LaTeX table (WITH contact forces) ──\n")
print(r"\begin{table}[t]")
print(r"\centering")
print(rf"\caption{{Peak joint torques during the optimized {ROBOT_NAME} backflip.}}")
print(rf"\label{{tab:backflip_torques_{ROBOT_NAME}}}")
print(r"\begin{tabular}{lrrrr}")
print(r"\hline")
print(r"Joint & $\tau_{\max}$\,[Nm] & $\tau_{\lim}$\,[Nm] & "
      r"$\omega_{\max}$\,[rad/s] & $\omega_{\lim}$\,[rad/s] \\")
print(r"\hline")
vlims = list(VEL_LIMITS.values())
for idx, (name, tau_max, tau_rms, v_max, elim, util_t) in enumerate(rows_yes):
    v_max_val = np.max(np.abs(v_arr[:, idx]))
    print(f"{SHORT_NAMES[name]} & {tau_max:.2f} & {elim} & {v_max_val:.2f} & {VEL_LIMITS[name]} \\\\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")