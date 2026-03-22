"""
Analyse joint torques for DodoDaimao backflip WITH contact forces.
Run from MimicDodo root:
    python trajectory_creation/se3_trajopt/src/examples/agile_exps/analyze_backflip_torques_dodo_daimao.py
"""
import sys, os
sys.path.insert(0, os.path.abspath("trajectory_creation/se3_trajopt/src"))

import numpy as np
import json
import glob
import pinocchio as pin
from robots.dodo_daimao.DodoDaimaoWrapper import DodoDaimao

SOLUTION_DIR  = "trajopt_solutions_batch/dodo_daimao_flip"
JOINT_NAMES   = [
    "hip_left",       "upper_leg_left",  "lower_leg_left",  "foot_left",
    "hip_right",      "upper_leg_right", "lower_leg_right", "foot_right",
]
EFFORT_LIMITS = {
    "hip_left": 27,        "upper_leg_left": 27,
    "lower_leg_left": 9,   "foot_left": 9,
    "hip_right": 27,       "upper_leg_right": 27,
    "lower_leg_right": 9,  "foot_right": 9,
}
VEL_LIMITS = {n: 6 for n in JOINT_NAMES}
SHORT_NAMES = {
    "hip_left":        r"\texttt{hip\_l}",
    "upper_leg_left":  r"\texttt{thigh\_l}",
    "lower_leg_left":  r"\texttt{knee\_l}",
    "foot_left":       r"\texttt{ankle\_l}",
    "hip_right":       r"\texttt{hip\_r}",
    "upper_leg_right": r"\texttt{thigh\_r}",
    "lower_leg_right": r"\texttt{knee\_r}",
    "foot_right":      r"\texttt{ankle\_r}",
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
print(f"Forces keys: {list(nodes[0].get('forces', {}).keys())}")

# ── Load robot ─────────────────────────────────────────────────────────────
robot     = DodoDaimao()
model     = robot.model
data      = robot.data

joint_ids = [model.getJointId(n) for n in JOINT_NAMES]
q_ids     = [model.joints[j].idx_q for j in joint_ids]
v_ids     = [model.joints[j].idx_v for j in joint_ids]

nj    = len(JOINT_NAMES)
q_arr = np.array([[q[idx] for idx in q_ids] for q in qs])
v_arr = np.gradient(q_arr, DT, axis=0)

# ── Build fext from contact forces ─────────────────────────────────────────
def build_fext(node_data, model, data, q):
    fext = [pin.Force.Zero() for _ in range(model.njoints)]
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    for frame_name, force_val in node_data.get("forces", {}).items():
        try:
            fv = np.array(force_val, dtype=float).flatten()
            if len(fv) < 3:
                continue
            f_world = fv[:3]
            if not model.existFrame(frame_name):
                continue
            fid   = model.getFrameId(frame_name)
            frame = model.frames[fid]
            jid   = frame.parentJoint
            oMf   = data.oMf[fid]
            f_local = oMf.rotation.T @ f_world
            fext[jid] += pin.Force(f_local, np.zeros(3))
        except Exception:
            pass
    return fext

# ── RNEA with contact forces ───────────────────────────────────────────────
tau_arr = np.zeros((K, nj))
for k in range(K):
    q_full = qs[k].copy()
    v_full = vs[k].copy() if vs[k].shape[0] == model.nv else np.zeros(model.nv)
    a_full = accs[k].copy() if accs[k].shape[0] == model.nv else np.zeros(model.nv)
    fext   = build_fext(nodes[k], model, data, q_full)
    tau_full = pin.rnea(model, data, q_full, v_full, a_full, fext)
    for i, vid in enumerate(v_ids):
        tau_arr[k, i] = tau_full[vid]

# ── Statistics ─────────────────────────────────────────────────────────────
print("\n" + "="*84)
print(f"{'Joint':<22} {'τ_max':>8} {'τ_rms':>8} {'ω_max':>8}  {'τ_lim':>6}  {'ω_lim':>6}  {'τ%':>6}")
print(f"{'':22} {'[Nm]':>8} {'[Nm]':>8} {'[r/s]':>8}  {'[Nm]':>6}  {'[r/s]':>6}")
print("-"*84)

rows = []
for i, name in enumerate(JOINT_NAMES):
    tau_max = np.max(np.abs(tau_arr[:, i]))
    tau_rms = np.sqrt(np.mean(tau_arr[:, i]**2))
    v_max   = np.max(np.abs(v_arr[:, i]))
    elim    = EFFORT_LIMITS[name]
    vlim    = VEL_LIMITS[name]
    util_t  = tau_max / elim * 100
    rows.append((name, tau_max, tau_rms, v_max, elim, vlim, util_t))
    print(f"{name:<22} {tau_max:>8.2f} {tau_rms:>8.2f} {v_max:>8.2f}  {elim:>6}  {vlim:>6}  {util_t:>5.1f}%")

print("="*84)

# ── Flight detection ───────────────────────────────────────────────────────
flight_nodes = [k for k in range(K)
                if not nodes[k].get("forces") or
                all((np.linalg.norm(v) < 1e-3 if isinstance(v, list) else abs(v) < 1e-3)
                    for v in nodes[k]["forces"].values())]
k1 = min(flight_nodes) if flight_nodes else 20
k2 = max(flight_nodes) if flight_nodes else 28
print(f"\nFlight: nodes {k1}–{k2}  (t={sum(dts[:k1]):.2f}s–{sum(dts[:k2]):.2f}s)")

print("\n── Peak torque phase ──")
for i, name in enumerate(JOINT_NAMES):
    k_peak = np.argmax(np.abs(tau_arr[:, i]))
    t_peak = sum(dts[:k_peak])
    phase  = "FLIGHT" if k1 <= k_peak <= k2 else "GROUND"
    print(f"  {name:<22}  τ_peak={tau_arr[k_peak,i]:+7.3f} Nm  at t={t_peak:.3f}s  ({phase})")

# ── LaTeX comparison table (paste Dodo values manually from previous run) ──
print("\n── LaTeX COMPARISON table (fill Dodo column from previous script) ──\n")
print(r"\begin{table}[t]")
print(r"\centering")
print(r"\caption{Comparison of peak joint torques and velocities for the optimized")
print(r"         backflip trajectories of Dodo and DodoDaimao. Values exceeding the")
print(r"         actuator limit are marked in bold.}")
print(r"\label{tab:backflip_torques_comparison}")
print(r"\begin{tabular}{lrr|rr|rr}")
print(r"\hline")
print(r" & \multicolumn{2}{c|}{Dodo} & \multicolumn{2}{c|}{DodoDaimao} & \multicolumn{2}{c}{Limits} \\")
print(r"Joint & $\tau_{\max}$ & $\omega_{\max}$ & $\tau_{\max}$ & $\omega_{\max}$ & $\tau_{\lim}$ & $\omega_{\lim}$ \\")
print(r" & [Nm] & [rad/s] & [Nm] & [rad/s] & [Nm] & [rad/s] \\")
print(r"\hline")

# Dodo values from previous run (hardcoded — copy from output above)
dodo_vals = {
    "hip":   (2.58, 4.53),
    "thigh": (4.01, 5.00),
    "knee":  (5.14, 5.00),
    "ankle": (0.32, 4.96),
}
dodo_lims = {"hip": (3,5), "thigh": (3,5), "knee": (3,5), "ankle": (3,5)}

joint_keys = ["hip", "thigh", "knee", "ankle"]
side_pairs = list(zip(rows[0:4], rows[4:8]))  # left, right averaged

for idx, jkey in enumerate(joint_keys):
    left  = rows[idx]
    right = rows[idx + 4]
    # Average left/right for daimao (symmetric)
    dd_tau = (left[1] + right[1]) / 2
    dd_vel = (left[3] + right[3]) / 2
    dd_lim_tau = left[4]
    dd_lim_vel = left[5]

    do_tau, do_vel = dodo_vals[jkey]
    do_lim_tau, do_lim_vel = dodo_lims[jkey]

    # Bold if exceeds limit
    do_tau_str = rf"\textbf{{{do_tau:.1f}}}" if do_tau > do_lim_tau else f"{do_tau:.1f}"
    dd_tau_str = rf"\textbf{{{dd_tau:.1f}}}" if dd_tau > dd_lim_tau else f"{dd_tau:.1f}"
    do_vel_str = rf"\textbf{{{do_vel:.2f}}}" if do_vel > do_lim_vel else f"{do_vel:.2f}"
    dd_vel_str = rf"\textbf{{{dd_vel:.2f}}}" if dd_vel > dd_lim_vel else f"{dd_vel:.2f}"

    print(rf"\texttt{{{jkey}}} & {do_tau_str} & {do_vel_str} & {dd_tau_str} & {dd_vel_str} & {dd_lim_tau} & {dd_lim_vel} \\")

print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")