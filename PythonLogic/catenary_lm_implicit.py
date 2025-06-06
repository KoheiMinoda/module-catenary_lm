import math
import numpy as np
import csv 

def myasinh(X):
    return math.log(X + math.sqrt(X*X + 1.0))

def myacosh(X):
    if X < 1.0:
        if X > 1.0 - 1e-9:
            X = 1.0
        else:
            raise ValueError(f"[Error] Domain error for myacosh: X = {X} < 1")
    return math.log(X + math.sqrt(X*X - 1.0))

FP_COORDS = {"x": 5.2, "y": 0.0, "z": -70.0}
AP_COORDS = {"x": 853.87, "y": 0.0, "z": -320.0}
L_TOTAL = 902.2  # [m]
XACC = 1e-4

MAXIT_P0_SOLVER = 100
MAXIT_RTSAFE = 200

h_span = abs(FP_COORDS["z"] - AP_COORDS["z"])
L_APFP = math.sqrt((FP_COORDS["x"] - AP_COORDS["x"])**2 +
                   (FP_COORDS["y"] - AP_COORDS["y"])**2)

# 振動の設定
AMP_FL = 1.0 
PERIOD_FL = 20.0 
OMEGA_FL = 2.0*math.pi / PERIOD_FL

d_param = (L_APFP - (L_TOTAL - h_span)) / h_span
l_param = L_TOTAL / h_span

_current_p0_for_funcd = 0.0

# ----- ランプ関数 -----

RAMP_T = 10.0
def grav_scale(time):
    if time <= 0.0:
        return 0.0
    if time >= RAMP_T:
        return 1.0
    fsf = time / RAMP_T
    return 3.0*fsf*fsf - 2.0*fsf*fsf*fsf
# -----------------------

def _solve_for_p0_in_funcd(x_candidate, l_val, xacc_p0):
    p0 = 0.0

    for _ in range(MAXIT_P0_SOLVER):
        cos_p0 = math.cos(p0)
        if abs(cos_p0) < 1e-12:
            break 

        tan_p0 = math.tan(p0)
        func1 = 1.0 / x_candidate + 1.0 / cos_p0
        
        sqrt_arg_f1 = func1**2 - 1.0
        
        if sqrt_arg_f1 < 0 and sqrt_arg_f1 > -1e-9: 
            sqrt_arg_f1 = 0.0
        elif sqrt_arg_f1 < 0:
            return p0

        sqrt_val_f1 = math.sqrt(sqrt_arg_f1)
        f1_for_p0 = x_candidate * (sqrt_val_f1 - tan_p0) - l_val

        if abs(f1_for_p0) < xacc_p0:
            return p0

        if abs(sqrt_val_f1) < 1e-9: 
            break

        term_A_df1 = func1 * tan_p0 / (cos_p0 * sqrt_val_f1)
        term_B_df1 = 1.0 / (cos_p0**2) 
        df1_for_p0 = x_candidate * (term_A_df1 - term_B_df1)

        if abs(df1_for_p0) < 1e-12:
            break
        
        p0 = p0 - f1_for_p0 / df1_for_p0
        
        if not (-math.pi/2 * 0.99 < p0 < math.pi/2 * 0.99) :
            break
            
    return p0


def _funcd_equations(x_val, d_val, l_val, xacc_for_p0_solver):
    global _current_p0_for_funcd 
    f_res, df_res = 0.0, 0.0
    p0_local = 0.0

    if abs(x_val) < 1e-9:
        f_res = -d_val
        df_res = 0.0 
        p0_local = 0.0

    elif x_val > 0:

        if l_val <= 0: 
            X1 = 1.0 / x_val + 1.0
            term_acosh_X1 = myacosh(X1) 
            sqrt_1_2x = math.sqrt(1.0 + 2.0 * x_val)
            f_res = x_val * term_acosh_X1 - sqrt_1_2x + 1.0 - d_val
            
            sqrt_X1sq_m1 = math.sqrt(max(0,X1**2 - 1.0))
            if abs(x_val * sqrt_X1sq_m1) < 1e-9: 
                df_res = term_acosh_X1 - (1.0 / sqrt_1_2x) 
            else:
                df_res = term_acosh_X1 - (1.0 / sqrt_1_2x) - (1.0 / (x_val * sqrt_X1sq_m1))
            
            p0_local = 0.0

        else: 
            threshold_x = (l_val**2 - 1.0) / 2.0

            if x_val > threshold_x:
                p0_local = _solve_for_p0_in_funcd(x_val, l_val, xacc_for_p0_solver)
                tan_p0 = math.tan(p0_local)
                X2 = l_val / x_val + tan_p0
                asinh_X2 = myasinh(X2)
                asinh_tan_p0 = myasinh(tan_p0)
                f_res = x_val * (asinh_X2 - asinh_tan_p0) - l_val + 1.0 - d_val
                
                sqrt_X2sq_p1 = math.sqrt(X2**2 + 1.0)
                if abs(x_val * sqrt_X2sq_p1) < 1e-9:
                    df_res = asinh_X2 - asinh_tan_p0
                else:
                    df_res = (asinh_X2 - asinh_tan_p0) - l_val / (x_val * sqrt_X2sq_p1)
            
            else: 
                X5 = 1.0 / x_val + 1.0
                term_acosh_X5 = myacosh(X5)
                sqrt_1_2x_alt = math.sqrt(1.0 + 2.0 * x_val)
                f_res = x_val * term_acosh_X5 - sqrt_1_2x_alt + 1.0 - d_val

                sqrt_X5sq_m1 = math.sqrt(max(0,X5**2 - 1.0))
                if abs(x_val * sqrt_X5sq_m1) < 1e-9:
                    df_res = term_acosh_X5 - (1.0 / sqrt_1_2x_alt)
                else:
                    df_res = term_acosh_X5 - (1.0 / sqrt_1_2x_alt) - (1.0 / (x_val * sqrt_X5sq_m1))
                p0_local = 0.0

    else:
        raise ValueError("[Error] x_val < 0 encountered in _funcd_equations")
    
    _current_p0_for_funcd = p0_local
    return f_res, df_res, p0_local


def _rtsafe_solver(x1_rt, x2_rt, xacc_rt, d_rt, l_rt, xacc_p0_rt):
    p0_for_root = 0.0 
    
    f_low, _, p0_low = _funcd_equations(x1_rt, d_rt, l_rt, xacc_p0_rt)
    f_high, _, p0_high = _funcd_equations(x2_rt, d_rt, l_rt, xacc_p0_rt)

    if (f_low > 0 and f_high > 0) or (f_low < 0 and f_high < 0):
        raise ValueError(f"[Error] Root not bracketed in rtsafe: f({x1_rt})={f_low}, f({x2_rt})={f_high}")

    if f_low == 0: p0_for_root = p0_low; return x1_rt, p0_for_root
    if f_high == 0: p0_for_root = p0_high; return x2_rt, p0_for_root

    if f_low < 0:
        xl_rt, xh_rt = x1_rt, x2_rt
    else:
        xl_rt, xh_rt = x2_rt, x1_rt
    
    rts_curr = 0.5 * (x1_rt + x2_rt)
    dx_old = abs(x2_rt - x1_rt)
    dx_curr = dx_old
    
    f_curr, df_curr, p0_curr = _funcd_equations(rts_curr, d_rt, l_rt, xacc_p0_rt)
    p0_for_root = p0_curr

    for _ in range(MAXIT_RTSAFE):
        if (((rts_curr - xh_rt) * df_curr - f_curr) * ((rts_curr - xl_rt) * df_curr - f_curr) > 0.0) or \
           (abs(2.0 * f_curr) > abs(dx_old * df_curr)):
            dx_old = dx_curr
            dx_curr = 0.5 * (xh_rt - xl_rt)
            rts_curr = xl_rt + dx_curr

            if xl_rt == rts_curr:
                _, _, p0_for_root = _funcd_equations(rts_curr, d_rt, l_rt, xacc_p0_rt)
                return rts_curr, p0_for_root
        
        else:
            dx_old = dx_curr
            if abs(df_curr) < 1e-12:
                dx_curr = 0.5 * (xh_rt - xl_rt)
                rts_curr = xl_rt + dx_curr

                if xl_rt == rts_curr: 
                     _, _, p0_for_root = _funcd_equations(rts_curr, d_rt, l_rt, xacc_p0_rt)
                     return rts_curr, p0_for_root
            
            else:
                dx_curr = f_curr / df_curr

            temp_rts = rts_curr
            rts_curr -= dx_curr
            if temp_rts == rts_curr: 
                _, _, p0_for_root = _funcd_equations(rts_curr, d_rt, l_rt, xacc_p0_rt)
                return rts_curr, p0_for_root
        
        if abs(dx_curr) < xacc_rt:
            _, _, p0_for_root = _funcd_equations(rts_curr, d_rt, l_rt, xacc_p0_rt)
            return rts_curr, p0_for_root

        f_curr, df_curr, p0_curr = _funcd_equations(rts_curr, d_rt, l_rt, xacc_p0_rt)
        p0_for_root = p0_curr 

        if f_curr < 0:
            xl_rt = rts_curr
        else:
            xh_rt = rts_curr
            
    raise RuntimeError("[Error] Max iterations exceeded in rtsafe")

ans_x_final = 0.0

if d_param <= 0:
    ans_x_final = 0.0
    print("Condition d_param <= 0 met. Ans_x set to 0.")

elif l_param >= 1.0 and d_param >= (math.sqrt(l_param**2 - 1.0) - (l_param - 1.0)):
    raise ValueError("[Error] Line length L_total is too short")

else:
    rt_x1 = XACC
    rt_x2 = max(100.0, l_param * l_param * 2.0) 

    f1_check, _, _ = _funcd_equations(rt_x1, d_param, l_param, XACC)
    f2_check, _, _ = _funcd_equations(rt_x2, d_param, l_param, XACC)

    if (f1_check * f2_check) > 0:
        rt_x1 = XACC
        rt_x2 = 1.0e6 
        f1_check_wide, _, _ = _funcd_equations(rt_x1, d_param, l_param, XACC)
        f2_check_wide, _, _ = _funcd_equations(rt_x2, d_param, l_param, XACC)
        if (f1_check_wide * f2_check_wide) > 0:
             raise ValueError("[Error] Cannot bracket root for rtsafe even with wide range")
        else:
             print(f"Using wide bracket for rtsafe: [{rt_x1}, {rt_x2}]")
    
    print(f"Solving for Ans_x with rtsafe: bracket [{rt_x1:.3e}, {rt_x2:.3e}], xacc={XACC}")
    ans_x_final, _ = _rtsafe_solver(rt_x1, rt_x2, XACC, d_param, l_param, XACC)

print(f"Calculated Ans_x: {ans_x_final:.6f}")

a_catenary = ans_x_final * h_span
print(f"Catenary parameter 'a': {a_catenary:.3f} m")

xf, zf = FP_COORDS["x"], FP_COORDS["z"]
xa, za = AP_COORDS["x"], AP_COORDS["z"]

Xs_rel = xa - xf 
Zs_rel = za - zf
S_line = L_TOTAL

if abs(a_catenary) < 1e-6 : 
    
    if abs(L_APFP) < 1e-6:
        nodes = []
        num_internal_nodes = 19
        num_segments = num_internal_nodes + 1
        s_seg_vert = L_TOTAL / num_segments
        z_start_hang = max(zf,za) 
        x_hang = xf 
        y_hang = 0.0
        
        for i_node in range(1, num_internal_nodes + 1):
            s_node_from_top = i_node * s_seg_vert
            node_z_hang = z_start_hang - s_node_from_top 
            node_z_val = zf - s_node_from_top 
            
            if node_z_val < za and L_TOTAL > h_span : 
                pass 
            
            nodes.append({"id": i_node, "x": x_hang, "y": y_hang, "z": node_z_val, "s_from_fp": s_node_from_top})
        
        internal_nodes_coords_final = nodes

    else: 
        print("Ans_x=0 with L_APFP > 0. Catenary failed.")
        internal_nodes_coords_final = []

else:
    denominator_term_sum_x = 2 * math.sinh(Xs_rel / (2 * a_catenary))
    
    if abs(denominator_term_sum_x) < 1e-9: 
        sum_norm_x_half = Zs_rel / Xs_rel if abs(Xs_rel) > 1e-6 else 0.0 
    
    else:
        sum_norm_x_half = myasinh( (Zs_rel / a_catenary) / denominator_term_sum_x )

    diff_norm_x_half = Xs_rel / (2 * a_catenary)
    
    x1_n = sum_norm_x_half - diff_norm_x_half 
    x2_n = sum_norm_x_half + diff_norm_x_half 

    x_gm = xf - a_catenary * x1_n
    z_offset = zf - a_catenary * math.cosh(x1_n)

    print(f"Global catenary: a={a_catenary:.3f}, x_gm={x_gm:.3f}, z_offset={z_offset:.3f}")
    
    num_internal_nodes = 19
    num_segments = num_internal_nodes + 1
    s_segment_len = S_line / num_segments 

    internal_nodes_coords_final = []

    for i_node in range(1, num_internal_nodes + 1):
        s_k_arc_from_fp = i_node * s_segment_len 

        arg_for_asinh = s_k_arc_from_fp / a_catenary + math.sinh(x1_n)
        xk_n = myasinh(arg_for_asinh)
        
        node_x_coord = x_gm + a_catenary * xk_n
        node_y_coord = 0.0 
        node_z_coord = a_catenary * math.cosh(xk_n) + z_offset
        
        internal_nodes_coords_final.append({
            "id": i_node,
            "x": node_x_coord,
            "y": node_y_coord,
            "z": node_z_coord,
            "s_from_fp": s_k_arc_from_fp
        })
    
    print("----------------------------------------------------------------------")
    print("Anchor Point:   X={:.3f}, Y={:.3f}, Z={:.3f}".format(
      AP_COORDS['x'], AP_COORDS['y'], AP_COORDS['z']))
    
    # ========= カテナリー理論による初期形状ここまで ==============

EA_GLOBAL = 5.0e8
CA_GLOBAL = 2.0e5

LineDiameter = 0.09017 # [m]
LineMass = 77.71 # [kg/m]
WaterRho = 1025.0 # [kg_m^3]
RHO_LINE = (LineMass - WaterRho*0.25*math.pi*LineDiameter**2) # [kg/m]
g = 9.80665

# 海底反力と摩擦力
SEABED_Z = AP_COORDS["z"]
K_SEABED = 1.0e7
C_SEABED = 1.0e4

MU_STATIC = 0.6 # 静止摩擦係数
MU_DYNAMIC = 0.5 # 動摩擦係数
V_SLIP_TOL = 1.0e-3 # 滑り判定速度

DT = 0.01
T_END  = 500.0

# ---- ノードとセグメントの生成 ------------------------------------------
nodes_xyz0 = [FP_COORDS] + internal_nodes_coords_final + [AP_COORDS]  # 21
num_nodes = len(nodes_xyz0) # 21
num_segs  = num_nodes - 1 # 20

segments = []
for k in range(num_segs):
    xi, xj = nodes_xyz0[k], nodes_xyz0[k+1]
    L0 = math.dist((xi['x'], xi['y'], xi['z']),
                   (xj['x'], xj['y'], xj['z']))
    segments.append({
        "i": k,
        "j": k+1,
        "L0": L0,
        "EA": EA_GLOBAL,
        "CA": CA_GLOBAL,
        "mass": RHO_LINE*L0
    })

# ---- 各ノードに質量を半分ずつ配分 --------------------------------------
m_node = [0.0]*num_nodes
for seg in segments:
    m_half = 0.5*seg["mass"]
    m_node[seg["i"]] += m_half
    m_node[seg["j"]] += m_half

nodes = []
for idx, coord in enumerate(nodes_xyz0):
    nodes.append({
        "pos": np.array([coord['x'], coord['y'], coord['z']], dtype=float),
        "vel": np.zeros(3),
        "mass": m_node[idx]
    })

# ---- 軸方向ばね･ダンパ力 ------------------------------------------------
def axial_forces(nodes, segments):
    
    f_node = [np.zeros(3) for _ in nodes]

    for seg in segments:
        i, j   = seg["i"], seg["j"]
        xi, xj = nodes[i]["pos"], nodes[j]["pos"]
        vi, vj = nodes[i]["vel"], nodes[j]["vel"]

        dx = xj - xi
        l  = np.linalg.norm(dx)
        if l < 1e-9: # 零長保護
            continue
        t = dx / l # 単位ベクトル

        Fel = seg["EA"] * (l - seg["L0"]) / seg["L0"]
        vrel = np.dot(vj - vi, t)
        Fd = seg["CA"] * vrel
        Fax = Fel + Fd
        if Fax < 0.0:
            Fax = 0.0 # 圧縮不可（張力のみ）

        Fvec = Fax * t
        f_node[i] +=  Fvec
        f_node[j] += -Fvec
    return f_node

# ---- 海底反力と摩擦力 ---------------

def seabed_contact_forces(nodes):

    f_node = [np.zeros(3) for _ in nodes]

    for k, nd in enumerate(nodes):
        z = nd["pos"][2]
        vz = nd["vel"][2]

        # ----- 反力 ------
        pen = SEABED_Z - z
        if pen <= 0.0:
            continue

        Fz_norm = K_SEABED*pen - C_SEABED*vz # 上向きを正
        if Fz_norm < 0.0:
            Fz_norm = 0.0 # 引張は無し

        # ----- 摩擦 ------
        v_xy = nd["vel"].copy()
        v_xy[2] = 0.0
        v_norm = np.linalg.norm(v_xy)

        if v_norm < V_SLIP_TOL:
            f_fric_cap = MU_STATIC * Fz_norm # 一応出したが，海底に斜面がないと意味ないかも
            F_fric = - v_xy * 0.0
        else:
            F_fric = - MU_DYNAMIC * Fz_norm * (v_xy / v_norm)

        f_node[k][2] += Fz_norm
        f_node[k] += F_fric
    
    return f_node

# ---- 残差 AssRes ----------

# ----  残差・ヤコビアン --------------------------------------------------
def assemble_residual(q_next, q_prev, v_prev, masses, segments, time_curr):

    N = len(masses)
    # ベクトルを 3成分ずつ並べ替えて nodes_like 構造へ
    q_mat = q_next.reshape((N,3))
    nodes_tmp = [{"pos":q_mat[i], "vel":None, "mass":masses[i]} for i in range(N)]
    # 速度は後退オイラー： v_{n+1} = (q_{n+1}-q_n)/dt
    v_next_mat = (q_next - q_prev)/DT
    for i in range(N):
        nodes_tmp[i]["vel"] = v_next_mat[i*3:(i+1)*3]

    f_int = axial_forces(nodes_tmp, segments)          # 3要素リスト×N
    f_int_flat = np.array(f_int).reshape(-1)           # (3N)

    # 外力（重力）も (3N)
    f_ext = np.zeros_like(f_int_flat)
    scale = grav_scale(time_curr)
    gvec = np.array([0.0,0.0,-g*scale])
    for i,m in enumerate(masses):
        f_ext[3*i:3*i+3] = m * gvec

    # 慣性項 M a
    a_flat = (v_next_mat.reshape(-1) - v_prev)/DT      # (3N)
    M_flat = np.repeat(masses, 3)                      # (3N)
    inert = M_flat * a_flat

    return inert - f_int_flat - f_ext                  # (3N)

# ---- 全体ヤコビアン AssJac ------

def assemble_jacobian(q_next, q_prev, v_prev, masses, segments, time_curr, eps=1e-6):
    R0 = assemble_residual(q_next, q_prev, v_prev, masses, segments, time_curr)
    size = R0.size
    J = np.zeros((size, size)) # 180 x 180
    for k in range(size):
        dq = np.zeros(size); dq[k] = eps
        Rk = assemble_residual(q_next+dq, q_prev, v_prev, masses, segments, time_curr)  # time_currを追加
        J[:,k] = (Rk - R0)/eps
    return J


# ---- 加速度計算 --------------------------------------------------------
def compute_acc(nodes, segments):
    f_axial = axial_forces(nodes, segments)
    f_seabed = seabed_contact_forces(nodes)

    acc = []
    for k, node in enumerate(nodes):
    
        if node["mass"] == 0.0:
            acc.append(np.zeros(3))
            continue

        Fg = np.array([0.0, 0.0, -node["mass"]*g])
        F_tot = f_axial[k] + f_seabed[k] + Fg # 合力

        acc.append(F_tot / node["mass"])
    return acc

OUTPUT_NODES = list(range(num_nodes)) # 0(FP) ~ 20(AP)
node_traj = {idx: [] for idx in OUTPUT_NODES} 

# ---- 時間積分ループ -----------------------------------------------------

print("\n--- Lumped-mass implicit simulation start ({} s, dt={} s) ---"
    .format(T_END, DT))

t = 0.0
q_prev_full  = np.concatenate([n["pos"] for n in nodes])
v_prev_full  = np.concatenate([n["vel"] for n in nodes])
masses_full  = np.array([n["mass"] for n in nodes])

# prescribe lists --------------------------------------
fixed_idx = [0, num_nodes-1]          # FL と AP
free_idx = [i for i in range(num_nodes) if i not in fixed_idx]
free_dof = np.array([[3*i,3*i+1,3*i+2] for i in free_idx]).flatten()

# 出力用
node_traj = {idx: [] for idx in OUTPUT_NODES}

while t <= T_END:
    t_n = t
    t_np1 = t + DT

    # --- フェアリーダーポイントを設定-------------------------
    x_fl = FP_COORDS['x'] + AMP_FL*np.sin(OMEGA_FL*t)
    vx_fl= AMP_FL*OMEGA_FL*np.cos(OMEGA_FL*t)
    nodes[0]["pos"][:] = [x_fl, FP_COORDS['y'], FP_COORDS['z']]
    nodes[0]["vel"][:] = [vx_fl, 0.0, 0.0]

    # 全自由度ベクトルを前回値で初期化
    q_init = np.concatenate([n["pos"] for n in nodes])
    v_prev = np.concatenate([n["vel"] for n in nodes])

    # --- unknown = free DOF のみ -------------------------------
    qk = q_init[free_dof].copy()

    # Newton 反復 ------------------------------------------
    for it in range(20):
        q_full          = q_init.copy()
        q_full[free_dof]= qk
        R = assemble_residual(q_full, q_prev_full, v_prev_full, masses_full, segments, time_curr=t_np1)[free_dof]
        if np.linalg.norm(R) < 1e-8:
            break
        J_full = assemble_jacobian(q_full, q_prev_full, v_prev_full, masses_full, segments, time_curr=t_np1)
        J = J_full[np.ix_(free_dof, free_dof)]
        dq = np.linalg.solve(J, -R)
        qk += dq

    # --- 収束後の状態を nodes に書き戻し -------------------------
    q_prev_full[free_dof] = qk
    v_new_full = (q_prev_full - v_prev_full*DT - q_prev_full)/(-DT)  # v_{n+1}
    for i,n in enumerate(nodes):
        n["pos"][:] = q_prev_full[3*i:3*i+3]
        n["vel"][:] = v_new_full[3*i:3*i+3]

    # ---- 記録
    for idx in OUTPUT_NODES:
        p = nodes[idx]["pos"]
        node_traj[idx].append([t, p[0], p[1], p[2]])

    t += DT

print("--- implicit simulation finished ---")


# ---- CSV ファイル書き出し ------------------------------------------------

for idx, rows in node_traj.items():
    fname = f"node_{idx}_traj.csv"
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time[s]", "x[m]", "y[m]", "z[m]"])
        writer.writerows(rows)
    print(f"内部ノード {idx} → {fname}")
