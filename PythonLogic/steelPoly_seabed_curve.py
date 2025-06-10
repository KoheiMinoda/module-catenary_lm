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
AMP_FL = 4.0 
PERIOD_FL = 20.0 
OMEGA_FL = 2.0*math.pi / PERIOD_FL

d_param = (L_APFP - (L_TOTAL - h_span)) / h_span
l_param = L_TOTAL / h_span

_current_p0_for_funcd = 0.0

# ----- ランプ関数 -----
RAMP_T = 30.0  # 10.0 → 30.0
def grav_scale(time):
    if time <= 0.0:
        return 0.0
    if time >= RAMP_T:
        return 1.0
    fsf = time / RAMP_T
    # より滑らかなランプ関数
    return fsf * fsf * (3.0 - 2.0 * fsf)
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

# ---- 材料特性の定義：段階的変化 --------------------------------
EA_STEEL = 5.0e8
CA_STEEL = 2.0e5

# 剛性差を小さくすると安定性向上
EA_POLYESTER = 1.0e8
CA_POLYESTER = 1.0e5
POLYESTER_MASS_PER_LENGTH = 45.0  # [kg/m]

POLYESTER_START_NODE = 5
POLYESTER_END_NODE = 10
TRANSITION_SEGMENTS = 2  # 遷移セグメント数

def get_material_properties(segment_index):
    
    if segment_index < POLYESTER_START_NODE - TRANSITION_SEGMENTS:
        # 完全なスチール
        return EA_STEEL, CA_STEEL, RHO_LINE, "Steel"
    
    elif segment_index < POLYESTER_START_NODE:
        # スチール→ポリエステル遷移
        ratio = (segment_index - (POLYESTER_START_NODE - TRANSITION_SEGMENTS)) / TRANSITION_SEGMENTS
        ea = EA_STEEL * (1 - ratio) + EA_POLYESTER * ratio
        ca = CA_STEEL * (1 - ratio) + CA_POLYESTER * ratio
        mass = RHO_LINE * (1 - ratio) + POLYESTER_MASS_PER_LENGTH * ratio
        return ea, ca, mass, "Transition_SP"
    
    elif segment_index < POLYESTER_END_NODE:
        # 完全なポリエステル
        return EA_POLYESTER, CA_POLYESTER, POLYESTER_MASS_PER_LENGTH, "Polyester"
    
    elif segment_index < POLYESTER_END_NODE + TRANSITION_SEGMENTS:
        # ポリエステル→スチール遷移
        ratio = (segment_index - POLYESTER_END_NODE) / TRANSITION_SEGMENTS
        ea = EA_POLYESTER * (1 - ratio) + EA_STEEL * ratio
        ca = CA_POLYESTER * (1 - ratio) + CA_STEEL * ratio
        mass = POLYESTER_MASS_PER_LENGTH * (1 - ratio) + RHO_LINE * ratio
        return ea, ca, mass, "Transition_PS"
    
    else:
        # 完全なスチール
        return EA_STEEL, CA_STEEL, RHO_LINE, "Steel"

LineDiameter = 0.09017 # [m]
LineMass = 77.71 # [kg/m]
WaterRho = 1025.0 # [kg_m^3]
RHO_LINE = (LineMass - WaterRho*0.25*math.pi*LineDiameter**2) # [kg/m]
g = 9.80665

# 海底反力と摩擦力
SEABED_BASE_Z = AP_COORDS["z"]
SEABED_AMPLITUDE = 10.0
SEABED_WAVELENGTH = 200.0

K_SEABED = 1.0e7
C_SEABED = 1.0e4

MU_STATIC = 0.6 # 静止摩擦係数
MU_DYNAMIC = 0.5 # 動摩擦係数
V_SLIP_TOL = 1.0e-3 # 滑り判定速度

# 時間積分パラメータ
DT = 0.005
T_END  = 500.0

RAYLEIGH_ALPHA = 0.05  # 質量比例減衰
RAYLEIGH_BETA = 0.001  # 剛性比例減衰 使用していない

# ---- ノードとセグメントの生成 ---------------------------
nodes_xyz0 = [FP_COORDS] + internal_nodes_coords_final + [AP_COORDS]  # 21
num_nodes = len(nodes_xyz0) # 21
num_segs  = num_nodes - 1 # 20

segments = []
for k in range(num_segs):
    xi, xj = nodes_xyz0[k], nodes_xyz0[k+1]
    L0 = math.dist((xi['x'], xi['y'], xi['z']),
                   (xj['x'], xj['y'], xj['z']))
    
    ea_val, ca_val, mass_per_length, material_type = get_material_properties(k)

    segments.append({
        "i": k,
        "j": k+1,
        "L0": L0,
        "EA": ea_val,
        "CA": ca_val,
        "mass": mass_per_length * L0,
        "material": material_type
    })
    
    print(f"セグメント {k}-{k+1}: {material_type}, EA={ea_val:.2e}, CA={ca_val:.2e}")

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

# ---- 軸方向ばね･ダンパ ---------------------------------------
def axial_forces(nodes, segments):
    f_node = [np.zeros(3) for _ in nodes]
    MAX_FORCE = 5.0e7  # 最大軸力制限 [N]

    for seg in segments:
        i, j   = seg["i"], seg["j"]
        xi, xj = nodes[i]["pos"], nodes[j]["pos"]
        vi, vj = nodes[i]["vel"], nodes[j]["vel"]

        dx = xj - xi
        l  = np.linalg.norm(dx)
        if l < 1e-9: # 零長保護
            continue
        t = dx / l # 単位ベクトル

        strain = (l - seg["L0"]) / seg["L0"]
        
        # 材料に応じた軸力計算
        if "Polyester" in seg.get("material", ""):
            # ポリエステルロープの非線形特性
            if strain < 0.01:  # 1%未満のひずみ
                Fel = seg["EA"] * 0.1 * strain / 0.01  # 非常に低剛性
            elif strain < 0.05:  # 1%〜5%のひずみ
                Fel = seg["EA"] * (0.1 + 0.4 * (strain - 0.01) / 0.04)
            else:  # 5%以上のひずみ
                Fel = seg["EA"] * (0.5 + 0.5 * min((strain - 0.05) / 0.05, 1.0))
            Fel = Fel * seg["L0"]
        else:
            # スチールケーブルの線形特性
            Fel = seg["EA"] * strain

        vrel = np.dot(vj - vi, t)
        Fd = seg["CA"] * vrel

        # 総軸力（制限付き）
        Fax = Fel + Fd
        Fax = max(0.0, min(Fax, MAX_FORCE))  # 0〜MAX_FORCEに制限

        Fvec = Fax * t
        f_node[i] +=  Fvec
        f_node[j] += -Fvec
    return f_node

# ---- 海底面の高さを計算 ------------------------------------------------
def get_seabed_z(x_coord):
    return SEABED_BASE_Z + SEABED_AMPLITUDE * math.sin(2 * math.pi * x_coord / SEABED_WAVELENGTH)

# ---- 海底反力と摩擦力 --------------------------------------------------
def seabed_contact_forces(nodes):
    f_node = [np.zeros(3) for _ in nodes]

    for k, nd in enumerate(nodes):
        x, z = nd["pos"][0], nd["pos"][2]
        vz = nd["vel"][2]

        seabed_z_local = get_seabed_z(x)

        # ----- 反力 ------
        pen = seabed_z_local - z
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
            F_fric = np.zeros(2)
        else:
            F_fric_2d = - MU_DYNAMIC * Fz_norm * (v_xy[:2] / v_norm)
            F_fric = np.concatenate([F_fric_2d, [0.0]])

        f_node[k][2] += Fz_norm
        f_node[k][:2] += F_fric[:2]
    
    return f_node

# ---- 加速度計算（レーリー減衰付き） ------------------------------------
def compute_acc(nodes, segments, time):
    f_axial = axial_forces(nodes, segments)
    f_seabed = seabed_contact_forces(nodes)
    scale = grav_scale(time)

    acc = []
    for k, node in enumerate(nodes):
        if node["mass"] == 0.0:
            acc.append(np.zeros(3))
            continue

        # 重力
        Fg = np.array([0.0, 0.0, -node["mass"]*g*scale])
        
        # レーリー減衰力（数値安定性向上）
        F_rayleigh = -RAYLEIGH_ALPHA * node["mass"] * node["vel"]
        
        # 合力
        F_tot = f_axial[k] + f_seabed[k] + Fg + F_rayleigh

        acc.append(F_tot / node["mass"])
    return acc

OUTPUT_NODES = list(range(num_nodes)) # 0(FP) ~ 20(AP)
node_traj = {idx: [] for idx in OUTPUT_NODES} 

# ---- 時間積分ループ -----------------------------------------------------
print("\n--- Lumped-mass explicit simulation start ({} s, dt={} s) ---"
      .format(T_END, DT))
t = 0.0
traj_out = []

nodes[0]["mass"] = 0.0

step_count = 0
output_interval = int(0.1 / DT)  # 0.1秒ごとに出力

while t <= T_END:
    
    x_fl  = FP_COORDS['x'] + AMP_FL * math.sin(OMEGA_FL * t)
    vx_fl = AMP_FL * OMEGA_FL * math.cos(OMEGA_FL * t)
    nodes[0]["pos"][:] = [x_fl, FP_COORDS['y'], FP_COORDS['z']]
    nodes[0]["vel"][:] = [vx_fl, 0.0, 0.0]

    a_list = compute_acc(nodes, segments, t)

    # 速度制限（安定性向上）
    MAX_VELOCITY = 20.0  # [m/s]
    
    for k in range(1, num_nodes - 1):
        # 速度更新
        nodes[k]["vel"] += a_list[k]*DT
        
        # 速度制限適用
        vel_mag = np.linalg.norm(nodes[k]["vel"])
        if vel_mag > MAX_VELOCITY:
            nodes[k]["vel"] = nodes[k]["vel"] * (MAX_VELOCITY / vel_mag)
        
        # 位置更新
        nodes[k]["pos"] += nodes[k]["vel"]*DT

    # データ記録（間引き出力で効率化）
    if step_count % output_interval == 0:
        for idx in OUTPUT_NODES:
            p = nodes[idx]["pos"]
            node_traj[idx].append([t, p[0], p[1], p[2]])
    
    t += DT
    step_count += 1
    
    # 進捗表示
    if step_count % (10 * output_interval) == 0:
        print(f"Time: {t:.1f}s / {T_END}s ({t/T_END*100:.1f}%)")

print("--- simulation finished ---")

# ---- セグメント材料情報をCSVで出力 ------------------------------------
segment_info = []
for k, seg in enumerate(segments):
    segment_info.append([
        k, # セグメント番号
        seg["i"], # 開始ノード
        seg["j"], # 終了ノード
        seg["material"], # 材料種類
        seg["EA"], # 軸剛性
        seg["CA"], # 減衰係数
        seg["L0"] # 初期長さ
    ])

with open("segment_materials.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["segment_id", "node_i", "node_j", "material", "EA[N]", "CA[Ns/m]", "L0[m]"])
    writer.writerows(segment_info)

print("セグメント材料情報 → segment_materials.csv")

# ---- 海底形状データの出力 --------------------------------------------
print("--- Outputting seabed profile ---")

# x座標の範囲を決定（FPからAPまで + 余裕）
x_min = min(FP_COORDS["x"], AP_COORDS["x"]) - 100.0
x_max = max(FP_COORDS["x"], AP_COORDS["x"]) + 100.0
x_seabed = np.linspace(x_min, x_max, 200)  # 200点でサンプリング

seabed_profile = []
for x in x_seabed:
    z_seabed = get_seabed_z(x)
    seabed_profile.append([x, 0.0, z_seabed])  # [x, y, z]

# CSV出力
with open("seabed_profile.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x[m]", "y[m]", "z[m]"])
    writer.writerows(seabed_profile)

print("海底形状 → seabed_profile.csv")

# ---- CSV ファイル書き出し ------------------------------------------------
for idx, rows in node_traj.items():
    fname = f"node_{idx}_traj.csv"
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time[s]", "x[m]", "y[m]", "z[m]"])
        writer.writerows(rows)
    print(f"ノード {idx} → {fname}")

print("\n=== シミュレーション完了 ===")
print(f"総計算時間: {T_END}s")
print(f"時間刻み: {DT}s")
print(f"総ステップ数: {step_count}")
print(f"ポリエステル部分: ノード{POLYESTER_START_NODE}〜{POLYESTER_END_NODE}")
print("出力ファイル:")
print("  - node_XX_traj.csv: 各ノードの軌跡")
print("  - segment_materials.csv: セグメント材料情報")
print("  - seabed_profile.csv: 海底形状")
