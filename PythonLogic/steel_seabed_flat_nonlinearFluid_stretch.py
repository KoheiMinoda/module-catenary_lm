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

# ============= 座標定義 =============
FP_COORDS = {"x": 5.2, "y": 0.0, "z": -70.0}
AP_COORDS = {"x": 853.87, "y": 0.0, "z": -320.0}
L_NATURAL = 902.2  # [m] 自然長（伸びなし）
XACC = 1e-4

MAXIT_P0_SOLVER = 100
MAXIT_RTSAFE = 200

h_span = abs(FP_COORDS["z"] - AP_COORDS["z"])
L_APFP = math.sqrt((FP_COORDS["x"] - AP_COORDS["x"])**2 +
                   (FP_COORDS["y"] - AP_COORDS["y"])**2)

_current_p0_for_funcd = 0.0

# ========= 強制振動の設定 ===========
AMP_FL = 4.0 
PERIOD_FL = 20.0 
OMEGA_FL = 2.0*math.pi / PERIOD_FL

# ============= 材料特性 =============
EA_CHAIN = 2.525e8  # チェーンの軸剛性 [N]
CA_CHAIN = 0.0  # チェーンの減衰係数
LineDiameter = 0.0945 # [m]
LineDryMass = 54.75 # [kg/m]
WaterRho = 1025.0 # [kg/m³]
# RHO_LINE = (LineDryMass - WaterRho * (math.pi * (LineDiameter*0.5)**2)) # [kg/m]
RHO_LINE = 47.5609 # [kg/m]
# RHO_LINE = 54.75 # [kg/m]
g = 9.80665

# ============= 高度化流体力パラメータ =============
RHO_WATER = 1025.0  # 海水密度 [kg/m³]
KINEMATIC_VISCOSITY = 1.05e-6  # 動粘性係数 [m²/s]

# 係留索の流体力係数（OrcaFlexに準拠）
CD_NORMAL_X = 2.6
CD_NORMAL_Y = 2.6
CD_NORMAL_Z = 1.4
CD_TANGENTIAL = 0.01

CM_NORMAL_X = 1.0
CM_NORMAL_Y = 1.0
CM_NORMAL_Z = 0.5
CM_TANGENTIAL = 0.0

# 海流設定：線形近似：OrcaFlex との比較では結局使わない
CURRENT_SURFACE = 0.5 # 表面流速 [m/s]
CURRENT_BOTTOM = 0.1 # 海底流速 [m/s]
CURRENT_DIRECTION = 0.0  # 流向 [deg]

# 波浪条件
WAVE_HEIGHT = 7.0  # 波高 [m]
WAVE_PERIOD = 8.0  # 波周期 [s]
WAVE_LENGTH = 99.5  # 波長 [m]
WAVE_DIRECTION = 180.0  # 波向 [deg]
WATER_DEPTH = 320.0  # 水深 [m]

# 海底面定義
SEABED_BASE_Z = AP_COORDS["z"]
V_SLIP_TOL = 1.0e-3
K_SEABED = 1.0e5
C_SEABED = 0.0
MU_STATIC = 0.6
MU_DYNAMIC = 0.5 

# 時間積分パラメータ
DT = 0.001
T_STATIC = 160.0  # 静的平衡時間
T_END = 500.0  # 総解析時間
RAYLEIGH_ALPHA = 0.1

# ============= カテナリー計算関数 =============
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

def solve_catenary(fp_coords, ap_coords, L_total):
    h_span = abs(fp_coords["z"] - ap_coords["z"])
    L_APFP = math.sqrt((fp_coords["x"] - ap_coords["x"])**2 +
                       (fp_coords["y"] - ap_coords["y"])**2)
    
    d_param = (L_APFP - (L_total - h_span)) / h_span
    l_param = L_total / h_span
    
    ans_x_final = 0.0
    if d_param <= 0:
        ans_x_final = 0.0
    elif l_param >= 1.0 and d_param >= (math.sqrt(l_param**2 - 1.0) - (l_param - 1.0)):
        return None  # 長さが不足
    else:
        rt_x1 = XACC
        rt_x2 = max(100.0, l_param * l_param * 2.0) 
        try:
            ans_x_final, _ = _rtsafe_solver(rt_x1, rt_x2, XACC, d_param, l_param, XACC)
        except:
            return None
    
    a_catenary = ans_x_final * h_span
    
    # カテナリー形状計算
    xf, zf = fp_coords["x"], fp_coords["z"]
    xa, za = ap_coords["x"], ap_coords["z"]
    Xs_rel = xa - xf 
    Zs_rel = za - zf
    
    if abs(a_catenary) < 1e-6:
        return None  # 垂直吊り下げは今回は考慮しない
    
    # グローバル座標系の計算
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
    
    # 弧長配列の計算
    num_internal_nodes = 19
    num_segments = num_internal_nodes + 1
    s_segment_len = L_total / num_segments 
    arc_lengths = [i * s_segment_len for i in range(num_segments + 1)]
    
    # セグメント長の計算
    segment_lengths = [s_segment_len] * num_segments
    
    return {
        'catenary_parameter': a_catenary,
        'x_gm': x_gm,
        'z_offset': z_offset,
        'x1_n': x1_n,
        'arc_lengths': arc_lengths,
        'segment_lengths': segment_lengths,
        'total_length': L_total
    }

def calculate_tension_distribution(catenary_result, rho_line, g):
    a = catenary_result['catenary_parameter']
    tensions = []
    
    for s in catenary_result['arc_lengths']:
        # カテナリー理論による張力
        T_horizontal = rho_line * g * a  # 水平張力成分
        xi = myasinh(s / a + math.sinh(catenary_result['x1_n']))
        T_total = T_horizontal * math.cosh(xi)
        tensions.append(T_total)
    
    return tensions

def calculate_catenary_with_stretch(fp_coords, ap_coords, L_natural, EA, rho_line, g, tolerance=1e-6, max_iter=20):
    print("\n--- ここで重力による伸びを考慮したカテナリー計算 ---")
    L_current = L_natural  # 初期推定
    
    for iteration in range(max_iter):
        print(f"反復 {iteration+1}: 現在の TOTAL LENGTH = {L_current:.3f} m")
        
        # 現在の長さでカテナリー計算
        catenary_result = solve_catenary(fp_coords, ap_coords, L_current)
        
        if catenary_result is None:
            break
            
        # 張力分布計算
        tensions = calculate_tension_distribution(catenary_result, rho_line, g)
        
        # 各セグメントの伸び計算
        total_stretch = 0.0
        for i, tension in enumerate(tensions[:-1]):  # セグメント数は節点数-1
            segment_natural_length = L_natural / len(tensions[:-1])  # 自然長ベース
            stretch = (tension / EA) * segment_natural_length
            total_stretch += stretch
        
        # 新しい総長
        L_new = L_natural + total_stretch
        
        print(f"  計算された STRETCH : {total_stretch:.6f} m ({total_stretch/L_natural*100:.3f}%)")
        print(f"  新しい TOTAL LENGTH: {L_new:.3f} m")
        
        # 収束判定
        if abs(L_new - L_current) < tolerance:
            print(f"[Complete] TOTAL LENGTH: {L_new:.3f} m")
            print(f"TOTAL STRETCH: {total_stretch:.6f} m ({total_stretch/L_natural*100:.3f}%)")
            return catenary_result, L_new, tensions
            
        L_current = L_new
    
    print(f"最大反復数({max_iter})に到達")
    return catenary_result, L_current, tensions

# ============= 伸びを考慮したカテナリー計算実行 =============
catenary_result, L_stretched, tension_distribution = calculate_catenary_with_stretch(
    FP_COORDS, AP_COORDS, L_NATURAL, EA_CHAIN, RHO_LINE, g
)

if catenary_result is None:
    raise ValueError("[Error] 伸びを考慮したカテナリー計算に失敗")

# 内部ノード位置の計算
print(f"\n--- 内部ノード位置計算 ---")
print(f"カテナリーパラメータ 'a': {catenary_result['catenary_parameter']:.3f} m")

internal_nodes_coords_final = []
num_internal_nodes = 19
num_segments = num_internal_nodes + 1
s_segment_len = L_stretched / num_segments

for i_node in range(1, num_internal_nodes + 1):
    s_k_arc_from_fp = i_node * s_segment_len 
    arg_for_asinh = s_k_arc_from_fp / catenary_result['catenary_parameter'] + math.sinh(catenary_result['x1_n'])
    xk_n = myasinh(arg_for_asinh)
    node_x_coord = catenary_result['x_gm'] + catenary_result['catenary_parameter'] * xk_n
    node_y_coord = 0.0 
    node_z_coord = catenary_result['catenary_parameter'] * math.cosh(xk_n) + catenary_result['z_offset']
    internal_nodes_coords_final.append({
        "id": i_node,
        "x": node_x_coord,
        "y": node_y_coord,
        "z": node_z_coord,
        "s_from_fp": s_k_arc_from_fp
    })

print(f"内部ノード数: {len(internal_nodes_coords_final)}")

# ============= 流体力計算関数群 =============

def get_current_velocity(z_coord, t):
    depth_ratio = abs(z_coord) / WATER_DEPTH
    depth_ratio = min(1.0, max(0.0, depth_ratio))
    
    current_magnitude = CURRENT_SURFACE * (1 - depth_ratio) + CURRENT_BOTTOM * depth_ratio
    
    current_dir_rad = math.radians(CURRENT_DIRECTION)
    u_current = current_magnitude * math.cos(current_dir_rad)
    v_current = current_magnitude * math.sin(current_dir_rad)
    w_current = 0.0
    
    return np.array([u_current, v_current, w_current])

def get_wave_velocity_acceleration(x, z, t):
    k = 2 * math.pi / WAVE_LENGTH
    omega = 2 * math.pi / WAVE_PERIOD
    amplitude = WAVE_HEIGHT / 2
    
    wave_dir_rad = math.radians(WAVE_DIRECTION)
    kx = k * math.cos(wave_dir_rad)
    ky = k * math.sin(wave_dir_rad)
    
    phase = kx * x + ky * 0.0 - omega * t
    
    depth_from_surface = abs(z)
    if depth_from_surface >= WATER_DEPTH:
        cosh_factor = math.cosh(k * WATER_DEPTH) / math.sinh(k * WATER_DEPTH)
        sinh_factor = 1.0 / math.sinh(k * WATER_DEPTH)
    else:
        cosh_factor = math.cosh(k * (WATER_DEPTH - depth_from_surface)) / math.sinh(k * WATER_DEPTH)
        sinh_factor = math.sinh(k * (WATER_DEPTH - depth_from_surface)) / math.sinh(k * WATER_DEPTH)
    
    u_wave = amplitude * omega * cosh_factor * math.cos(phase) * math.cos(wave_dir_rad)
    v_wave = amplitude * omega * cosh_factor * math.cos(phase) * math.sin(wave_dir_rad)
    w_wave = amplitude * omega * sinh_factor * math.sin(phase)
    
    du_dt = -amplitude * omega**2 * cosh_factor * math.sin(phase) * math.cos(wave_dir_rad)
    dv_dt = -amplitude * omega**2 * cosh_factor * math.sin(phase) * math.sin(wave_dir_rad)
    dw_dt = amplitude * omega**2 * sinh_factor * math.cos(phase)
    
    velocity = np.array([u_wave, v_wave, w_wave])
    acceleration = np.array([du_dt, dv_dt, dw_dt])
    
    return velocity, acceleration

def reynolds_drag_correction(velocity_magnitude, diameter, kinematic_viscosity, surface_roughness=0.001):

    if velocity_magnitude < 1e-6:
        return 1.0
    
    # レイノルズ数
    Re = velocity_magnitude * diameter / kinematic_viscosity
    
    # 滑らかな円柱の基本抗力係数
    if Re < 1e3:
        cd_smooth = 1.2
    elif Re <= 2e5:
        cd_smooth = 1.2 - 0.4 * math.log10(Re / 1e3)
    else:
        cd_smooth = 0.3
    
    # 表面粗さ修正
    roughness_ratio = surface_roughness / diameter
    roughness_factor = (1 + roughness_ratio / (1 + roughness_ratio))**0.25
    
    cd_corrected = cd_smooth * roughness_factor
    
    # チェーンの場合の追加修正
    chain_factor = 1.3  # 経験的係数
    
    return cd_corrected * chain_factor

def froude_krylov_force(seg_length, diameter, fluid_acceleration):
    volume = math.pi * (diameter/2)**2 * seg_length
    F_FK = RHO_WATER * volume * fluid_acceleration
    return F_FK

def added_mass_force(seg_vector, seg_length, diameter, structure_acc, fluid_acc, CM_normal, CM_tangential):
    
    volume = math.pi * (diameter/2)**2 * seg_length
    
    seg_length_calc = np.linalg.norm(seg_vector)
    if seg_length_calc < 1e-9:
        return np.zeros(3)
    t_vec = seg_vector / seg_length_calc
    
    acc_rel = structure_acc - fluid_acc
    
    acc_rel_t_scalar = np.dot(acc_rel, t_vec)
    acc_rel_t = acc_rel_t_scalar * t_vec
    acc_rel_n = acc_rel - acc_rel_t
    
    F_AM_n = -RHO_WATER * volume * CM_normal * acc_rel_n
    F_AM_t = -RHO_WATER * volume * CM_tangential * acc_rel_t
    
    return F_AM_n + F_AM_t

def drag_force_advanced(seg_vector, seg_length, diameter, relative_velocity, CD_normal, CD_tangential):

    seg_length_calc = np.linalg.norm(seg_vector)
    if seg_length_calc < 1e-9:
        return np.zeros(3)
    t_vec = seg_vector / seg_length_calc
    
    # 相対速度の分解
    u_rel_t_scalar = np.dot(relative_velocity, t_vec)
    u_rel_t = u_rel_t_scalar * t_vec
    u_rel_n = relative_velocity - u_rel_t
    
    # 速度の大きさ
    u_rel_n_mag = np.linalg.norm(u_rel_n)
    u_rel_t_mag = abs(u_rel_t_scalar)
    
    # 投影面積
    area_normal = diameter * seg_length
    area_tangential = math.pi * diameter * seg_length  # 接線方向（表面積）
    
    # 抗力計算
    if u_rel_n_mag > 1e-6:
        F_drag_n = 0.5 * RHO_WATER * CD_normal * area_normal * u_rel_n_mag * u_rel_n
    else:
        F_drag_n = np.zeros(3)
    
    if u_rel_t_mag > 1e-6:
        F_drag_t = 0.5 * RHO_WATER * CD_tangential * area_tangential * u_rel_t_mag * u_rel_t
    else:
        F_drag_t = np.zeros(3)
    
    return F_drag_n + F_drag_t

def calculate_advanced_morison_forces(nodes, segments, t):
    """
    F_total = F_froude_krylov + F_added_mass + F_drag_corrected
    """
    f_node = [np.zeros(3) for _ in nodes]
    
    for seg_idx, seg in enumerate(segments):
        i, j = seg["i"], seg["j"]
        
        # セグメント中点での計算
        pos_mid = 0.5 * (nodes[i]["pos"] + nodes[j]["pos"])
        vel_mid = 0.5 * (nodes[i]["vel"] + nodes[j]["vel"])
        
        # ノードに加速度が保存されていることを前提
        if "acc" in nodes[i] and "acc" in nodes[j]:
            acc_mid = 0.5 * (nodes[i]["acc"] + nodes[j]["acc"])
        else:
            acc_mid = np.zeros(3)  # 加速度が無い場合はゼロとする
        
        x_mid, y_mid, z_mid = pos_mid[0], pos_mid[1], pos_mid[2]
        
        # セグメントベクトル
        seg_vec = nodes[j]["pos"] - nodes[i]["pos"]
        seg_length = np.linalg.norm(seg_vec)
        if seg_length < 1e-9:
            continue
        
        # 流体速度・加速度の取得
        u_current = get_current_velocity(z_mid, t)
        u_wave, dudt_wave = get_wave_velocity_acceleration(x_mid, z_mid, t)
        
        # 総流体速度・加速度
        u_fluid_total = u_wave
        dudt_fluid_total = dudt_wave  # 海流の加速度は 0 と仮定
        
        # 相対速度・加速度
        u_rel = u_fluid_total - vel_mid
        dudt_rel = dudt_fluid_total - acc_mid
        
        # Froude-Krylov力（波浪慣性力）
        F_FK = froude_krylov_force(seg_length, LineDiameter, dudt_fluid_total)
        
        # 付加質量力（接線・法線分離）
        F_AM = added_mass_force(seg_vec, seg_length, LineDiameter,
                               acc_mid, dudt_fluid_total, 
                               (CM_NORMAL_X + CM_NORMAL_Y + CM_NORMAL_Z) / 3,  # 平均値
                               CM_TANGENTIAL)
        
        # 抗力（レイノルズ数修正付き）
        u_rel_mag = np.linalg.norm(u_rel)
        cd_correction = reynolds_drag_correction(u_rel_mag, LineDiameter, 
                                               KINEMATIC_VISCOSITY)
        
        # 平均抗力係数（方向別係数の平均）
        cd_avg = (CD_NORMAL_X + CD_NORMAL_Y + CD_NORMAL_Z) / 3
        
        F_drag = drag_force_advanced(seg_vec, seg_length, LineDiameter,
                                   u_rel, cd_avg * cd_correction, 
                                   CD_TANGENTIAL * cd_correction)
        
        # 総流体力
        F_total = F_FK + F_AM + F_drag
        
        # ノードに力を配分（半分ずつ）
        f_node[i] += 0.5 * F_total
        f_node[j] += 0.5 * F_total
    
    return f_node

# ============= ノード・セグメント生成（自然長ベース） =============
nodes_xyz0 = [FP_COORDS] + internal_nodes_coords_final + [AP_COORDS]
num_nodes = len(nodes_xyz0)
num_segs = num_nodes - 1

# 自然長ベースでのセグメント長計算
L_natural_segment = L_NATURAL / num_segs

segments = []
for k in range(num_segs):
    xi, xj = nodes_xyz0[k], nodes_xyz0[k+1]
    L_current = math.dist((xi['x'], xi['y'], xi['z']), (xj['x'], xj['y'], xj['z']))
    segments.append({
        "i": k, "j": k+1, 
        "L0": L_natural_segment,  # 自然長：strain はこれを基準とする
        "L_current": L_current,   # 現在長（初期位置での長さ）
        "EA": EA_CHAIN, "CA": CA_CHAIN,
        "mass": RHO_LINE * L_natural_segment,  # 質量は自然長ベース
        "material": "Chain"
    })

print(f"自然長セグメント長: {L_natural_segment:.3f} m")
print(f"初期状態平均セグメント長: {sum([seg['L_current'] for seg in segments])/len(segments):.3f} m")

# 質量配分（自然長ベース）
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
        "acc": np.zeros(3),  # 加速度を初期化
        "mass": m_node[idx]
    })

# ============= 力計算関数 =============
def axial_forces(nodes, segments):
    f_node = [np.zeros(3) for _ in nodes]
    tensions = [0.0] * len(segments)
    MAX_FORCE = 5.0e7

    for idx, seg in enumerate(segments):
        i, j = seg["i"], seg["j"]
        xi, xj = nodes[i]["pos"], nodes[j]["pos"]
        vi, vj = nodes[i]["vel"], nodes[j]["vel"]

        dx = xj - xi
        l_current = np.linalg.norm(dx)
        if l_current < 1e-9:
            continue
        t = dx / l_current

        # 自然長 L0 を基準としたひずみ計算
        strain = (l_current - seg["L0"]) / seg["L0"]
        Fel = seg["EA"] * strain
        vrel = np.dot(vj - vi, t)
        Fd = seg["CA"] * vrel
        Fax = Fel + Fd
        Fax = max(0.0, min(Fax, MAX_FORCE))
        tensions[idx] = Fax

        Fvec = Fax * t
        f_node[i] += Fvec
        f_node[j] += -Fvec
    return f_node, tensions

def get_seabed_z(x_coord):
    return SEABED_BASE_Z

def seabed_contact_forces(nodes):
    f_node = [np.zeros(3) for _ in nodes]

    for k, nd in enumerate(nodes):
        x, z = nd["pos"][0], nd["pos"][2]
        vz = nd["vel"][2]

        seabed_z_local = get_seabed_z(x)

        pen = seabed_z_local - z
        if pen <= 0.0:
            continue

        Fz_norm = K_SEABED*pen - C_SEABED*vz
        if Fz_norm < 0.0:
            Fz_norm = 0.0

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

def calculate_effective_mass(nodes, segments):
    effective_masses = []
    
    for k, node in enumerate(nodes):
        # 構造質量
        m_structure = node["mass"]
        
        # 付加質量の計算
        m_added = 0.0
        
        # このノードに接続するセグメントの付加質量
        for seg in segments:
            if seg["i"] == k or seg["j"] == k:
                # セグメント長
                seg_length = seg["L0"]  # または現在長
                
                # 体積
                volume = math.pi * (LineDiameter/2)**2 * seg_length
                
                # 付加質量（方向平均）
                CM_avg = (CM_NORMAL_X + CM_NORMAL_Y + CM_NORMAL_Z) / 3
                m_added_seg = RHO_WATER * volume * CM_avg
                
                # 半分をこのノードに配分
                m_added += 0.5 * m_added_seg
        
        # 有効質量
        m_effective = m_structure + m_added
        effective_masses.append(m_effective)
    
    return effective_masses

def compute_acc(nodes, segments, t):

    effective_masses = calculate_effective_mass(nodes, segments)

    f_axial, tensions = axial_forces(nodes, segments)
    f_seabed = seabed_contact_forces(nodes)
    f_fluid = calculate_advanced_morison_forces(nodes, segments, t)  # 流体力

    acc = []
    for k, node in enumerate(nodes):
        if node["mass"] == 0.0:
            acc.append(np.zeros(3))
            continue

        Fg = np.array([0.0, 0.0, -node["mass"]*g])
        F_rayleigh = np.zeros(3)

        F_tot = f_axial[k] + f_seabed[k] + f_fluid[k] + Fg + F_rayleigh

        acc_calc = F_tot / effective_masses[k]
        acc.append(acc_calc)
        
        # ノードに加速度を保存：次ステップの計算で使用
        node["acc"] = acc_calc.copy()
    
    return acc, tensions

# ============= 出力用データ設定 =============
OUTPUT_NODES = list(range(num_nodes))
node_traj = {idx: [] for idx in OUTPUT_NODES}
tension_data = []
fluid_force_data = []
convergence_data = []

# T=160 での静的平衡位置を保存するための変数
equilibrium_positions_t20 = None

# ============= 時間積分ループ =============
print("\n--- シミュレーション開始 ---")
print(f"Phase 1: 静的平衡解析（重力+高度化流体力） (0 - {T_STATIC}s)")
print(f"Phase 2: 強制振動解析（重力+高度化流体力+強制振動） ({T_STATIC} - {T_END}s)")
print(f"解析条件:")
print(f"  - 自然長: {L_NATURAL} m")
print(f"  - 伸び考慮後初期総長: {L_stretched:.3f} m")
print(f"  - 伸び率: {(L_stretched-L_NATURAL)/L_NATURAL*100:.3f}%")
print(f"  - 波高: {WAVE_HEIGHT} m, 周期: {WAVE_PERIOD} s")
print(f"  - 強制振動振幅: {AMP_FL} m, 周期: {PERIOD_FL} s")

t = 0.0
step_count = 0
output_interval = int(0.1 / DT)

# 収束判定用の変数
convergence_check_interval = int(1.0 / DT)  # 1秒ごとにチェック
convergence_threshold = 0.01  # 0.01m/s以下の変位速度で収束とみなす

while t <= T_END:
    
    # Phase 1: 静的平衡（重力 + 流体力, 強制振動なし）
    if t <= T_STATIC:
        phase = 1
        # フェアリーダーは初期位置に固定
        nodes[0]["pos"][:] = [FP_COORDS['x'], FP_COORDS['y'], FP_COORDS['z']]
        nodes[0]["vel"][:] = [0.0, 0.0, 0.0]
        nodes[0]["acc"][:] = [0.0, 0.0, 0.0]
        
        # 重力 + 流体力, 強制振動なし
        a_list, segment_tensions = compute_acc(nodes, segments, t)
        
        # 収束チェック
        if step_count % convergence_check_interval == 0 and step_count > 0:
            max_vel = 0.0
            for k in range(1, num_nodes - 1):  # 内部ノードのみ
                vel_mag = np.linalg.norm(nodes[k]["vel"])
                max_vel = max(max_vel, vel_mag)
            
            convergence_data.append([t, max_vel, phase])
            
            if max_vel < convergence_threshold and t > 10.0:
                print(f"静的平衡収束達成 t = {t:.1f}s (最大速度: {max_vel:.6f} m/s)")
        
        # T=160 で位置を保存
        if abs(t - T_STATIC) < DT/2:
            equilibrium_positions_t20 = [node["pos"].copy() for node in nodes]
            print(f"T={T_STATIC}sでの平衡位置を保存")
    
    # Phase 2: の開始
    else:
        phase = 2
        # フェアリーダー強制振動
        vibration_time = t - T_STATIC
        x_fl = FP_COORDS['x'] + AMP_FL * math.sin(OMEGA_FL * vibration_time)
        vx_fl = AMP_FL * OMEGA_FL * math.cos(OMEGA_FL * vibration_time)
        ax_fl = -AMP_FL * OMEGA_FL**2 * math.sin(OMEGA_FL * vibration_time)
        nodes[0]["pos"][:] = [x_fl, FP_COORDS['y'], FP_COORDS['z']]
        nodes[0]["vel"][:] = [vx_fl, 0.0, 0.0]
        nodes[0]["acc"][:] = [ax_fl, 0.0, 0.0]
        
        # 重力 + 流体力 + 強制振動
        a_list, segment_tensions = compute_acc(nodes, segments, t)

    # 速度・位置更新（内部ノードのみ）
    MAX_VELOCITY = 40.0
    
    for k in range(1, num_nodes - 1):
        # 速度更新
        nodes[k]["vel"] += a_list[k]*DT
        
        # 速度制限
        vel_mag = np.linalg.norm(nodes[k]["vel"])
        if vel_mag > MAX_VELOCITY:
            nodes[k]["vel"] = nodes[k]["vel"] * (MAX_VELOCITY / vel_mag)
        
        # 位置更新
        nodes[k]["pos"] += nodes[k]["vel"]*DT

    # データ記録
    if step_count % output_interval == 0:
        # ノード位置記録
        for idx in OUTPUT_NODES:
            p = nodes[idx]["pos"]
            v = nodes[idx]["vel"]
            a = nodes[idx]["acc"]
            
            # 平衡位置からの変位計算
            displacement_from_equilibrium = 0.0
            if equilibrium_positions_t20 is not None:
                displacement_from_equilibrium = np.linalg.norm(p - equilibrium_positions_t20[idx])
            
            node_traj[idx].append([t, p[0], p[1], p[2], 
                                 np.linalg.norm(v), np.linalg.norm(a), 
                                 phase, displacement_from_equilibrium])
        
        # 張力データ記録
        fairleader_tension = segment_tensions[0]
        anchor_tension = segment_tensions[-1]
        max_tension = max(segment_tensions)
        avg_tension = sum(segment_tensions) / len(segment_tensions)
        min_tension = min(segment_tensions)
        tension_data.append([t, fairleader_tension, anchor_tension, max_tension, 
                           avg_tension, min_tension, phase])
        
        # 流体力データ記録
        mid_node_idx = num_nodes // 2
        pos_mid = nodes[mid_node_idx]["pos"]
        vel_mid = nodes[mid_node_idx]["vel"]
        acc_mid = nodes[mid_node_idx]["acc"]
        
        u_current = get_current_velocity(pos_mid[2], t)
        u_wave, dudt_wave = get_wave_velocity_acceleration(pos_mid[0], pos_mid[2], t)
        u_fluid_total = u_current + u_wave
        
        # フェアリーダーの強制振動情報も記録
        fl_displacement = 0.0
        fl_velocity = 0.0
        fl_acceleration = 0.0
        if phase == 2:
            vibration_time = t - T_STATIC
            fl_displacement = AMP_FL * math.sin(OMEGA_FL * vibration_time)
            fl_velocity = AMP_FL * OMEGA_FL * math.cos(OMEGA_FL * vibration_time)
            fl_acceleration = -AMP_FL * OMEGA_FL**2 * math.sin(OMEGA_FL * vibration_time)
        
        # レイノルズ数計算
        u_rel_mag = np.linalg.norm(u_fluid_total - vel_mid)
        Re = u_rel_mag * LineDiameter / KINEMATIC_VISCOSITY if u_rel_mag > 1e-6 else 0.0
        cd_correction = reynolds_drag_correction(u_rel_mag, LineDiameter, KINEMATIC_VISCOSITY)
        
        fluid_force_data.append([
            t, phase,
            u_fluid_total[0], u_fluid_total[1], u_fluid_total[2],
            dudt_wave[0], dudt_wave[1], dudt_wave[2],
            u_rel_mag, Re, cd_correction,
            pos_mid[2], fl_displacement, fl_velocity, fl_acceleration,
            np.linalg.norm(acc_mid)
        ])
    
    t += DT
    step_count += 1
    
    # 進捗表示
    if step_count % (100 * output_interval) == 0:  # 10秒ごと
        if phase == 1:
            phase_name = "静的平衡（重力+高度化流体力）"
        else:
            phase_name = "強制振動（重力+高度化流体力+振動）"
        print(f"Time: {t:.1f}s / {T_END}s ({t/T_END*100:.1f}%) - {phase_name}")

print("--- シミュレーション完了 ---")

# ============= 解析結果の出力 =============

# セグメント材料情報
segment_info = []
for k, seg in enumerate(segments):
    initial_strain = (seg["L_current"] - seg["L0"]) / seg["L0"]
    segment_info.append([
        k, seg["i"], seg["j"], seg["material"], 
        seg["EA"], seg["CA"], 
        seg["L0"], seg["L_current"], initial_strain
    ])

with open("segment_materials_advanced.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["segment_id", "node_i", "node_j", "material", 
                     "EA[N]", "CA[Ns/m]", "L0_natural[m]", "L_initial[m]", "initial_strain[-]"])
    writer.writerows(segment_info)

# 海底形状データ
x_min = min(FP_COORDS["x"], AP_COORDS["x"]) - 100.0
x_max = max(FP_COORDS["x"], AP_COORDS["x"]) + 100.0
x_seabed = np.linspace(x_min, x_max, 200)

seabed_profile = []
for x in x_seabed:
    z_seabed = SEABED_BASE_Z
    seabed_profile.append([x, 0.0, z_seabed])

with open("seabed_profile_advanced.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x[m]", "y[m]", "z[m]"])
    writer.writerows(seabed_profile)

# 張力データ（詳細統計と phase 情報も追加）
with open("tension_data_advanced.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time[s]", "fairleader_tension[N]", "anchor_tension[N]", 
                     "max_tension[N]", "avg_tension[N]", "min_tension[N]", "phase"])
    writer.writerows(tension_data)

# 流体力データ
with open("advanced_fluid_conditions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "time[s]", "phase", 
        "fluid_vel_x[m/s]", "fluid_vel_y[m/s]", "fluid_vel_z[m/s]",
        "fluid_acc_x[m/s2]", "fluid_acc_y[m/s2]", "fluid_acc_z[m/s2]",
        "relative_velocity_magnitude[m/s]", "reynolds_number[-]", "cd_correction_factor[-]",
        "depth[m]", "fairleader_displacement[m]", "fairleader_velocity[m/s]", 
        "fairleader_acceleration[m/s2]", "structure_acceleration_magnitude[m/s2]"
    ])
    writer.writerows(fluid_force_data)

# 収束データ phase 情報付き
with open("convergence_data_advanced.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time[s]", "max_velocity[m/s]", "phase"])
    writer.writerows(convergence_data)

# T=160での平衡位置データ→強制振動開始時の初期位置
if equilibrium_positions_t20 is not None:
    equilibrium_data = []
    for idx, pos in enumerate(equilibrium_positions_t20):
        equilibrium_data.append([idx, pos[0], pos[1], pos[2]])
    
    with open("equilibrium_positions_t20_advanced.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x[m]", "y[m]", "z[m]"])
        writer.writerows(equilibrium_data)

# 流体力パラメータ記録
with open("advanced_fluid_parameters.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["parameter", "value", "unit", "description"])
    writer.writerows([
        ["RHO_WATER", RHO_WATER, "kg/m3", "海水密度"],
        ["KINEMATIC_VISCOSITY", KINEMATIC_VISCOSITY, "m2/s", "動粘性係数"],
        ["CD_NORMAL_X", CD_NORMAL_X, "-", "X方向抗力係数"],
        ["CD_NORMAL_Y", CD_NORMAL_Y, "-", "Y方向抗力係数"],
        ["CD_NORMAL_Z", CD_NORMAL_Z, "-", "Z方向抗力係数"],
        ["CD_TANGENTIAL", CD_TANGENTIAL, "-", "接線方向抗力係数"],
        ["CM_NORMAL_X", CM_NORMAL_X, "-", "X方向付加質量係数"],
        ["CM_NORMAL_Y", CM_NORMAL_Y, "-", "Y方向付加質量係数"],
        ["CM_NORMAL_Z", CM_NORMAL_Z, "-", "Z方向付加質量係数"],
        ["CM_TANGENTIAL", CM_TANGENTIAL, "-", "接線方向付加質量係数"],
        ["WAVE_HEIGHT", WAVE_HEIGHT, "m", "波高"],
        ["WAVE_PERIOD", WAVE_PERIOD, "s", "波周期"],
        ["WAVE_LENGTH", WAVE_LENGTH, "m", "波長"],
        ["LineDiameter", LineDiameter, "m", "係留索直径"],
        ["AMP_FL", AMP_FL, "m", "強制振動振幅"],
        ["PERIOD_FL", PERIOD_FL, "s", "強制振動周期"]
    ])

# カテナリー計算結果
catenary_info = [
    ["L_NATURAL", L_NATURAL, "m", "自然長"],
    ["L_STRETCHED", L_stretched, "m", "伸び考慮後総長"],
    ["STRETCH_RATIO", (L_stretched-L_NATURAL)/L_NATURAL*100, "%", "伸び率"],
    ["L_NATURAL_SEGMENT", L_natural_segment, "m", "自然長セグメント長"],
    ["INITIAL_AVG_SEGMENT", sum([seg['L_current'] for seg in segments])/len(segments), "m", "初期平均セグメント長"],
    ["CATENARY_PARAMETER", catenary_result['catenary_parameter'], "m", "カテナリーパラメータ"],
    ["T_STATIC", T_STATIC, "s", "静的平衡時間"],
    ["T_END", T_END, "s", "総解析時間"],
    ["EA_CHAIN", EA_CHAIN, "N", "軸剛性"],
    ["RHO_LINE", RHO_LINE, "kg/m", "係留索線密度"]
]

with open("catenary_analysis_advanced.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["parameter", "value", "unit", "description"])
    writer.writerows(catenary_info)

# シミュレーション設定記録
simulation_settings = [
    ["FLUID_MODEL", "Advanced Morison Equation", "-", "使用流体力モデル"],
    ["FROUDE_KRYLOV_FORCE", "Enabled", "-", "Froude-Krylov力"],
    ["ADDED_MASS_SEPARATION", "Enabled", "-", "付加質量力の接線・法線分離"],
    ["REYNOLDS_CORRECTION", "Enabled", "-", "レイノルズ数による抗力修正"],
    ["RELATIVE_MOTION_EFFECTS", "Enabled", "-", "相対運動効果"],
    ["PHASE_1_DESCRIPTION", "Static equilibrium with gravity and advanced fluid forces", "-", "Phase1の内容"],
    ["PHASE_2_DESCRIPTION", "Forced oscillation with gravity and advanced fluid forces", "-", "Phase2の内容"],
    ["T_STATIC", T_STATIC, "s", "静的平衡時間"],
    ["T_VIBRATION_START", T_STATIC, "s", "強制振動開始時刻"],
    ["T_END", T_END, "s", "総解析時間"],
    ["DT", DT, "s", "時間刻み"],
    ["CONVERGENCE_THRESHOLD", convergence_threshold, "m/s", "収束判定閾値"],
    ["MAX_VELOCITY", 20.0, "m/s", "速度制限値"],
    ["RAYLEIGH_ALPHA", RAYLEIGH_ALPHA, "-", "レーリー減衰係数（無効化中）"]
]

with open("advanced_simulation_settings.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["parameter", "value", "unit", "description"])
    writer.writerows(simulation_settings)

# ノード軌跡データ
for idx, rows in node_traj.items():
    fname = f"node_{idx}_traj_advanced.csv"
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time[s]", "x[m]", "y[m]", "z[m]", "velocity_magnitude[m/s]", 
                        "acceleration_magnitude[m/s2]", "phase", "displacement_from_t20_equilibrium[m]"])
        writer.writerows(rows)
    print(f"ノード {idx} → {fname}")

print("\n=== シミュレーション完了 ===")
print(f"自然長: {L_NATURAL} m")
print(f"伸び考慮後初期総長: {L_stretched:.3f} m")
print(f"伸び率: {(L_stretched-L_NATURAL)/L_NATURAL*100:.3f}%")
print(f"自然長ベースセグメント長: {L_natural_segment:.3f} m")
print(f"Phase 1 (静的平衡): 0 - {T_STATIC}s")
print(f"Phase 2 (強制振動): {T_STATIC} - {T_END}s")
print(f"総計算時間: {T_END:.1f}s")
print(f"時間刻み: {DT}s")
print(f"総ステップ数: {step_count}")

print("解析情報:")
print("  - advanced_fluid_parameters.csv: 高度化流体力パラメータ")
print("  - advanced_simulation_settings.csv: 高度化シミュレーション設定")
print("  - convergence_data_advanced.csv: 収束履歴")
print("  - equilibrium_positions_t20_advanced.csv: T=160 平衡位置")
print("  - catenary_analysis_advanced.csv: カテナリー解析結果")
