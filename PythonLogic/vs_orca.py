# シンプルな軸力と海底摩擦

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

# ============= Coordinate Definitions =============
FP_COORDS = {"x": 5.2, "y": 0.0, "z": -70.0}
AP_COORDS = {"x": 853.87, "y": 0.0, "z": -320.0}
# AP_COORDS = {"x": 200.0, "y": 0.0, "z": -320.0}
L_NATURAL = 902.2  # [m] Natural length: no stretch
# L_NATURAL = 320.0
XACC = 1e-4

MAXIT_P0_SOLVER = 100
MAXIT_RTSAFE = 200

h_span = abs(FP_COORDS["z"] - AP_COORDS["z"])
L_APFP = math.sqrt((FP_COORDS["x"] - AP_COORDS["x"])**2 +
                   (FP_COORDS["y"] - AP_COORDS["y"])**2)

_current_p0_for_funcd = 0.0

# ========= Forced Vibration Settings ===========
AMP_FL = 2.0 
PERIOD_FL = 20.0 
OMEGA_FL = 2.0*math.pi / PERIOD_FL

# ============= Material Properties =============
EA_CHAIN = 2.525e8  # Chain axial stiffness [N]
CA_CHAIN = 0.0  # Chain damping coefficient
LineDiameter = 0.0945 # [m]
LineDryMass = 54.75 # [kg/m]
WaterRho = 1025.0 # [kg/m³]
RHO_LINE = 47.5609 # [kg/m]
g = 9.80665
CONTACT_DIAMETER = 0.01 # [m]
STRUCT_DAMP_RATIO = 0.000

# ============= Fluid Force Parameters =============
RHO_WATER = 1025.0  # Seawater density [kg/m³]
KINEMATIC_VISCOSITY = 1.35e-6  # Kinematic viscosity [m²/s]

# Mooring line fluid force coefficients: Based on OrcaFlex
CD_NORMAL = 2.6
CD_AXIAL = 1.4
CLIFT = 0.0
DIAM_DRAG_NORMAL = 0.05
DIAM_DRAG_AXIAL  = 0.01592

CM_NORMAL = 1.0
CM_TANGENTIAL = 0.5 

# Current settings: Linear approximation : No use for vs_OrcaFlex
CURRENT_SURFACE = 0.5 # Surface current velocity [m/s]
CURRENT_BOTTOM = 0.1 # Bottom current velocity [m/s]
CURRENT_DIRECTION = 0.0  # Current direction [deg]

# Wave conditions
WAVE_HEIGHT = 4.0  # Wave height [m]
WAVE_PERIOD = 8.0  # Wave period [s]
WAVE_LENGTH = (g*WAVE_PERIOD**2) / (2*math.pi)  # Wave length [m]
WAVE_DIRECTION = 0.0  # Wave direction [deg] Checked
WATER_DEPTH = 320.0  # Water depth [m]
P_0 = 101325.0

wave_components = [{
    'height': WAVE_HEIGHT,
    'period': WAVE_PERIOD,
    'phase': 0.0,
    'direction': WAVE_DIRECTION
}]

# Seabed definition
SEABED_BASE_Z = AP_COORDS["z"]
K_SEABED_NORMAL = 1.0e5
MU_LATERAL = 0.0
MU_AXIAL_STATIC = 0.0
V_SLIP_TOL_LATERAL = 1.0e-5
SMOOTH_CLEARANCE = 0.01
SMOOTH_EPS = 0.01

# ====== Time Integration Parameters ======
DT = 0.001
T_STATIC = 20.0  # Static equilibrium time
T_END = 500.0  # Total analysis time
RAYLEIGH_ALPHA = 0.07854  # 質量比例減衰係数 [1/s] (低周波減衰)
RAYLEIGH_BETA = 0.12732  # 剛性比例減衰係数 [s] (高周波減衰)

# ============= Catenary Calculation Functions =============
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

# Solve catenary equations using pure catenary theory (no stretch consideration)
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
        return None
    else:
        rt_x1 = XACC
        rt_x2 = max(100.0, l_param * l_param * 2.0) 
        try:
            ans_x_final, _ = _rtsafe_solver(rt_x1, rt_x2, XACC, d_param, l_param, XACC)
        except:
            return None
    
    a_catenary = ans_x_final * h_span
    
    # Catenary shape calculation
    xf, zf = fp_coords["x"], fp_coords["z"]
    xa, za = ap_coords["x"], ap_coords["z"]
    Xs_rel = xa - xf 
    Zs_rel = za - zf
    
    if abs(a_catenary) < 1e-6:
        return None  # Vertical hanging case not considered
    
    # Global coordinate system calculation
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
    
    # Arc length array calculation
    num_internal_nodes = 19
    num_segments = num_internal_nodes + 1
    s_segment_len = L_total / num_segments 
    arc_lengths = [i * s_segment_len for i in range(num_segments + 1)]
    
    # Segment length calculation
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
    # Calculate tension distribution using catenary theory
    a = catenary_result['catenary_parameter']
    tensions = []
    
    for s in catenary_result['arc_lengths']:
        # Tension from catenary theory
        T_horizontal = rho_line * g * a  # Horizontal tension component
        xi = myasinh(s / a + math.sinh(catenary_result['x1_n']))
        T_total = T_horizontal * math.cosh(xi)
        tensions.append(T_total)
    
    return tensions

# ============= カテナリー理論 =============

catenary_result = solve_catenary(FP_COORDS, AP_COORDS, L_NATURAL)

if catenary_result is None:
    raise ValueError("[Error] Pure catenary calculation failed")

tension_distribution = calculate_tension_distribution(catenary_result, RHO_LINE, g)

internal_nodes_coords_final = []
num_internal_nodes = 19
num_segments = num_internal_nodes + 1
s_segment_len = L_NATURAL / num_segments

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

print(f"[Complete] Number of internal nodes: {len(internal_nodes_coords_final)}")

# ============= ノード，セグメントの設定 =============
nodes_xyz0 = [FP_COORDS] + internal_nodes_coords_final + [AP_COORDS]
num_nodes = len(nodes_xyz0)
num_segs = num_nodes - 1

L_natural_segment = L_NATURAL / num_segs

segments = []
for k in range(num_segs):
    xi, xj = nodes_xyz0[k], nodes_xyz0[k+1]
    L_current = math.dist((xi['x'], xi['y'], xi['z']), (xj['x'], xj['y'], xj['z']))
    segments.append({
        "i": k, "j": k+1, 
        "L0": L_natural_segment,
        "L_current": L_current,
        "EA": EA_CHAIN, "CA": CA_CHAIN,
        "mass": RHO_LINE * L_natural_segment,
        "material": "Chain"
    })

print(f"[Complete] Natural segment length: {L_natural_segment:.3f} m")
print(f"[Complete] Initial state average segment length: {sum([seg['L_current'] for seg in segments])/len(segments):.3f} m")

m_node = [0.0]*num_nodes
for seg in segments:
    m_half = 0.5*seg["mass"]
    m_node[seg["i"]] += m_half
    m_node[seg["j"]] += m_half

"""
if len(segments) > 0:
    # フェアリーダー側
    m_node[0] += 0.5 * segments[0]["mass"]
    # アンカー側  
    m_node[-1] += 0.5 * segments[-1]["mass"]
"""
    
nodes = []
for idx, coord in enumerate(nodes_xyz0):
    nodes.append({
        "pos": np.array([coord['x'], coord['y'], coord['z']], dtype=float),
        "vel": np.zeros(3),
        "acc": np.zeros(3),
        "mass": m_node[idx]
    })

# =============================================================
# ============= Fluid Force Calculation Functions =============
# =============================================================


# checked
def get_current_velocity(z_coord, t):
    depth_ratio = abs(z_coord) / WATER_DEPTH
    depth_ratio = min(1.0, max(0.0, depth_ratio))
    
    current_magnitude = CURRENT_SURFACE * (1 - depth_ratio) + CURRENT_BOTTOM * depth_ratio
    
    current_dir_rad = math.radians(CURRENT_DIRECTION)
    u_current = current_magnitude * math.cos(current_dir_rad)
    v_current = current_magnitude * math.sin(current_dir_rad)
    w_current = 0.0
    
    return np.array([u_current, v_current, w_current])

# checked
def solve_dispersion(omega, depth, g=9.80665, tol=1e-6, maxit=50):
    k = omega**2 / g
    for _ in range(maxit):
        tanh_kh = math.tanh(k * depth)
        f = g * k * tanh_kh - omega**2
        df = g * tanh_kh + g * k * depth * (1 - tanh_kh**2)
        dk = -f / df
        k += dk
        if abs(dk) < tol:
            break
    return k

# checked
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

# checked
def compute_rayleigh_damping_forces(nodes, segments):
    n_nodes = len(nodes)
    f_damp = [np.zeros(3) for _ in range(n_nodes)]

    # 質量
    for k, node in enumerate(nodes):
        f_damp[k] += - RAYLEIGH_ALPHA * node["mass"] * node["vel"]

    # 剛性
    for seg in segments:
        i, j = seg["i"], seg["j"]
        pos_i, pos_j = nodes[i]["pos"], nodes[j]["pos"]
        vel_i, vel_j = nodes[i]["vel"], nodes[j]["vel"]

        seg_vec = pos_j - pos_i
        seg_length = np.linalg.norm(seg_vec)
        if seg_length < 1e-9:
            continue
        t_vec = seg_vec / seg_length

        rel_vel_t = np.dot(vel_j - vel_i, t_vec)

        # 要素剛性 K = EA / L0
        K_el = seg["EA"] / seg["L0"]

        # 剛性比例減衰力
        Fd_scalar = RAYLEIGH_BETA * K_el * rel_vel_t

        # ベクトル化してノードに分配
        Fd_vec = Fd_scalar * t_vec
        f_damp[i] +=  Fd_vec
        f_damp[j] += -Fd_vec

    return f_damp

def froude_krylov_force(seg_vector, diameter, fluid_acceleration):
    seg_length = np.linalg.norm(seg_vector)
    if seg_length < 1e-12:
        return np.zeros(3)
    volume = math.pi * (diameter/2)**2 * seg_length
    F_FK = RHO_WATER * volume * np.asarray(fluid_acceleration, float)
    return F_FK

# checked
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

# checked
def drag_force_advanced(seg_vector, seg_length, diam_normal, diam_axial, rel_vel, cd_normal, cd_axial):
    seg_length = np.linalg.norm(seg_vector)
    if seg_length < 1e-9:
        return np.zeros(3)
    t_vec = seg_vector / seg_length
    
    vel_t_scalar = np.dot(rel_vel, t_vec)
    vel_t = vel_t_scalar * t_vec
    vel_n = rel_vel - vel_t
    mag_t = abs(vel_t_scalar)
    mag_n = np.linalg.norm(vel_n)

    area_n = diam_normal * seg_length
    area_t = math.pi * diam_axial * seg_length
    
    # 法線方向
    if mag_n > 1e-6:
        Fd_n = 0.5 * RHO_WATER * cd_normal * area_n * mag_n * vel_n
    else:
        Fd_n = np.zeros(3)
    
    if mag_t > 1e-6:
        Fd_t = 0.5 * RHO_WATER * cd_axial * area_t * mag_t * vel_t
    else:
        Fd_t = np.zeros(3)
    
    return Fd_n + Fd_t

def calculate_advanced_morison_forces(nodes, segments, t):
    f_node = [np.zeros(3) for _ in nodes]
    
    for seg in segments:
        i, j = seg["i"], seg["j"]
        pos_i, pos_j = nodes[i]["pos"], nodes[j]["pos"]
        vel_i, vel_j = nodes[i]["vel"], nodes[j]["vel"]
        
        pos_mid = 0.5 * (nodes[i]["pos"] + nodes[j]["pos"])
        vel_mid = 0.5 * (nodes[i]["vel"] + nodes[j]["vel"])
        acc_mid = 0.5 * (nodes[i]["acc"] + nodes[j]["acc"])

        seg_vec = pos_j - pos_i
        seg_length = np.linalg.norm(seg_vec)
        if seg_length < 1e-9:
            continue
        
        u_curr = get_current_velocity(pos_mid[2], t)
        u_wave, a_wave = get_wave_velocity_acceleration(pos_mid[0], pos_mid[2], t)
        
        u_fluid = u_curr + u_wave
        a_fluid = a_wave
        
        F_FK = froude_krylov_force(seg_vec, LineDiameter, a_fluid)
        
        F_AM = added_mass_force(
            seg_vec, 
            seg_length, 
            LineDiameter,
            acc_mid, 
            a_fluid, 
            CM_NORMAL,  
            CM_TANGENTIAL
        )
        
        u_rel = u_fluid - vel_mid
        cd_n = CD_NORMAL
        cd_t = CD_AXIAL
        F_drag = drag_force_advanced(
            seg_vec, 
            seg_length, 
            DIAM_DRAG_NORMAL,
            DIAM_DRAG_AXIAL,
            u_rel, 
            cd_n, 
            cd_t
        )
        
        F_total = F_FK + F_AM + F_drag
        f_node[i] += 0.5 * F_total
        f_node[j] += 0.5 * F_total
    
    return f_node


# ======================================================
# ============= 軸力 =============
# ====================================================


# checked
def axial_forces(nodes, segments):

    f_node = [np.zeros(3) for _ in nodes]
    tensions = [0.0] * len(segments)
    MAX_FORCE = 100.0e9

    for idx, seg in enumerate(segments):
        i, j = seg["i"], seg["j"]
        pos_i, pos_j = nodes[i]["pos"], nodes[j]["pos"]
        vel_i, vel_j = nodes[i]["vel"], nodes[j]["vel"]

        seg_vec = pos_j - pos_i
        seg_length = np.linalg.norm(seg_vec)
        if seg_length < 1e-9:
            continue
        t_vec = seg_vec / seg_length

        strain = (seg_length - seg["L0"]) / seg["L0"]
        Fel = seg["EA"] * strain
        rel_vel = vel_j - vel_i
        dl_dt = np.dot(seg_vec, rel_vel) / seg_length
        strain_rate = dl_dt / seg["L0"]

        if strain > 0:  # 引張のみ
            F_elastic = seg["EA"] * strain
        else:
            F_elastic = 0.0 
        
        F_strain_rate_damping = seg["EA"] * STRUCT_DAMP_RATIO * strain_rate

        total_tension = F_elastic + F_strain_rate_damping

        total_tension = max(0.0, min(total_tension, MAX_FORCE))
        tensions[idx] = total_tension

        tension_vector = total_tension * t_vec
        f_node[i] += tension_vector
        f_node[j] += - tension_vector
    return f_node, tensions


# =================================
# ========= 海底接触力 ==============
# =================================


def get_seabed_z(x_coord):
    return SEABED_BASE_Z

def get_seabed_penetration(node_pos, diameter):
    x, z = node_pos[0], node_pos[2]
    seabed_z_local = get_seabed_z(x)
    penetration = seabed_z_local - z + CONTACT_DIAMETER
    return max(0.0, penetration)

def is_segment_on_seabed(nodes, seg, diameter):
    i, j = seg["i"], seg["j"]
    pen_i = get_seabed_penetration(nodes[i]["pos"], diameter)
    pen_j = get_seabed_penetration(nodes[j]["pos"], diameter)
    
    return pen_i > 0.0 and pen_j > 0.0

def smooth_seabed_force(raw_penetration, K, seg_length, radius, clearance=SMOOTH_CLEARANCE, eps=SMOOTH_EPS):
    p_eff = raw_penetration + clearance + CONTACT_DIAMETER
    if p_eff <= 0.0:
        return 0.0
    # セグメントあたりの定数化された剛性
    K_eff = K * seg_length * radius * p_eff
    # sqrt 関数で p→0 付近の勾配を 0 に近づける
    return K_eff * (math.sqrt(p_eff**2 + eps**2) - eps)

# checked
def calculate_segment_seabed_forces(nodes, seg, diameter):
    i, j = seg["i"], seg["j"]
    pos_i, pos_j = nodes[i]["pos"], nodes[j]["pos"]
    vel_i, vel_j = nodes[i]["vel"], nodes[j]["vel"]
    
    seg_vec = pos_j - pos_i
    seg_length = np.linalg.norm(seg_vec)
    if seg_length < 1e-9:
        return np.zeros(3), np.zeros(3)
    radius = diameter / 2.0
    
    seabed_z = get_seabed_z(0)
    raw_i = seabed_z - pos_i[2]
    raw_j = seabed_z - pos_j[2]
    raw_avg = 0.5 * (raw_i + raw_j)
    
    F_normal = smooth_seabed_force(raw_avg, K_SEABED_NORMAL, seg_length, radius)
    if F_normal < 0.0:
        F_normal = 0.0
    
    pos_mid = 0.5*(pos_i + pos_j)
    vel_mid = 0.5*(vel_i + vel_j)
    vel_lat = vel_mid[:2]
    mag_lat = np.linalg.norm(vel_lat)
    if mag_lat > V_SLIP_TOL_LATERAL:
        F_fric_lat = - MU_LATERAL * F_normal * (vel_lat / mag_lat)
    else:
        F_fric_lat = np.zeros(2)

    # ベクトル合成
    F_norm_vec   = np.array([0.0, 0.0, F_normal])
    F_fric_vec   = np.array([F_fric_lat[0], F_fric_lat[1], 0.0])
    F_total      = F_norm_vec + F_fric_vec
    
    # セグメント両端に均等配分
    f_i = 0.5 * F_total
    f_j = 0.5 * F_total
    
    return f_i, f_j

def seabed_contact_forces_segment_based(nodes, segments, t):
    f_node = [np.zeros(3) for _ in nodes]
    
    for seg in segments:
        if is_segment_on_seabed(nodes, seg, LineDiameter):
            f_i, f_j = calculate_segment_seabed_forces(nodes, seg, LineDiameter)
            
            f_node[seg["i"]] += f_i
            f_node[seg["j"]] += f_j
    
    return f_node

# checked
def calculate_effective_mass(nodes, segments):

    effective_masses = []
    
    for k, node in enumerate(nodes):
        m_structure = node["mass"]
        m_added = 0.0
        
        for seg in segments:
            if seg["i"] == k or seg["j"] == k:
                if "L_current" in seg:
                    seg_length = seg["L_current"]
                else:
                    seg_length = seg["L0"]
                
                volume = math.pi * (LineDiameter/2)**2 * seg_length
                
                pos_i = np.array(nodes[seg["i"]]["pos"])
                pos_j = np.array(nodes[seg["j"]]["pos"])
                diff = pos_j - pos_i
                length = np.linalg.norm(diff)
                if length < 1e-9:
                    continue
                t_vec = diff / length

                tz = abs(t_vec[2])
                CM_eff = CM_TANGENTIAL * tz**2 + CM_NORMAL * (1 - tz**2)

                m_added_seg = RHO_WATER * volume * CM_eff
                m_added += 0.5 * m_added_seg
        
        effective_masses.append(m_structure + m_added)
    
    return effective_masses

# checked
def compute_acc(nodes, segments, t):
    effective_masses = calculate_effective_mass(nodes, segments)

    f_axial, tensions = axial_forces(nodes, segments)
    f_seabed = seabed_contact_forces_segment_based(nodes, segments, t)
    f_fluid = calculate_advanced_morison_forces(nodes, segments, t)
    # f_rayleigh = compute_rayleigh_damping_forces(nodes, segments)

    acc = []
    for k, node in enumerate(nodes):

        Fg = np.array([0.0, 0.0, -node["mass"]*g])
        # F_tot = f_axial[k] + f_seabed[k] + f_fluid[k] + f_rayleigh[k] + Fg
        F_tot = f_axial[k] + f_seabed[k] + f_fluid[k] + Fg

        m_eff = effective_masses[k]
        a_k = F_tot / m_eff
        node["acc"] = a_k.copy()
        acc.append(a_k)
    
    return acc, tensions

# ============= Output Data Setup =============
OUTPUT_NODES = list(range(num_nodes))
node_traj = {idx: [] for idx in OUTPUT_NODES}
tension_data = []
fluid_force_data = []
convergence_data = []

equilibrium_positions_t20 = None

# ============= Time Integration Loop =============
print("\n--- Simulation Start ---")
print(f"Phase 1: Static equilibrium analysis (gravity + advanced fluid forces) (0 - {T_STATIC}s)")
print(f"Phase 2: Forced vibration analysis (gravity + advanced fluid forces + forced vibration) ({T_STATIC} - {T_END}s)")
print(f"Analysis conditions:")
print(f"  - Natural length: {L_NATURAL} m")
print(f"  - Initial catenary total length: {L_NATURAL:.3f} m (pure catenary theory)")
print(f"  - Wave height: {WAVE_HEIGHT} m, period: {WAVE_PERIOD} s")
print(f"  - Forced vibration amplitude: {AMP_FL} m, period: {PERIOD_FL} s")

t = 0.0
step_count = 0
output_interval = int(0.1 / DT)

convergence_check_interval = int(1.0 / DT)  # Check every 1 second
convergence_threshold = 0.01  # Consider converged if displacement velocity is below 0.01m/s

while t <= T_END:
    
    # Phase 1: Static equilibrium (gravity + fluid forces, no forced vibration)
    if t <= T_STATIC:
        phase = 1
        nodes[0]["pos"][:] = [FP_COORDS['x'], FP_COORDS['y'], FP_COORDS['z']]
        nodes[0]["vel"][:] = [0.0, 0.0, 0.0]
        nodes[0]["acc"][:] = [0.0, 0.0, 0.0]
        
        a_list, segment_tensions = compute_acc(nodes, segments, t)
        
        # Convergence check
        if step_count % convergence_check_interval == 0 and step_count > 0:
            max_vel = 0.0
            for k in range(1, num_nodes - 1):  # Internal nodes only
                vel_mag = np.linalg.norm(nodes[k]["vel"])
                max_vel = max(max_vel, vel_mag)
            
            convergence_data.append([t, max_vel, phase])
            
            if max_vel < convergence_threshold and t > 10.0:
                print(f"[Complete] Static equilibrium convergence achieved at t = {t:.1f}s (max velocity: {max_vel:.6f} m/s)")
        
        # Save positions at T=20
        if abs(t - T_STATIC) < DT/2:
            equilibrium_positions_t20 = [node["pos"].copy() for node in nodes]
            print(f"[Complete] Equilibrium positions saved at T={T_STATIC}s")
    
    # Phase 2: Start of dynamic analysis
    else:
        phase = 2
        vibration_time = t - T_STATIC
        x_fl = FP_COORDS['x'] + AMP_FL * math.sin(OMEGA_FL * vibration_time)
        vx_fl = AMP_FL * OMEGA_FL * math.cos(OMEGA_FL * vibration_time)
        ax_fl = -AMP_FL * OMEGA_FL**2 * math.sin(OMEGA_FL * vibration_time)
        nodes[0]["pos"][:] = [x_fl, FP_COORDS['y'], FP_COORDS['z']]
        nodes[0]["vel"][:] = [vx_fl, 0.0, 0.0]
        nodes[0]["acc"][:] = [ax_fl, 0.0, 0.0]
        
        a_list, segment_tensions = compute_acc(nodes, segments, t)

    MAX_VELOCITY = 1000.0
    
    for k in range(1, num_nodes - 1):
        nodes[k]["vel"] += a_list[k]*DT
        
        vel_mag = np.linalg.norm(nodes[k]["vel"])
        if vel_mag > MAX_VELOCITY:
            nodes[k]["vel"] = nodes[k]["vel"] * (MAX_VELOCITY / vel_mag)
        
        nodes[k]["pos"] += nodes[k]["vel"]*DT

    if step_count % output_interval == 0:
        for idx in OUTPUT_NODES:
            p = nodes[idx]["pos"]
            v = nodes[idx]["vel"]
            a = nodes[idx]["acc"]
            
            displacement_from_equilibrium = 0.0
            if equilibrium_positions_t20 is not None:
                displacement_from_equilibrium = np.linalg.norm(p - equilibrium_positions_t20[idx])
            
            node_traj[idx].append([t, p[0], p[1], p[2], 
                                 np.linalg.norm(v), np.linalg.norm(a), 
                                 phase, displacement_from_equilibrium])
        
        fairleader_tension = segment_tensions[0]
        anchor_tension = segment_tensions[-1]
        max_tension = max(segment_tensions)
        avg_tension = sum(segment_tensions) / len(segment_tensions)
        min_tension = min(segment_tensions)
        tension_data.append([t, fairleader_tension, anchor_tension, max_tension, 
                           avg_tension, min_tension, phase])
        
        mid_node_idx = num_nodes // 2
        pos_mid = nodes[mid_node_idx]["pos"]
        vel_mid = nodes[mid_node_idx]["vel"]
        acc_mid = nodes[mid_node_idx]["acc"]
        
        u_current = get_current_velocity(pos_mid[2], t)
        u_wave, dudt_wave = get_wave_velocity_acceleration(pos_mid[0], pos_mid[2], t)
        u_fluid_total = u_current + u_wave
        
        fl_displacement = 0.0
        fl_velocity = 0.0
        fl_acceleration = 0.0
        if phase == 2:
            vibration_time = t - T_STATIC
            fl_displacement = AMP_FL * math.sin(OMEGA_FL * vibration_time)
            fl_velocity = AMP_FL * OMEGA_FL * math.cos(OMEGA_FL * vibration_time)
            fl_acceleration = -AMP_FL * OMEGA_FL**2 * math.sin(OMEGA_FL * vibration_time)
        
        u_rel_mag = np.linalg.norm(u_fluid_total - vel_mid)
        Re = u_rel_mag * LineDiameter / KINEMATIC_VISCOSITY if u_rel_mag > 1e-6 else 0.0
        
        fluid_force_data.append([
            t, phase,
            u_fluid_total[0], u_fluid_total[1], u_fluid_total[2],
            dudt_wave[0], dudt_wave[1], dudt_wave[2],
            u_rel_mag, Re,
            pos_mid[2], fl_displacement, fl_velocity, fl_acceleration,
            np.linalg.norm(acc_mid)
        ])
    
    t += DT
    step_count += 1
    
    if step_count % (100 * output_interval) == 0:  # Every 10 seconds
        if phase == 1:
            phase_name = "Static equilibrium"
        else:
            phase_name = "Forced vibration"
        print(f"Time: {t:.1f}s / {T_END}s ({t/T_END*100:.1f}%) - {phase_name}")

print("[Complete] --- Simulation Complete ---")

# ============= Analysis Results Output =============

segment_info = []
for k, seg in enumerate(segments):
    initial_strain = (seg["L_current"] - seg["L0"]) / seg["L0"]
    segment_info.append([
        k, seg["i"], seg["j"], seg["material"], 
        seg["EA"], seg["CA"], 
        seg["L0"], seg["L_current"], initial_strain
    ])

with open("segment_materials.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["segment_id", "node_i", "node_j", "material", 
                     "EA[N]", "CA[Ns/m]", "L0_natural[m]", "L_initial[m]", "initial_strain[-]"])
    writer.writerows(segment_info)

# Seabed profile data
x_min = min(FP_COORDS["x"], AP_COORDS["x"]) - 100.0
x_max = max(FP_COORDS["x"], AP_COORDS["x"]) + 100.0
x_seabed = np.linspace(x_min, x_max, 200)

seabed_profile = []
for x in x_seabed:
    z_seabed = SEABED_BASE_Z
    seabed_profile.append([x, 0.0, z_seabed])

with open("seabed_profile.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x[m]", "y[m]", "z[m]"])
    writer.writerows(seabed_profile)

# Tension data (detailed statistics with phase information)
with open("tension_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time[s]", "fairleader_tension[N]", "anchor_tension[N]", 
                     "max_tension[N]", "avg_tension[N]", "min_tension[N]", "phase"])
    writer.writerows(tension_data)

# Fluid force data
with open("fluid_conditions.csv", "w", newline="") as f:
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

# Convergence data with phase information
with open("convergence_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time[s]", "max_velocity[m/s]", "phase"])
    writer.writerows(convergence_data)

# Equilibrium positions at T=20 → Initial positions at start of forced vibration
if equilibrium_positions_t20 is not None:
    equilibrium_data = []
    for idx, pos in enumerate(equilibrium_positions_t20):
        equilibrium_data.append([idx, pos[0], pos[1], pos[2]])
    
    with open("equilibrium_positions_t20.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x[m]", "y[m]", "z[m]"])
        writer.writerows(equilibrium_data)

# Catenary calculation results
catenary_info = [
    ["L_NATURAL", L_NATURAL, "m", "Natural length"],
    ["CATENARY_TOTAL_LENGTH", L_NATURAL, "m", "Total length used in catenary calculation"],
    ["L_NATURAL_SEGMENT", L_natural_segment, "m", "Natural segment length"],
    ["INITIAL_AVG_SEGMENT", sum([seg['L_current'] for seg in segments])/len(segments), "m", "Initial average segment length"],
    ["CATENARY_PARAMETER", catenary_result['catenary_parameter'], "m", "Catenary parameter"],
    ["T_STATIC", T_STATIC, "s", "Static equilibrium time"],
    ["T_END", T_END, "s", "Total analysis time"],
    ["EA_CHAIN", EA_CHAIN, "N", "Axial stiffness"],
    ["RHO_LINE", RHO_LINE, "kg/m", "Mooring line density"]
]

with open("catenary_analysis.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["parameter", "value", "unit", "description"])
    writer.writerows(catenary_info)

# Simulation settings record
simulation_settings = [
    ["CATENARY_METHOD", "Pure Catenary Theory", "-", "Method used for initial geometry"],
    ["STRETCH_CONSIDERATION", "Disabled", "-", "Stretch consideration in initial geometry"],
    ["FLUID_MODEL", "Advanced Morison Equation", "-", "Fluid force model used"],
    ["FROUDE_KRYLOV_FORCE", "Enabled", "-", "Froude-Krylov force"],
    ["ADDED_MASS_SEPARATION", "Enabled", "-", "Added mass force tangential/normal separation"],
    ["REYNOLDS_CORRECTION", "Enabled", "-", "Reynolds number based drag correction"],
    ["RELATIVE_MOTION_EFFECTS", "Enabled", "-", "Relative motion effects"],
    ["PHASE_1_DESCRIPTION", "Static equilibrium with gravity and advanced fluid forces", "-", "Phase 1 content"],
    ["PHASE_2_DESCRIPTION", "Forced oscillation with gravity and advanced fluid forces", "-", "Phase 2 content"],
    ["T_STATIC", T_STATIC, "s", "Static equilibrium time"],
    ["T_VIBRATION_START", T_STATIC, "s", "Forced vibration start time"],
    ["T_END", T_END, "s", "Total analysis time"],
    ["DT", DT, "s", "Time step"],
    ["CONVERGENCE_THRESHOLD", convergence_threshold, "m/s", "Convergence criterion threshold"],
    ["MAX_VELOCITY", 40.0, "m/s", "Velocity limitation value"],
    ["RAYLEIGH_ALPHA", RAYLEIGH_ALPHA, "-", "Rayleigh damping coefficient (currently disabled)"]
]

with open("simulation_settings.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["parameter", "value", "unit", "description"])
    writer.writerows(simulation_settings)

for idx, rows in node_traj.items():
    fname = f"node_{idx}_traj.csv"
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time[s]", "x[m]", "y[m]", "z[m]", "velocity_magnitude[m/s]", 
                        "acceleration_magnitude[m/s2]", "phase", "displacement_from_t20_equilibrium[m]"])
        writer.writerows(rows)
    print(f"[Complete] Node {idx} → {fname}")

