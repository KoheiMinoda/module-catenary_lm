
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
# FP_COORDS = {"x": 0.0, "y": 0.0, "z": -70.0}
# AP_COORDS = {"x": 250.0, "y": 0.0, "z": -320.0}
FP_COORDS = {"x": 5.2, "y": 0.0, "z": -70.0}
# AP_COORDS = {"x": 853.87, "y": 0.0, "z": -320.0}
AP_COORDS = {"x": 255.2, "y": 0.0, "z": -320.0}
L_NATURAL = 354.96
# L_NATURAL = 902.2  # [m] Natural length: no stretch
XACC = 1e-4

MAXIT_P0_SOLVER = 100
MAXIT_RTSAFE = 200

h_span = abs(FP_COORDS["z"] - AP_COORDS["z"])
L_APFP = math.sqrt(
    (FP_COORDS["x"] - AP_COORDS["x"])**2 +
    (FP_COORDS["y"] - AP_COORDS["y"])**2)

_current_p0_for_funcd = 0.0

# ========= Forced Vibration Settings ===========
AMP_FL = 1
PERIOD_FL = 20.0 
OMEGA_FL = 2.0*math.pi / PERIOD_FL

# ============= Material Properties =============
EA_CHAIN = 2.525e8  # [N]
CA_CHAIN = 0.0  # Chain damping coefficient
LineDiameter = 0.0945 # [m]
LineDryMass = 54.75 # [kg/m]
WaterRho = 1025.0 # [kg/m³]
LINE_VOLUME_PER_M = math.pi * (LineDiameter / 2)**2 
BUOY_MASS_PER_M   = WaterRho * LINE_VOLUME_PER_M 
RHO_LINE_DRY = LineDryMass
RHO_LINE_WET = RHO_LINE_DRY - BUOY_MASS_PER_M
RHO_LINE = RHO_LINE_WET
g = 9.80665
CONTACT_DIAMETER = 0.18 # [m]
STRUCT_DAMP_RATIO = 0.00054473
POISSON_RATIO = 0.0
P_ATMOSPHERIC = 101325.0

# ============= Fluid Force Parameters =============
RHO_WATER = 1025.0  # Seawater density [kg/m³]
KINEMATIC_VISCOSITY = 1.35e-6  # Kinematic viscosity [m²/s]

# Mooring line fluid force coefficients
CD_NORMAL = 2.6
CD_AXIAL = 1.4
# CD_NORMAL = 1.0
# CD_AXIAL = 1.0
CLIFT = 0.0
DIAM_DRAG_NORMAL = 0.05
DIAM_DRAG_AXIAL  = 0.01592

CM_NORMAL_X = 2.0
CM_NORMAL_Y = 2.0
CM_AXIAL_Z = 1.5

CA_NORMAL_X = CM_NORMAL_X - 1.0
CA_NORMAL_Y = CM_NORMAL_Y - 1.0  
CA_AXIAL_Z = CM_AXIAL_Z - 1.0

# Current settings: Linear approximation : No use for vs_OrcaFlex
CURRENT_SURFACE = 0.0 # Surface current velocity [m/s]
CURRENT_BOTTOM = 0.0 # Bottom current velocity [m/s]
CURRENT_DIRECTION = 0.0  # Current direction [deg]

# Wave conditions
WAVE_HEIGHT = 0.0  # [m]
WAVE_PERIOD = 20.0  # [s]
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
RAYLEIGH_ALPHA = 0.0  # [1/s]
RAYLEIGH_BETA = 0.0 # [s]

class DirectionalEffectiveMassCalculator:
    
    def __init__(self, ca_coefficients, line_diameter, water_density):
        self.CA_NORMAL_X = ca_coefficients[0]
        self.CA_NORMAL_Y = ca_coefficients[1] 
        self.CA_AXIAL_Z = ca_coefficients[2]
        self.line_diameter = line_diameter
        self.water_density = water_density
    
    def calculate_local_coordinate_system(self, seg_vector):
        
        seg_length = np.linalg.norm(seg_vector)
        if seg_length < 1e-9:
            return (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
        
        t_vec = seg_vector / seg_length
        
        if abs(t_vec[2]) > 0.99:
            n1_vec = np.array([1.0, 0.0, 0.0])
            n2_vec = np.array([0.0, 1.0, 0.0])
        else:
            temp_z = np.array([0.0, 0.0, 1.0])
            n1_vec = np.cross(t_vec, temp_z)
            n1_vec = n1_vec / np.linalg.norm(n1_vec)
            
            n2_vec = np.cross(t_vec, n1_vec)
            n2_vec = n2_vec / np.linalg.norm(n2_vec)
        
        return t_vec, n1_vec, n2_vec
    
    def calculate_segment_added_mass_matrix(self, seg_vector, seg_length):
        volume = math.pi * (self.line_diameter/2)**2 * seg_length
        
        t_vec, n1_vec, n2_vec = self.calculate_local_coordinate_system(seg_vector)
        
        m_local = np.diag([
            self.water_density * volume * self.CA_NORMAL_X,
            self.water_density * volume * self.CA_NORMAL_Y,
            self.water_density * volume * self.CA_AXIAL_Z
        ])
        
        R = np.column_stack([n1_vec, n2_vec, t_vec])
        
        M_added_global = R @ m_local @ R.T
        
        return M_added_global
    
    def calculate_nodal_effective_mass_matrix(self, nodes, segments):
        
        n_nodes = len(nodes)
        effective_mass_matrices = []
        
        for k in range(n_nodes):
        
            m_structure = nodes[k]["mass"]
            M_structure = np.eye(3) * m_structure
        
            M_added_total = np.zeros((3, 3))
            
            for seg in segments:
                if seg["i"] == k or seg["j"] == k:
            
                    pos_i = np.array(nodes[seg["i"]]["pos"])
                    pos_j = np.array(nodes[seg["j"]]["pos"])
                    seg_vector = pos_j - pos_i
                    
                    if "L_current" in seg:
                        seg_length = seg["L_current"]
                    else:
                        seg_length = seg["L0"]
                    
                    M_seg_added = self.calculate_segment_added_mass_matrix(seg_vector, seg_length)
                    
                    M_added_total += 0.5 * M_seg_added
            
            M_effective = M_structure + M_added_total
            
            eigenvals = np.linalg.eigvals(M_effective)
            min_eigenval = np.min(eigenvals)
            if min_eigenval < 1e-6:
                M_effective += np.eye(3) * (1e-6 - min_eigenval)
            
            effective_mass_matrices.append(M_effective)
        
        return effective_mass_matrices
    
    def calculate_scalar_effective_masses(self, nodes, segments):
        
        mass_matrices = self.calculate_nodal_effective_mass_matrix(nodes, segments)
        scalar_masses = []
        
        for M_matrix in mass_matrices:
            scalar_mass = np.trace(M_matrix) / 3.0
            scalar_masses.append(scalar_mass)
        
        return scalar_masses
    
    def calculate_directional_effective_masses(self, nodes, segments):
        
        mass_matrices = self.calculate_nodal_effective_mass_matrix(nodes, segments)
        directional_masses = []
        
        for M_matrix in mass_matrices:
            mx = M_matrix[0, 0]
            my = M_matrix[1, 1] 
            mz = M_matrix[2, 2]
            directional_masses.append([mx, my, mz])
        
        return directional_masses

# ============= マトリックス形式の運動方程式求解 =============

class MatrixBasedDynamicsSolver:
    
    def __init__(self, mass_calculator):
        self.mass_calculator = mass_calculator
    
    def compute_acceleration_with_matrix(self, nodes, segments, force_vectors, t):
        
        n_nodes = len(nodes)
        
        mass_matrices = self.mass_calculator.calculate_nodal_effective_mass_matrix(nodes, segments)
        
        accelerations = []
        
        for k in range(n_nodes):
            M_eff = mass_matrices[k]
            F_total = force_vectors[k]
            
            try:
                acceleration = np.linalg.solve(M_eff, F_total)
            except np.linalg.LinAlgError:
                acceleration = np.linalg.pinv(M_eff) @ F_total
            
            accelerations.append(acceleration)
            
            nodes[k]["acc"] = acceleration.copy()
        
        return accelerations

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
    struct_mass_seg = RHO_LINE_DRY * L_natural_segment
    buoy_mass_seg   = BUOY_MASS_PER_M * L_natural_segment
    segments.append({
        "i": k, 
        "j": k+1, 
        "L0": L_natural_segment,
        "L_current": L_current,
        "EA": EA_CHAIN, 
        "CA": CA_CHAIN,
        "mass": struct_mass_seg, # 慣性用
        "buoy": buoy_mass_seg, # 浮力用
        "material": "Chain"
    })

print(f"[Complete] Natural segment length: {L_natural_segment:.3f} m")
print(f"[Complete] Initial state average segment length: {sum([seg['L_current'] for seg in segments])/len(segments):.3f} m")

m_node = [0.0]*num_nodes # 乾燥質量
b_node = [0.0] * num_nodes # 浮力用質量
for seg in segments:
    m_half = 0.5*seg["mass"]
    b_half = 0.5 * seg["buoy"]
    m_node[seg["i"]] += m_half
    m_node[seg["j"]] += m_half
    b_node[seg["i"]] += b_half
    b_node[seg["j"]] += b_half
    
nodes = []
for idx, coord in enumerate(nodes_xyz0):
    nodes.append({
        "pos": np.array([coord['x'], coord['y'], coord['z']], dtype=float),
        "vel": np.zeros(3),
        "acc": np.zeros(3),
        "mass": m_node[idx], # 慣性項で使う乾燥質量
        "buoy": b_node[idx] # 浮力計算で使う
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

        K_el = seg["EA"] / seg["L0"]

        Fd_scalar = RAYLEIGH_BETA * K_el * rel_vel_t

        Fd_vec = Fd_scalar * t_vec
        f_damp[i] +=  Fd_vec
        f_damp[j] += -Fd_vec

    return f_damp

# checked
def froude_krylov_force(seg_vector, diameter, fluid_acc):
    seg_length = np.linalg.norm(seg_vector)
    if seg_length < 1e-12:
        return np.zeros(3)
    
    volume = math.pi * (diameter/2)**2 * seg_length

    t_vec = seg_vector / seg_length

    fluid_acc_axial_scalar = np.dot(fluid_acc, t_vec)
    fluid_acc_axial = fluid_acc_axial_scalar * t_vec
    fluid_acc_normal = fluid_acc - fluid_acc_axial

    if abs(t_vec[2]) > 0.99:
        local_x = np.array([1.0, 0.0, 0.0])
        local_y = np.array([0.0, 1.0, 0.0])
    else:
        temp_z = np.array([0.0, 0.0, 1.0])
        local_y = np.cross(t_vec, temp_z)
        local_y = local_y / np.linalg.norm(local_y)
        local_x = np.cross(local_y, t_vec)
    
    fluid_acc_normal_x = np.dot(fluid_acc_normal, local_x)
    fluid_acc_normal_y = np.dot(fluid_acc_normal, local_y)

    F_FK_x = RHO_WATER * volume * CM_NORMAL_X * fluid_acc_normal_x * local_x
    F_FK_y = RHO_WATER * volume * CM_NORMAL_Y * fluid_acc_normal_y * local_y
    F_FK_z = RHO_WATER * volume * CM_AXIAL_Z * fluid_acc_axial_scalar * t_vec

    F_FK_total = F_FK_x + F_FK_y + F_FK_z

    return F_FK_total

# checked
def added_mass_force(seg_vector, seg_length, diameter, structure_acc, fluid_acc):
    volume = math.pi * (diameter/2)**2 * seg_length
    
    seg_length_calc = np.linalg.norm(seg_vector)
    if seg_length_calc < 1e-9:
        return np.zeros(3)
    
    t_vec = seg_vector / seg_length_calc
    
    acc_rel = structure_acc - fluid_acc
    
    acc_rel_axial_scalar = np.dot(acc_rel, t_vec)
    acc_rel_axial = acc_rel_axial_scalar * t_vec
    acc_rel_normal = acc_rel - acc_rel_axial

    if abs(t_vec[2]) > 0.99:
        local_x = np.array([1.0, 0.0, 0.0])
        local_y = np.array([0.0, 1.0, 0.0])
    else:
        temp_z = np.array([0.0, 0.0, 1.0])
        local_y = np.cross(t_vec, temp_z)
        local_y = local_y / np.linalg.norm(local_y)
        local_x = np.cross(local_y, t_vec)
    
    acc_normal_x = np.dot(acc_rel_normal, local_x)
    acc_normal_y = np.dot(acc_rel_normal, local_y)
    
    F_AM_x = - RHO_WATER * volume * CA_NORMAL_X * acc_normal_x * local_x
    F_AM_y = - RHO_WATER * volume * CA_NORMAL_Y * acc_normal_y * local_y
    F_AM_z = - RHO_WATER * volume * CA_AXIAL_Z * acc_rel_axial_scalar * t_vec

    F_AM_total = F_AM_x + F_AM_y + F_AM_z
    
    return F_AM_total

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

EXCLUDE_FLUID_FORCE_NODES = []

def calculate_advanced_morison_forces(nodes, segments, t):
    f_node = [np.zeros(3) for _ in nodes]
    
    for seg in segments:
        i, j = seg["i"], seg["j"]
        pos_i, pos_j = nodes[i]["pos"], nodes[j]["pos"]
        vel_i, vel_j = nodes[i]["vel"], nodes[j]["vel"]
        
        pos_mid = 0.5 * (pos_i + pos_j)
        vel_mid = 0.5 * (vel_i + vel_j)
        acc_mid = 0.5 * (nodes[i]["acc"] + nodes[j]["acc"])

        seg_vec = pos_j - pos_i
        seg_length = np.linalg.norm(seg_vec)
        if seg_length < 1e-9:
            continue
        
        # u_curr = get_current_velocity(pos_mid[2], t)
        u_wave, a_wave = get_wave_velocity_acceleration(pos_mid[0], pos_mid[2], t)
        
        # u_fluid = u_curr + u_wave
        u_fluid = u_wave
        a_fluid = a_wave
        
        F_FK = froude_krylov_force(seg_vec, LineDiameter, a_fluid)
        
        F_AM = added_mass_force(
            seg_vec, 
            seg_length, 
            LineDiameter,
            acc_mid, 
            a_fluid
        )
        
        u_rel = u_fluid - vel_mid
        F_drag = drag_force_advanced(
            seg_vec, 
            seg_length, 
            DIAM_DRAG_NORMAL,
            DIAM_DRAG_AXIAL,
            u_rel, 
            CD_NORMAL, 
            CD_AXIAL
        )
        
        F_total = F_FK + F_AM + F_drag
        # F_total = F_FK + F_drag
        
        if i not in EXCLUDE_FLUID_FORCE_NODES:
            f_node[i] += 0.5 * F_total
        if j not in EXCLUDE_FLUID_FORCE_NODES:
            f_node[j] += 0.5 * F_total
    
    return f_node


# ============= 行列ソルバー用の力の計算関数 =============

def calculate_rhs_forces_for_matrix_solver(nodes, segments, t, mass_calculator):

    n_nodes = len(nodes)
    
    f_axial, tensions = axial_forces(nodes, segments)
    f_seabed = seabed_contact_forces_segment_based(nodes, segments, t)
    f_damping = compute_rayleigh_damping_forces(nodes, segments) # 減衰力も追加

    f_fluid_rhs = [np.zeros(3) for _ in range(n_nodes)]
    for seg in segments:
        i, j = seg["i"], seg["j"]
        pos_i, pos_j = nodes[i]["pos"], nodes[j]["pos"]
        vel_i, vel_j = nodes[i]["vel"], nodes[j]["vel"]
        
        pos_mid = 0.5 * (pos_i + pos_j)
        vel_mid = 0.5 * (vel_i + vel_j)
        
        seg_vec = pos_j - pos_i
        seg_length = np.linalg.norm(seg_vec)
        if seg_length < 1e-9:
            continue
            
        # 流体速度と加速度を取得
        # u_curr = get_current_velocity(pos_mid[2], t)
        u_wave, a_fluid = get_wave_velocity_acceleration(pos_mid[0], pos_mid[2], t)
        # u_fluid = u_curr + u_wave
        u_fluid = u_wave

        F_FK = froude_krylov_force(seg_vec, LineDiameter, a_fluid)
        
        u_rel = u_fluid - vel_mid
        F_drag = drag_force_advanced(
            seg_vec, 
            seg_length, 
            DIAM_DRAG_NORMAL, 
            DIAM_DRAG_AXIAL,
            u_rel, 
            CD_NORMAL, 
            CD_AXIAL
        )

        M_seg_added = mass_calculator.calculate_segment_added_mass_matrix(seg_vec, seg_length)
        F_am_fluid_part = M_seg_added @ a_fluid

        F_fluid_seg_total = F_FK + F_drag + F_am_fluid_part
        
        # ノードに力を配分
        if i not in EXCLUDE_FLUID_FORCE_NODES:
            f_fluid_rhs[i] += 0.5 * F_fluid_seg_total
        if j not in EXCLUDE_FLUID_FORCE_NODES:
            f_fluid_rhs[j] += 0.5 * F_fluid_seg_total

    force_vectors = []
    for k, node in enumerate(nodes):
        Fg = np.array([0.0, 0.0, -node["mass"] * g]) # 重力
        Fbuoy = np.array([0.0, 0.0,  nodes[k]["buoy"] * g]) # 浮力
        F_total_rhs = f_axial[k] + f_seabed[k] + f_damping[k] + f_fluid_rhs[k] + Fg + Fbuoy
        force_vectors.append(F_total_rhs)
        
    return force_vectors, tensions

# ======================================================
# ============= 軸力 =============
# ====================================================


def calculate_external_pressure(z_coord):
    depth = abs(min(0.0, z_coord))
    return P_ATMOSPHERIC + WaterRho * g * depth

# checked
def axial_forces(nodes, segments):

    f_node = [np.zeros(3) for _ in nodes]
    tensions = [0.0] * len(segments)
    MAX_FORCE = 100.0e9

    for idx, seg in enumerate(segments):
        i, j = seg["i"], seg["j"]
        pos_i, pos_j = nodes[i]["pos"], nodes[j]["pos"]
        vel_i, vel_j = nodes[i]["vel"], nodes[j]["vel"]

        mid_z = 0.5 * (pos_i[2] + pos_j[2])
        Po = calculate_external_pressure(mid_z)
        Ao = 0.25 * math.pi * (0.05)**2
        poisson_effect = -2.0 * POISSON_RATIO * Po * Ao
        pressure_term = Po * Ao

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
            F_elastic = seg["EA"] * strain # + (Po*Ao)
        else:
            F_elastic = 0.0 
        
        F_strain_rate_damping = seg["EA"] * STRUCT_DAMP_RATIO * strain_rate

        total_tension = F_elastic + F_strain_rate_damping + poisson_effect

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
    K_eff = K * seg_length * radius * p_eff
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

    F_norm_vec = np.array([0.0, 0.0, F_normal])
    F_fric_vec = np.array([F_fric_lat[0], F_fric_lat[1], 0.0])
    F_total = F_norm_vec + F_fric_vec
    
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

    ca_coefficients = [CA_NORMAL_X, CA_NORMAL_Y, CA_AXIAL_Z]
    
    mass_calculator = DirectionalEffectiveMassCalculator(
        ca_coefficients, LineDiameter, RHO_WATER
    )
    
    scalar_masses = mass_calculator.calculate_scalar_effective_masses(nodes, segments)
    
    return scalar_masses

# checked
def compute_acc(nodes, segments, t):
    
    USE_MATRIX_SOLVER = True 
    
    if USE_MATRIX_SOLVER:
        ca_coefficients = [CA_NORMAL_X, CA_NORMAL_Y, CA_AXIAL_Z]
        mass_calculator = DirectionalEffectiveMassCalculator(
            ca_coefficients, LineDiameter, RHO_WATER
        )
        matrix_solver = MatrixBasedDynamicsSolver(mass_calculator)
        
        force_vectors, tensions = calculate_rhs_forces_for_matrix_solver(nodes, segments, t, mass_calculator)
        
        accelerations = matrix_solver.compute_acceleration_with_matrix(
            nodes, segments, force_vectors, t
        )
        
        return accelerations, tensions
    
    else:
        effective_masses = calculate_effective_mass(nodes, segments)

        f_axial, tensions = axial_forces(nodes, segments)
        f_seabed = seabed_contact_forces_segment_based(nodes, segments, t)
        f_fluid = calculate_advanced_morison_forces(nodes, segments, t)

        acc = []
        for k, node in enumerate(nodes):
            Fg = np.array([0.0, 0.0, -node["mass"]*g])
            Fbuoy = np.array([0.0, 0.0,  nodes[k]["buoy"] * g])

            F_tot = f_axial[k] + f_seabed[k] + f_fluid[k] + Fg + Fbuoy

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

t = 0.0
step_count = 0
output_interval = int(0.1 / DT)

convergence_check_interval = int(1.0 / DT)
convergence_threshold = 0.01

while t <= T_END:
    
    if t <= T_STATIC:
        phase = 1
        nodes[0]["pos"][:] = [FP_COORDS['x'], FP_COORDS['y'], FP_COORDS['z']]
        nodes[0]["vel"][:] = [0.0, 0.0, 0.0]
        nodes[0]["acc"][:] = [0.0, 0.0, 0.0]
        
        a_list, segment_tensions = compute_acc(nodes, segments, t)
        
        if step_count % convergence_check_interval == 0 and step_count > 0:
            max_vel = 0.0
            for k in range(1, num_nodes - 1):  # Internal nodes only
                vel_mag = np.linalg.norm(nodes[k]["vel"])
                max_vel = max(max_vel, vel_mag)
            
            convergence_data.append([t, max_vel, phase])
            
            if max_vel < convergence_threshold and t > 10.0:
                print(f"[Complete] Static equilibrium convergence achieved at t = {t:.1f}s (max velocity: {max_vel:.6f} m/s)")
    
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

with open("tension_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time[s]", "fairleader_tension[N]", "anchor_tension[N]", 
                     "max_tension[N]", "avg_tension[N]", "min_tension[N]", "phase"])
    writer.writerows(tension_data)

for idx, rows in node_traj.items():
    fname = f"node_{idx}_traj.csv"
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time[s]", "x[m]", "y[m]", "z[m]", "velocity_magnitude[m/s]", 
                        "acceleration_magnitude[m/s2]", "phase", "displacement_from_t20_equilibrium[m]"])
        writer.writerows(rows)
    print(f"[Complete] Node {idx} → {fname}")
