
import math
import numpy as np
import csv


def grav_scale(time, ramp_t=10.0):
    if time <= 0.0:
        return 0.0
    if time >= ramp_t:
        return 1.0
    fsf = time / ramp_t
    return 3.0*fsf*fsf - 2.0*fsf*fsf*fsf


class LumpedMassSimulator:
    
    def __init__(self, config):

        self.config = config
        self.nodes = []
        self.segments = []
        self.num_nodes = 0
        self.num_segs = 0
        
    def setup_nodes_and_segments(self, nodes_xyz0, ea_global, ca_global, rho_line):
        """ノードとセグメントの設定"""
        self.num_nodes = len(nodes_xyz0)
        self.num_segs = self.num_nodes - 1
        
        # セグメント生成
        self.segments = []
        for k in range(self.num_segs):
            xi, xj = nodes_xyz0[k], nodes_xyz0[k+1]
            L0 = math.dist((xi['x'], xi['y'], xi['z']),
                          (xj['x'], xj['y'], xj['z']))
            self.segments.append({
                "i": k,
                "j": k+1,
                "L0": L0,
                "EA": ea_global,
                "CA": ca_global,
                "mass": rho_line * L0
            })
        
        # 各ノードに質量を半分ずつ配分
        m_node = [0.0] * self.num_nodes
        for seg in self.segments:
            m_half = 0.5 * seg["mass"]
            m_node[seg["i"]] += m_half
            m_node[seg["j"]] += m_half
        
        # ノード初期化
        self.nodes = []
        for idx, coord in enumerate(nodes_xyz0):
            self.nodes.append({
                "pos": np.array([coord['x'], coord['y'], coord['z']], dtype=float),
                "vel": np.zeros(3),
                "mass": m_node[idx]
            })
        
        # 浮体側ノードの質量を0に設定（強制変位）
        self.nodes[0]["mass"] = 0.0
    
    def axial_forces(self):
        
        f_node = [np.zeros(3) for _ in self.nodes]

        for seg in self.segments:
            i, j = seg["i"], seg["j"]
            xi, xj = self.nodes[i]["pos"], self.nodes[j]["pos"]
            vi, vj = self.nodes[i]["vel"], self.nodes[j]["vel"]

            dx = xj - xi
            l = np.linalg.norm(dx)
            if l < 1e-9:  # 零長保護
                continue
            t = dx / l  # 単位ベクトル

            # 弾性力
            Fel = seg["EA"] * (l - seg["L0"]) / seg["L0"]
            
            # 減衰力
            vrel = np.dot(vj - vi, t)
            Fd = seg["CA"] * vrel
            
            # 軸力（張力のみ）
            Fax = Fel + Fd
            if Fax < 0.0:
                Fax = 0.0  # 圧縮不可

            Fvec = Fax * t
            f_node[i] += Fvec
            f_node[j] += -Fvec
        
        return f_node
    
    def seabed_contact_forces(self):
        """海底反力と摩擦力の計算"""
        f_node = [np.zeros(3) for _ in self.nodes]
        
        seabed_z = self.config["seabed_z"]
        k_seabed = self.config["k_seabed"]
        c_seabed = self.config["c_seabed"]
        mu_static = self.config["mu_static"]
        mu_dynamic = self.config["mu_dynamic"]
        v_slip_tol = self.config["v_slip_tol"]

        for k, nd in enumerate(self.nodes):
            z = nd["pos"][2]
            vz = nd["vel"][2]

            # 反力
            pen = seabed_z - z
            if pen <= 0.0:
                continue

            Fz_norm = k_seabed * pen - c_seabed * vz  # 上向きを正
            if Fz_norm < 0.0:
                Fz_norm = 0.0  # 引張は無し

            
            v_xy = nd["vel"].copy()
            v_xy[2] = 0.0
            v_norm = np.linalg.norm(v_xy)

            if v_norm < v_slip_tol:
                F_fric = -v_xy * 0.0  # 静止摩擦（簡易実装）
            else:
                F_fric = -mu_dynamic * Fz_norm * (v_xy / v_norm)

            f_node[k][2] += Fz_norm
            f_node[k] += F_fric
        
        return f_node
    
    def compute_acceleration(self, time):
        
        f_axial = self.axial_forces()
        f_seabed = self.seabed_contact_forces()
        scale = grav_scale(time, self.config["ramp_t"])
        g = self.config["g"]

        acc = []
        for k, node in enumerate(self.nodes):
            if node["mass"] == 0.0:
                acc.append(np.zeros(3))
                continue

            Fg = np.array([0.0, 0.0, -node["mass"] * g * scale])
            
            F_tot = f_axial[k] + f_seabed[k] + Fg

            acc.append(F_tot / node["mass"])
        
        return acc
    
    def apply_floating_body_motion(self, time):
        
        fp_coords = self.config["fp_coords"]
        amp_fl = self.config["amp_fl"]
        omega_fl = self.config["omega_fl"]
        
        x_fl = fp_coords['x'] + amp_fl * math.sin(omega_fl * time)
        vx_fl = amp_fl * omega_fl * math.cos(omega_fl * time)
        
        self.nodes[0]["pos"][:] = [x_fl, fp_coords['y'], fp_coords['z']]
        self.nodes[0]["vel"][:] = [vx_fl, 0.0, 0.0]
    
    def time_integration_step(self, time, dt):
        
        self.apply_floating_body_motion(time)
        
        a_list = self.compute_acceleration(time)
        
        
        for k in range(1, self.num_nodes - 1):
            self.nodes[k]["vel"] += a_list[k] * dt
            self.nodes[k]["pos"] += self.nodes[k]["vel"] * dt
    
    def run_simulation(self, output_nodes=None):
        
        dt = self.config["dt"]
        t_end = self.config["t_end"]
        
        if output_nodes is None:
            output_nodes = list(range(self.num_nodes))
        
        
        node_traj = {idx: [] for idx in output_nodes}
        
        print(f"\n--- Lumped-mass explicit simulation start ({t_end} s, dt={dt} s) ---")
        
        t = 0.0
        while t <= t_end:
            
            self.time_integration_step(t, dt)
            
            for idx in output_nodes:
                p = self.nodes[idx]["pos"]
                node_traj[idx].append([t, p[0], p[1], p[2]])
            
            t += dt
        
        print("--- simulation finished ---")
        return node_traj
    
    def save_results_to_csv(self, node_traj, output_dir="."):
        for idx, rows in node_traj.items():
            fname = f"{output_dir}/node_{idx}_traj.csv"
            with open(fname, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time[s]", "x[m]", "y[m]", "z[m]"])
                writer.writerows(rows)
            print(f"ノード {idx} → {fname}")
