import math
from catenary_theory import CatenarySolver
from lumped_mass_dynamics import LumpedMassSimulator


def main():
    
    # ========== 設定パラメータ ==========
    
    # 座標設定
    FP_COORDS = {"x": 5.2, "y": 0.0, "z": -70.0} # 浮体側端点
    AP_COORDS = {"x": 853.87, "y": 0.0, "z": -320.0} # アンカー側端点
    L_TOTAL = 902.2 # 係留索全長 [m]
    
    # 材料特性
    EA_GLOBAL = 5.0e8 # 軸剛性 [N]
    CA_GLOBAL = 2.0e5 # 軸減衰 [N·s/m]
    
    # 係留索物性
    LineDiameter = 0.09017 # 直径 [m]
    LineMass = 77.71 # 単位長さ質量 [kg/m]
    WaterRho = 1025.0 # 海水密度 [kg/m³]
    g = 9.80665 # 重力加速度 [m/s²]
    
    # 有効線密度（浮力考慮）
    RHO_LINE = LineMass - WaterRho * 0.25 * math.pi * LineDiameter**2
    
    # 海底反力・摩擦設定
    SEABED_Z = AP_COORDS["z"] # 海底レベル
    K_SEABED = 1.0e7 # 海底剛性 [N/m]
    C_SEABED = 1.0e4 # 海底減衰 [N·s/m]
    MU_STATIC = 0.6 # 静止摩擦係数
    MU_DYNAMIC = 0.5 # 動摩擦係数
    V_SLIP_TOL = 1.0e-3 # 滑り判定速度 [m/s]
    
    # 浮体振動設定
    AMP_FL = 1.0 # 振幅 [m]
    PERIOD_FL = 20.0 # 周期 [s]
    OMEGA_FL = 2.0 * math.pi / PERIOD_FL # 角振動数 [rad/s]
    
    # 時間積分設定
    DT = 0.01 # 時間刻み [s]
    T_END = 500.0 # 解析時間 [s]
    RAMP_T = 10.0 # 重力ランプ時間 [s]
    
    # 内部ノード数
    NUM_INTERNAL_NODES = 19
    
    # ========== カテナリー理論による初期形状計算 ==========
    print("="*70)
    print("カテナリー理論による初期形状計算")
    print("="*70)
    
    # カテナリーソルバー初期化
    catenary_solver = CatenarySolver(
        fp_coords=FP_COORDS,
        ap_coords=AP_COORDS,
        l_total=L_TOTAL
    )
    
    # 初期ノード座標生成
    internal_nodes_coords, a_catenary = catenary_solver.generate_initial_nodes(
        num_internal_nodes=NUM_INTERNAL_NODES
    )
    
    # 全ノード座標（FP + 内部ノード + AP）
    nodes_xyz0 = [FP_COORDS] + internal_nodes_coords + [AP_COORDS]
    
    print(f"総ノード数: {len(nodes_xyz0)}")
    print(f"セグメント数: {len(nodes_xyz0) - 1}")
    
    # ========== 動的解析設定 ==========
    print("\n" + "="*70)
    print("ランプドマス法による動的解析")
    print("="*70)
    
    # シミュレーション設定辞書
    sim_config = {
        # 座標
        "fp_coords": FP_COORDS,
        "ap_coords": AP_COORDS,
        
        # 物理定数
        "g": g,
        
        # 海底設定
        "seabed_z": SEABED_Z,
        "k_seabed": K_SEABED,
        "c_seabed": C_SEABED,
        "mu_static": MU_STATIC,
        "mu_dynamic": MU_DYNAMIC,
        "v_slip_tol": V_SLIP_TOL,
        
        # 浮体振動
        "amp_fl": AMP_FL,
        "omega_fl": OMEGA_FL,
        
        # 時間積分
        "dt": DT,
        "t_end": T_END,
        "ramp_t": RAMP_T
    }
    
    # ランプドマスシミュレーター初期化
    simulator = LumpedMassSimulator(sim_config)
    
    # ノードとセグメント設定
    simulator.setup_nodes_and_segments(
        nodes_xyz0=nodes_xyz0,
        ea_global=EA_GLOBAL,
        ca_global=CA_GLOBAL,
        rho_line=RHO_LINE
    )
    
    # 出力ノード指定（全ノード）
    output_nodes = list(range(len(nodes_xyz0)))
    
    # ========== シミュレーション実行 ==========
    node_trajectories = simulator.run_simulation(output_nodes=output_nodes)
    
    # ========== 結果保存 ==========
    print("\n" + "="*70)
    print("結果保存")
    print("="*70)
    
    simulator.save_results_to_csv(node_trajectories)
    
    print("\n解析完了!")
    print(f"カテナリーパラメータ a = {a_catenary:.3f} m")
    print(f"解析時間: {T_END} s")
    print(f"出力ファイル: node_0_traj.csv ~ node_{len(nodes_xyz0)-1}_traj.csv")


if __name__ == "__main__":
    main()
