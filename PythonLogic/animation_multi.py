import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os
import math

# 海底面パラメータ（シミュレーションコードと同じ値）
SEABED_BASE_Z = -320.0
SEABED_AMPLITUDE = 0.0
SEABED_WAVELENGTH = 200.0
# SEABED_AMPLITUDE = 10.0
# SEABED_WAVELENGTH = 200.0

# ポリエステル部分の定義（シミュレーションコードと同じ値）
POLYESTER_START_NODE = 5
POLYESTER_END_NODE = 10

def get_seabed_z(x_coord):
    return SEABED_BASE_Z + SEABED_AMPLITUDE * math.sin(2 * math.pi * x_coord / SEABED_WAVELENGTH)

def load_node_data():
    node_data = {}
    
    for node_id in range(21):
        filename = f"node_{node_id}_traj.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            node_data[node_id] = {
                'time': df['time[s]'].values,
                'x': df['x[m]'].values,
                'y': df['y[m]'].values,
                'z': df['z[m]'].values
            }
            print(f"Loaded {filename}: {len(df)} time steps")
        else:
            print(f"[Warning] {filename} not found")
    
    return node_data

def get_segment_material(seg_id):
    if 5 <= seg_id <= 9:
        return "Polyester"
    elif 3 <= seg_id <= 4: # 遷移部分（前）
        return "Transition"
    elif 10 <= seg_id <= 11: # 遷移部分（後）
        return "Transition"
    else:
        return "Steel"

def get_segment_color(material):
    color_map = {
        "Steel": "blue",
        "Polyester": "red", 
        "Transition": "orange"
    }
    return color_map.get(material, "blue")

def find_time_indices_for_1sec_intervals(time_array):
    indices = []
    target_times = np.arange(0, time_array[-1] + 1, 1.0)  # 1秒間隔
    
    for target_time in target_times:
        closest_idx = np.argmin(np.abs(time_array - target_time))
        if closest_idx not in indices:  # 重複を避ける
            indices.append(closest_idx)
    
    return indices

def create_line_animation(node_data, use_1sec_intervals=True):
    
    if 0 not in node_data:
        print("[Error] Node 0 data not found")
        return None, None
    
    total_steps = len(node_data[0]['time'])
    time_array = node_data[0]['time']
    
    # フレームインデックスの決定
    if use_1sec_intervals:
        frame_indices = find_time_indices_for_1sec_intervals(time_array)
        print(f"Using 1-second intervals: {len(frame_indices)} frames")
    else:
        skip_frames = 100
        frame_indices = list(range(0, total_steps, skip_frames))
        print(f"Using every {skip_frames} frames: {len(frame_indices)} frames")
    
    num_frames = len(frame_indices)
    
    print(f"Total time steps: {total_steps}")
    print(f"Time range: {time_array[0]:.2f} - {time_array[-1]:.2f} seconds")
    
    # 座標範囲の計算
    all_x = []
    all_z = []
    for node_id in range(21):
        if node_id in node_data:
            all_x.extend(node_data[node_id]['x'])
            all_z.extend(node_data[node_id]['z'])
    
    x_min, x_max = min(all_x), max(all_x)
    z_min, z_max = min(all_z), max(all_z)
    
    x_margin = (x_max - x_min) * 0.1
    z_margin = max((z_max - z_min) * 0.1, 50.0)
    x_lim = [x_min - x_margin, x_max + x_margin]
    z_lim = [z_min - z_margin, z_max + z_margin]

    # 海底面も考慮した範囲設定
    seabed_min = SEABED_BASE_Z - SEABED_AMPLITUDE
    if z_lim[0] > seabed_min - 20:
        z_lim[0] = seabed_min - 20
    
    print(f"X range: {x_lim[0]:.2f} to {x_lim[1]:.2f} m")
    print(f"Z range: {z_lim[0]:.2f} to {z_lim[1]:.2f} m")

    # 海底面の座標を事前計算
    x_seabed = np.linspace(x_lim[0], x_lim[1], 500)
    z_seabed = [get_seabed_z(x) for x in x_seabed]
    
    # プロット設定
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(x_lim)
    ax.set_ylim(z_lim)
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Z [m]', fontsize=12)
    ax.set_title('Mooring Line Animation with Material Classification', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 海底面を描画（固定）
    ax.fill_between(x_seabed, z_seabed, z_lim[0], 
                   color='brown', alpha=0.3, label='Seabed')
    ax.plot(x_seabed, z_seabed, 'brown', linewidth=2, label='Seabed Surface')
    
    # セグメント別ライン描画用オブジェクトを準備（各セグメント個別）
    segment_lines = []
    segment_materials = []
    
    for seg_id in range(20):  # 20セグメント
        material = get_segment_material(seg_id)
        color = get_segment_color(material)
        line, = ax.plot([], [], color=color, linewidth=4, alpha=0.9)
        segment_lines.append(line)
        segment_materials.append(material)
        print(f"セグメント {seg_id}: {material} ({color})")
    
    # 凡例用のダミーライン
    ax.plot([], [], color='blue', linewidth=4, label='Steel Cable', alpha=0.9)
    ax.plot([], [], color='red', linewidth=4, label='Polyester Rope', alpha=0.9)
    ax.plot([], [], color='orange', linewidth=4, label='Transition', alpha=0.9)
    
    # ノード描画用オブジェクト
    nodes_plot, = ax.plot([], [], 'ko', markersize=6, alpha=0.7, label='Nodes')
    
    # フェアリーダーポイントとアンカーポイントを強調表示
    fp_point, = ax.plot([], [], 'go', markersize=12, label='Fairlead Point', zorder=10)
    ap_point, = ax.plot([], [], 'ro', markersize=12, label='Anchor Point', zorder=10)
    
    # 接触ノード表示用
    contact_points, = ax.plot([], [], 'yo', markersize=8, label='Seabed Contact', zorder=9)
    
    # 時間表示用テキスト
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       fontsize=14, verticalalignment='top', weight='bold',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # 統計情報表示用テキスト
    stats_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, 
                        fontsize=10, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.legend(loc='upper right', fontsize=10)
    
    def animate(frame_num):
        data_idx = frame_indices[frame_num]
        current_time = time_array[data_idx]
        
        # 各ノードの座標を取得
        x_coords = []
        z_coords = []
        contact_nodes = []
        
        for node_id in range(21):
            if node_id in node_data:
                x = node_data[node_id]['x'][data_idx]
                z = node_data[node_id]['z'][data_idx]
                x_coords.append(x)
                z_coords.append(z)
                
                # 海底との接触判定（接触許容誤差：2m）
                seabed_z_local = get_seabed_z(x)
                if z <= seabed_z_local + 2.0:
                    contact_nodes.append((x, z))
        
        if len(x_coords) != 21:
            return tuple(segment_lines) + (nodes_plot, fp_point, ap_point, contact_points, time_text, stats_text)
        
        # 各セグメントのライン描画
        for seg_id in range(20):  # 20セグメント
            i, j = seg_id, seg_id + 1
            seg_x = [x_coords[i], x_coords[j]]
            seg_z = [z_coords[i], z_coords[j]]
            segment_lines[seg_id].set_data(seg_x, seg_z)
        
        # ノード点を描画
        nodes_plot.set_data(x_coords, z_coords)
        
        # 特別なポイントを描画
        fp_point.set_data([x_coords[0]], [z_coords[0]])
        ap_point.set_data([x_coords[20]], [z_coords[20]])
        
        # 接触ノードを描画
        if contact_nodes:
            contact_x, contact_z = zip(*contact_nodes)
            contact_points.set_data(contact_x, contact_z)
        else:
            contact_points.set_data([], [])
        
        # 時間表示を更新
        time_text.set_text(f'Time: {current_time:.1f} s\nFrame: {frame_num+1}/{num_frames}')
        
        # 統計情報を更新（ポリエステル部分の接触判定）
        polyester_contact_count = 0
        for x, z in contact_nodes:
            for node_id, (nx, nz) in enumerate(zip(x_coords, z_coords)):
                if abs(x - nx) < 0.1 and abs(z - nz) < 0.1:
                    if 5 <= node_id <= 10:  # ポリエステル部分のノード
                        polyester_contact_count += 1
                    break
        
        stats_info = f'Contact Nodes: {len(contact_nodes)}\n'
        stats_info += f'Polyester Contact: {polyester_contact_count}\n'
        stats_info += f'FP Position: ({x_coords[0]:.1f}, {z_coords[0]:.1f})'
        stats_text.set_text(stats_info)
        
        return tuple(segment_lines) + (nodes_plot, fp_point, ap_point, contact_points, time_text, stats_text)
    
    # アニメーション作成
    print("Creating animation...")
    interval = 200 if use_1sec_intervals else 100  # 1秒間隔の場合はゆっくり表示
    
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                 interval=interval, blit=True, repeat=True)
    
    return fig, anim

def save_animation(anim, filename='enhanced_mooring_animation.mp4'):
    try:
        print(f"Saving animation to {filename}...")
        anim.save(filename, writer='ffmpeg', fps=10, bitrate=2000)
        print(f"[Completed] Animation saved successfully as {filename}")
    except Exception as e:
        print(f"[Error] saving animation: {e}")
        print("Try saving as GIF instead...")
        try:
            gif_filename = filename.replace('.mp4', '.gif')
            anim.save(gif_filename, writer='pillow', fps=5)
            print(f"Animation saved as {gif_filename}")
        except Exception as e2:
            print(f"[Error] saving GIF: {e2}")

def main():
    print("=== Enhanced Mooring Line Animation ===")
    print("Loading node trajectory data...")
    
    node_data = load_node_data()
    
    if len(node_data) != 21:
        print(f"[Warning] Expected 21 nodes, but found {len(node_data)} nodes")
        print(f"Available nodes: {sorted(node_data.keys())}")
    
    if len(node_data) == 0:
        print("[Error] No node data found. Make sure CSV files are in the current directory.")
        return
    
    # 時間間隔を選択
    print("\nSelect time interval for animation:")
    print("1. 1-second intervals (recommended for detailed analysis)")
    print("2. Default intervals (every 100 frames)")
    
    choice = input("Enter choice (1 or 2, default=1): ").strip()
    use_1sec_intervals = (choice != "2")
    
    fig, anim = create_line_animation(node_data, use_1sec_intervals)
    
    if anim is None:
        return
    
    print("\n=== Animation Features ===")
    print("- Blue lines: Steel cable segments")
    print("- Red lines: Polyester rope segments") 
    print("- Orange lines: Transition segments")
    print("- Yellow dots: Nodes in contact with seabed")
    print("- Green dot: Fairlead point (moving)")
    print("- Red square: Anchor point (fixed)")
    print("- Brown area: Seabed profile")
    
    print("\nDisplaying animation...")
    plt.show()
    
    # 保存オプション
    save_choice = input("\nSave animation to file? (y/n): ").lower().strip()
    if save_choice == 'y':
        filename = input("Enter filename (default: enhanced_mooring_animation.mp4): ").strip()
        if not filename:
            filename = 'enhanced_mooring_animation.mp4'
        save_animation(anim, filename)

if __name__ == "__main__":
    main()
