import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import glob
import os

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

def create_line_animation(node_data, skip_frames=100):

    if 0 not in node_data:
        print("[Error] Node 0 data not found")
        return
    
    total_steps = len(node_data[0]['time'])
    time_array = node_data[0]['time']
    
    frame_indices = range(0, total_steps, skip_frames)
    num_frames = len(frame_indices)
    
    print(f"Total time steps: {total_steps}")
    print(f"Animation frames: {num_frames} (every {skip_frames} steps)")
    print(f"Time range: {time_array[0]:.2f} - {time_array[-1]:.2f} seconds")
    
    all_x = []
    all_z = []
    for node_id in range(21):
        if node_id in node_data:
            all_x.extend(node_data[node_id]['x'])
            all_z.extend(node_data[node_id]['z'])
    
    x_min, x_max = min(all_x), max(all_x)
    z_min, z_max = min(all_z), max(all_z)
    
    x_margin = (x_max - x_min) * 0.1
    z_margin = (z_max - z_min) * 0.1
    x_lim = [x_min - x_margin, x_max + x_margin]
    z_lim = [z_min - z_margin, z_max + z_margin]
    
    print(f"X range: {x_lim[0]:.2f} to {x_lim[1]:.2f} m")
    print(f"Z range: {z_lim[0]:.2f} to {z_lim[1]:.2f} m")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(x_lim)
    ax.set_ylim(z_lim)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title('Mooring Line Animation (X-Z Plane)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # ライン描画用のオブジェクト
    line, = ax.plot([], [], 'b-', linewidth=2, marker='o', markersize=4, 
                    markerfacecolor='red', markeredgecolor='darkred')
    
    # フェアリーダーポイントとアンカーポイントを強調表示
    fp_point, = ax.plot([], [], 'go', markersize=10, label='Fairlead Point')
    ap_point, = ax.plot([], [], 'ro', markersize=10, label='Anchor Point')
    
    # 時間表示用テキスト
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(loc='upper right')
    
    def animate(frame_num):
        
        data_idx = frame_indices[frame_num]
        current_time = time_array[data_idx]
        
        # 各ノードの座標を取得
        x_coords = []
        z_coords = []
        
        for node_id in range(21):
            if node_id in node_data:
                x_coords.append(node_data[node_id]['x'][data_idx])
                z_coords.append(node_data[node_id]['z'][data_idx])
        
        # ライン全体を描画
        if len(x_coords) == 21:
            line.set_data(x_coords, z_coords)
            
            fp_point.set_data([x_coords[0]], [z_coords[0]])
            ap_point.set_data([x_coords[20]], [z_coords[20]])
        
        # 時間表示を更新
        time_text.set_text(f'Time: {current_time:.2f} s\nFrame: {frame_num+1}/{num_frames}')
        
        return line, fp_point, ap_point, time_text
    
    # アニメーション作成
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                 interval=100, blit=True, repeat=True)
    
    return fig, anim

def save_animation(anim, filename='mooring_line_animation.mp4'):
    try:
        print(f"Saving animation to {filename}...")
        anim.save(filename, writer='ffmpeg', fps=10, bitrate=1800)
        print(f"[Completed] Animation saved successfully as {filename}")
    except Exception as e:
        print(f"[Error] saving animation: {e}")
        print("Try saving as GIF instead...")
        try:
            gif_filename = filename.replace('.mp4', '.gif')
            anim.save(gif_filename, writer='pillow', fps=10)
            print(f"Animation saved as {gif_filename}")
        except Exception as e2:
            print(f"[Error] saving GIF: {e2}")

def main():

    print("Loading node trajectory data...")
    
    node_data = load_node_data()
    
    if len(node_data) != 21:
        print(f"[Warning] Expected 21 nodes, but found {len(node_data)} nodes")
        print(f"Available nodes: {sorted(node_data.keys())}")
    
    if len(node_data) == 0:
        print("[Error] No node data found. Make sure CSV files are in the current directory.")
        return
    
    fig, anim = create_line_animation(node_data, skip_frames=100)
    
    if anim is None:
        return
    
    print("Displaying animation...")
    plt.show()
    
    save_choice = input("\nSave animation to file? (y/n): ").lower().strip()
    if save_choice == 'y':
        filename = input("Enter filename (default: mooring_line_animation.mp4): ").strip()
        if not filename:
            filename = 'mooring_line_animation.mp4'
        save_animation(anim, filename)

if __name__ == "__main__":
    main()
