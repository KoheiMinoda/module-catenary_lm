import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_compare_data():
    try:
        orca_df = pd.read_csv('orca_ap.csv')
        if 'Time' not in orca_df.columns or 'OrcaFlex_AP' not in orca_df.columns:
            return
        
        tension_df = pd.read_csv('tension_data.csv')
        if 'anchor_tension[N]' not in tension_df.columns:
            return
        
        skip_rows = 1000 # 最初の1000行をスキップ
        if len(orca_df) > skip_rows:
            orca_df = orca_df.iloc[skip_rows:].reset_index(drop=True)
        else:
            return
        
        if len(tension_df) > skip_rows:
            tension_df = tension_df.iloc[skip_rows:].reset_index(drop=True)
        else:
            return
        
        orca_df['Time'] = pd.to_numeric(orca_df['Time'], errors='coerce')
        orca_df['OrcaFlex_AP'] = pd.to_numeric(orca_df['OrcaFlex_AP'], errors='coerce')
        tension_df['anchor_tension[N]'] = pd.to_numeric(tension_df['anchor_tension[N]'], errors='coerce')
        
        # fairleader_tension [N] を [kN] に変換
        tension_df['anchor_tension[kN]'] = tension_df['anchor_tension[N]'] / 1000.0
        
        # NaN 値を除去
        orca_df = orca_df.dropna(subset=['Time', 'OrcaFlex_AP'])
        tension_df = tension_df.dropna(subset=['anchor_tension[kN]'])
        
        if 'Time' not in tension_df.columns:
            # orca_data の時間間隔を使用して tension_data の時間軸を作成
            if len(orca_df) > 1:
                time_step = orca_df['Time'].iloc[1] - orca_df['Time'].iloc[0]
                tension_time = np.arange(len(tension_df)) * time_step + orca_df['Time'].iloc[0]
            else:
                tension_time = np.arange(len(tension_df))
        else:
            tension_time = pd.to_numeric(tension_df['Time'], errors='coerce')
            tension_df = tension_df.dropna(subset=['Time'])
            tension_time = tension_df['Time'].values
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # OrcaFlex_FP
        ax1.plot(orca_df['Time'], orca_df['OrcaFlex_AP'], 'b-', linewidth=1.5, label='OrcaFlex_AP')
        ax1.set_ylabel('OrcaFlex_AP', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title('tension comparison', fontsize=14, fontweight='bold')
        
        # fairleader_tension[kN]
        ax2.plot(tension_time, tension_df['anchor_tension[kN]'], 'r-', linewidth=1.5, label='anchor_tension[kN]')
        ax2.set_ylabel('anchor_tension[kN]', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()

        plt.show()
        
        create_raw_comparison(orca_df, tension_df, tension_time)
        create_normalized_comparison(orca_df, tension_df, tension_time)
        
    except FileNotFoundError as e:
        print(f"{e}")
    except Exception as e:
        print(f"{e}")

def create_raw_comparison(orca_df, tension_df, tension_time):
    plt.figure(figsize=(12, 6))
    plt.plot(orca_df['Time'], orca_df['OrcaFlex_AP'], 'b-', linewidth=1.5, label='OrcaFlex_AP [kN]', alpha=0.8)
    plt.plot(tension_time, tension_df['anchor_tension[kN]'], 'r-', linewidth=1.5, label='anchor_tension[kN]', alpha=0.8)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('tension [kN]', fontsize=12)
    plt.title('tension comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('raw_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_normalized_comparison(orca_df, tension_df, tension_time):
    orca_normalized = (orca_df['OrcaFlex_AP'] - orca_df['OrcaFlex_AP'].min()) / \
                     (orca_df['OrcaFlex_AP'].max() - orca_df['OrcaFlex_AP'].min())
    
    tension_normalized = (tension_df['anchor_tension[kN]'] - tension_df['anchor_tension[kN]'].min()) / \
                        (tension_df['anchor_tension[kN]'].max() - tension_df['anchor_tension[kN]'].min())
    
    plt.figure(figsize=(12, 6))
    plt.plot(orca_df['Time'], orca_normalized, 'b-', linewidth=1.5, label='OrcaFlex_FP_normalized', alpha=0.8)
    plt.plot(tension_time, tension_normalized, 'r-', linewidth=1.5, label='anchor_tension[kN] _normalized', alpha=0.8)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Normalized Tension (0-1)', fontsize=12)
    plt.title('Normalized Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('normalized_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    load_and_compare_data()
