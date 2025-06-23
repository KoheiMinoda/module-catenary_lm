import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

def load_and_compare_data():
    try:
        orca_df = pd.read_csv('orca_fp.csv')
        if 'Time' not in orca_df.columns or 'OrcaFlex_FP' not in orca_df.columns:
            return
        
        tension_df = pd.read_csv('tension_data.csv')
        if 'fairleader_tension[N]' not in tension_df.columns:
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
        orca_df['OrcaFlex_FP'] = pd.to_numeric(orca_df['OrcaFlex_FP'], errors='coerce')
        tension_df['fairleader_tension[N]'] = pd.to_numeric(tension_df['fairleader_tension[N]'], errors='coerce')
        
        # fairleader_tension [N] を [kN] に変換
        tension_df['fairleader_tension[kN]'] = tension_df['fairleader_tension[N]'] / 1000.0
        
        # NaN 値を除去
        orca_df = orca_df.dropna(subset=['Time', 'OrcaFlex_FP'])
        tension_df = tension_df.dropna(subset=['fairleader_tension[kN]'])
        
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
        ax1.plot(orca_df['Time'], orca_df['OrcaFlex_FP'], 'b-', linewidth=1.5, label='OrcaFlex_FP')
        ax1.set_ylabel('OrcaFlex_FP', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title('tension comparison', fontsize=14, fontweight='bold')
        
        # fairleader_tension[kN]
        ax2.plot(tension_time, tension_df['fairleader_tension[kN]'], 'r-', linewidth=1.5, label='fairleader_tension[kN]')
        ax2.set_ylabel('fairleader_tension [kN]', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        create_raw_comparison(orca_df, tension_df, tension_time)
        create_normalized_comparison(orca_df, tension_df, tension_time)
        
        # 誤差率の計算
        calculate_error_rates(orca_df, tension_df, tension_time)
        
    except FileNotFoundError as e:
        print(f"{e}")
    except Exception as e:
        print(f"{e}")

def create_raw_comparison(orca_df, tension_df, tension_time):
    plt.figure(figsize=(12, 6))
    plt.plot(orca_df['Time'], orca_df['OrcaFlex_FP'], 'b-', linewidth=1.5, label='OrcaFlex_FP [kN]', alpha=0.8)
    plt.plot(tension_time, tension_df['fairleader_tension[kN]'], 'r-', linewidth=1.5, label='fairleader_tension [kN]', alpha=0.8)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('tension [kN]', fontsize=12)
    plt.title('tension comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(500, 700)
    plt.tight_layout()
    plt.savefig('raw_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_normalized_comparison(orca_df, tension_df, tension_time):
    orca_normalized = (orca_df['OrcaFlex_FP'] - orca_df['OrcaFlex_FP'].min()) / \
                     (orca_df['OrcaFlex_FP'].max() - orca_df['OrcaFlex_FP'].min())
    
    tension_normalized = (tension_df['fairleader_tension[kN]'] - tension_df['fairleader_tension[kN]'].min()) / \
                        (tension_df['fairleader_tension[kN]'].max() - tension_df['fairleader_tension[kN]'].min())
    
    plt.figure(figsize=(12, 6))
    plt.plot(orca_df['Time'], orca_normalized, 'b-', linewidth=1.5, label='OrcaFlex_FP_normalized', alpha=0.8)
    plt.plot(tension_time, tension_normalized, 'r-', linewidth=1.5, label='fairleader_tension[kN] _normalized', alpha=0.8)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Normalized Tension (0-1)', fontsize=12)
    plt.title('Normalized Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('normalized_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_error_rates(orca_df, tension_df, tension_time):
    """
    2つの張力データの誤差率を計算し、平均誤差と最大誤差を出力する
    """
    try:
        print("\n" + "="*50)
        print("誤差率計算結果")
        print("="*50)
        
        # 時間軸の範囲を確認
        orca_time_min, orca_time_max = orca_df['Time'].min(), orca_df['Time'].max()
        tension_time_min, tension_time_max = tension_time.min(), tension_time.max()
        
        # 重複する時間範囲を特定
        common_time_min = max(orca_time_min, tension_time_min)
        common_time_max = min(orca_time_max, tension_time_max)
        
        print(f"OrcaFlex時間範囲: {orca_time_min:.2f} - {orca_time_max:.2f}")
        print(f"Tension時間範囲: {tension_time_min:.2f} - {tension_time_max:.2f}")
        print(f"重複時間範囲: {common_time_min:.2f} - {common_time_max:.2f}")
        
        if common_time_min >= common_time_max:
            print("エラー: 時間軸の重複範囲がありません")
            return
        
        # 重複範囲内のデータを抽出
        orca_mask = (orca_df['Time'] >= common_time_min) & (orca_df['Time'] <= common_time_max)
        orca_common = orca_df[orca_mask].copy()
        
        # tension_dataを補間してorca_dataの時間点に合わせる
        if len(tension_time) > 1 and len(tension_df) > 1:
            # 線形補間を使用
            interp_func = interpolate.interp1d(tension_time, tension_df['fairleader_tension[kN]'], 
                                             kind='linear', bounds_error=False, fill_value=np.nan)
            tension_interpolated = interp_func(orca_common['Time'])
            
            # NaN値を除去
            valid_mask = ~np.isnan(tension_interpolated) & ~np.isnan(orca_common['OrcaFlex_FP'])
            orca_values = orca_common['OrcaFlex_FP'][valid_mask].values
            tension_values = tension_interpolated[valid_mask]
            
            if len(orca_values) == 0:
                print("エラー: 有効なデータポイントがありません")
                return
            
            # 誤差率の計算（相対誤差：%）
            # 誤差率 = |実測値 - 参照値| / |参照値| × 100
            # ここではOrcaFlexを参照値とする
            absolute_errors = np.abs(tension_values - orca_values)
            relative_errors = np.abs(absolute_errors / orca_values) * 100
            
            # 統計値の計算
            mean_error = np.mean(relative_errors)
            max_error = np.max(relative_errors)
            min_error = np.min(relative_errors)
            std_error = np.std(relative_errors)
            median_error = np.median(relative_errors)
            
            # 結果の出力
            print(f"\nデータポイント数: {len(orca_values)}")
            print(f"平均誤差率: {mean_error:.2f}%")
            print(f"最大誤差率: {max_error:.2f}%")
            print(f"最小誤差率: {min_error:.2f}%")
            print(f"誤差率標準偏差: {std_error:.2f}%")
            print(f"誤差率中央値: {median_error:.2f}%")
            
            # 絶対値での誤差も計算
            mean_abs_error = np.mean(absolute_errors)
            max_abs_error = np.max(absolute_errors)
            
            print(f"\n平均絶対誤差: {mean_abs_error:.2f} kN")
            print(f"最大絶対誤差: {max_abs_error:.2f} kN")
            
            # 誤差率の分布をプロット
            create_error_plot(orca_common['Time'][valid_mask], relative_errors, absolute_errors)
            
            
        else:
            print("エラー: 補間に十分なデータポイントがありません")
            
    except Exception as e:
        print(f"誤差計算中にエラーが発生しました: {e}")

def create_error_plot(time_values, relative_errors, absolute_errors):
    """誤差率の時系列プロットを作成"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 相対誤差率のプロット
    ax1.plot(time_values, relative_errors, 'g-', linewidth=1.0, alpha=0.7)
    ax1.set_ylabel('相対誤差率 [%]', fontsize=12)
    ax1.set_title('張力データの誤差率分析', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=np.mean(relative_errors), color='r', linestyle='--', 
                label=f'平均誤差率: {np.mean(relative_errors):.2f}%')
    ax1.legend()
    
    # 絶対誤差のプロット
    ax2.plot(time_values, absolute_errors, 'orange', linewidth=1.0, alpha=0.7)
    ax2.set_ylabel('絶対誤差 [kN]', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=np.mean(absolute_errors), color='r', linestyle='--', 
                label=f'平均絶対誤差: {np.mean(absolute_errors):.2f} kN')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    load_and_compare_data()
