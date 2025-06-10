import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import os

def load_tension_data(filename="tension_data.csv"):
    if not os.path.exists(filename):
        print(f"エラー: ファイル '{filename}' が見つかりません。")
        return None
    
    try:
        df = pd.read_csv(filename)
        print(f"データ読み込み完了: {len(df)} 行")
        print(f"時間範囲: {df['time[s]'].min():.1f} - {df['time[s]'].max():.1f} 秒")
        return df
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return None

def plot_tension_timeseries(df, save_plots=True):
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.figsize'] = (12, 8)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Chain Tension Analysis', fontsize=16, fontweight='bold')
    
    time = df['time[s]']
    fl_tension = df['fairleader_tension[N]']
    an_tension = df['anchor_tension[N]']
    
    # プロット1: 両方の張力の時系列
    axes[0, 0].plot(time, fl_tension/1000, 'b-', linewidth=1.5, label='Fairleader')
    axes[0, 0].plot(time, an_tension/1000, 'r-', linewidth=1.5, label='Anchor')
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Tension [kN]')
    axes[0, 0].set_title('Tension Time Series')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # プロット2: 差分の時系列
    tension_diff = fl_tension - an_tension
    axes[0, 1].plot(time, tension_diff/1000, 'g-', linewidth=1.5)
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Tension Difference [kN]')
    axes[0, 1].set_title('Fairleader - Anchor Tension')
    axes[0, 1].grid(True, alpha=0.3)
    
    # プロット3: フェアリーダー張力のヒストグラム
    axes[1, 0].hist(fl_tension/1000, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_xlabel('Fairleader Tension [kN]')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Fairleader Tension Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # プロット4: アンカー張力のヒストグラム
    axes[1, 1].hist(an_tension/1000, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1, 1].set_xlabel('Anchor Tension [kN]')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Anchor Tension Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('tension_analysis.png', dpi=300, bbox_inches='tight')
        print("プロット保存: tension_analysis.png")
    
    plt.show()

def calculate_statistics(df):
    fl_tension = df['fairleader_tension[N]']
    an_tension = df['anchor_tension[N]']
    
    print("\n=== 張力統計値 ===")
    print(f"フェアリーダー張力:")
    print(f"  平均値: {fl_tension.mean()/1000:.2f} kN")
    print(f"  最大値: {fl_tension.max()/1000:.2f} kN")
    print(f"  最小値: {fl_tension.min()/1000:.2f} kN")
    print(f"  標準偏差: {fl_tension.std()/1000:.2f} kN")
    
    print(f"\nアンカー張力:")
    print(f"  平均値: {an_tension.mean()/1000:.2f} kN")
    print(f"  最大値: {an_tension.max()/1000:.2f} kN")
    print(f"  最小値: {an_tension.min()/1000:.2f} kN")
    print(f"  標準偏差: {an_tension.std()/1000:.2f} kN")
    
    # 相関係数
    correlation = np.corrcoef(fl_tension, an_tension)[0, 1]
    print(f"\n張力間の相関係数: {correlation:.3f}")

def frequency_analysis(df, save_plots=True):
    time = df['time[s]'].values
    fl_tension = df['fairleader_tension[N]'].values
    an_tension = df['anchor_tension[N]'].values
    
    # サンプリング周波数を計算
    dt = time[1] - time[0]
    fs = 1.0 / dt
    
    # パワースペクトル密度を計算
    f_fl, psd_fl = signal.welch(fl_tension, fs, nperseg=1024)
    f_an, psd_an = signal.welch(an_tension, fs, nperseg=1024)
    
    # プロット
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(f_fl, psd_fl, 'b-', linewidth=1.5, label='Fairleader')
    plt.semilogy(f_an, psd_an, 'r-', linewidth=1.5, label='Anchor')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density [N²/Hz]')
    plt.title('Power Spectral Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.0)  # 1Hz以下に焦点
    
    # フェーズプロット
    plt.subplot(1, 2, 2)
    plt.plot(fl_tension/1000, an_tension/1000, 'k.', markersize=0.5, alpha=0.5)
    plt.xlabel('Fairleader Tension [kN]')
    plt.ylabel('Anchor Tension [kN]')
    plt.title('Tension Phase Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('tension_frequency_analysis.png', dpi=300, bbox_inches='tight')
        print("周波数解析プロット保存: tension_frequency_analysis.png")
    
    plt.show()
    
    # 支配的周波数を特定
    dominant_freq_fl = f_fl[np.argmax(psd_fl[1:])] if len(psd_fl) > 1 else 0
    dominant_freq_an = f_an[np.argmax(psd_an[1:])] if len(psd_an) > 1 else 0
    
    print(f"\n=== 周波数解析結果 ===")
    print(f"サンプリング周波数: {fs:.2f} Hz")
    print(f"フェアリーダー張力の支配的周波数: {dominant_freq_fl:.4f} Hz")
    print(f"アンカー張力の支配的周波数: {dominant_freq_an:.4f} Hz")

def export_summary_csv(df):

    time = df['time[s]']
    fl_tension = df['fairleader_tension[N]']
    an_tension = df['anchor_tension[N]']
    
    # 移動平均を計算（10秒窓）
    window_size = int(10.0 / (time.iloc[1] - time.iloc[0]))  # 10秒間の窓
    fl_ma = fl_tension.rolling(window=window_size, center=True).mean()
    an_ma = an_tension.rolling(window=window_size, center=True).mean()
    
    # サマリーデータフレーム作成
    summary_df = pd.DataFrame({
        'time[s]': time,
        'fairleader_tension[N]': fl_tension,
        'anchor_tension[N]': an_tension,
        'fairleader_tension_10s_avg[N]': fl_ma,
        'anchor_tension_10s_avg[N]': an_ma,
        'tension_difference[N]': fl_tension - an_tension,
        'fairleader_tension[kN]': fl_tension / 1000,
        'anchor_tension[kN]': an_tension / 1000
    })
    
    summary_df.to_csv('tension_summary.csv', index=False)
    print("サマリーデータ出力: tension_summary.csv")

def main():
    print("=== Chain Tension Analysis ===")
    
    # データ読み込み
    df = load_tension_data()
    if df is None:
        return
    
    # 基本統計値の計算
    calculate_statistics(df)
    
    # 時系列プロット
    plot_tension_timeseries(df)
    
    # 周波数解析
    frequency_analysis(df)
    
    # サマリーCSV出力
    export_summary_csv(df)
    
    print("\n=== 解析完了 ===")

if __name__ == "__main__":
    main()
