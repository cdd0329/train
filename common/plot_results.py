"""训练曲线可视化 - 从results.csv画图"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
from pathlib import Path

def plot_results(csv_path, output_path, title="Training Curves"):
    """读取results.csv画训练曲线"""
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} 不存在")
        return
    
    df = pd.read_csv(csv_path)
    epochs = df['epoch']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16)
    
    # 第一行：损失曲线
    loss_cols = [c for c in df.columns if 'loss' in c]
    for i, col in enumerate(loss_cols):
        if i < 3:
            ax = axes[0, i]
            ax.plot(epochs, df[col], 'b-', linewidth=1.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(col.replace('train/', ''))
            ax.set_title(col.replace('train/', ''))
            ax.grid(True, alpha=0.3)
    
    # 第二行：指标曲线
    metric_cols = [c for c in df.columns if 'metrics/' in c]
    colors = ['g', 'r', 'c', 'm']
    for i, col in enumerate(metric_cols):
        if i < 3:
            ax = axes[1, i]
            ax.plot(epochs, df[col], colors[i % len(colors)], linewidth=1.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(col.replace('metrics/', ''))
            ax.set_title(col.replace('metrics/', ''))
            ax.grid(True, alpha=0.3)
    
    # 最后一张图放 mAP50 + mAP50-95 双曲线
    ax = axes[1, 2]
    for col in metric_cols:
        if 'mAP50' in col and '95' not in col:
            ax.plot(epochs, df[col], 'g-', linewidth=2, label='mAP50')
        elif 'mAP50-95' in col:
            ax.plot(epochs, df[col], 'orange', linewidth=2, label='mAP50-95')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_title('mAP Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] 曲线图已保存: {output_path}")
    plt.close()

def plot_tune_comparison(results_dir, output_path):
    """对比多组调参结果的mAP曲线"""
    runs = sorted(Path(results_dir).glob("tune_*/results.csv"))
    if not runs:
        print(f"[ERROR] {results_dir} 下没有调参结果")
        return
    
    plt.figure(figsize=(12, 6))
    for csv_file in runs:
        df = pd.read_csv(csv_file)
        label = csv_file.parent.name.replace('tune_', '')
        plt.plot(df['epoch'], df['metrics/mAP50(B)'], linewidth=1.5, label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel('mAP50')
    plt.title('Hyperparameter Tuning Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] 调参对比图: {output_path}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv) > 2 else csv_path.replace('.csv', '.png')
        plot_results(csv_path, out_path)
    else:
        print("用法: python plot_results.py <results.csv> [output.png]")
