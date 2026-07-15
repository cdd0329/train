"""
训练监控脚本 — 每 30 分钟检查一次，记录日志
用法: python scripts/monitor_train.py
"""
import os, time, csv, json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results/runs/yolov8n_voc07_12/results.csv"
LOG_PATH = ROOT / "training_log.txt"

def log(msg):
    ts = datetime.now().strftime("%m-%d %H:%M")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def read_latest():
    """读取最后一行的训练指标"""
    if not Path(CSV_PATH).exists():
        return None
    with open(CSV_PATH) as f:
        lines = list(csv.reader(f))
    if len(lines) < 2:
        return None
    header = lines[0]
    last = lines[-1]
    d = dict(zip(header, last))
    return {
        "epoch": int(d.get("epoch", 0)),
        "mAP50": float(d.get("metrics/mAP50(B)", 0)),
        "mAP50_95": float(d.get("metrics/mAP50-95(B)", 0)),
        "precision": float(d.get("metrics/precision(B)", 0)),
        "recall": float(d.get("metrics/recall(B)", 0)),
        "train_loss": float(d.get("train/box_loss", 0)),
        "val_loss": float(d.get("val/box_loss", 0)),
    }

def main():
    log("====== 训练监控启动 ======")
    log(f"监控文件: {CSV_PATH}")
    log(f"检查间隔: 30分钟")

    prev_epoch = 0
    stall_count = 0

    while True:
        d = read_latest()

        if d is None:
            log("等待 results.csv 生成...")
            time.sleep(60)
            continue

        # 检查进度
        epoch = d["epoch"]
        progress = epoch / 100 * 100

        if epoch == prev_epoch:
            stall_count += 1
            if stall_count >= 3:
                log("⚠️ 训练可能已卡住！连续 3 次检查 epoch 未变")
        else:
            stall_count = 0
            eta = "完成!" if epoch >= 100 else f"还需约 {(100-epoch)*3.2:.0f} 分钟"
            log(f"Epoch {epoch}/100 ({progress:.0f}%) | "
                f"mAP50={d['mAP50']:.4f} | mAP50-95={d['mAP50_95']:.4f} | "
                f"P={d['precision']:.3f} R={d['recall']:.3f} | "
                f"train_loss={d['train_loss']:.4f} val_loss={d['val_loss']:.4f} | {eta}")

        prev_epoch = epoch

        # 训练完成
        if epoch >= 100:
            log("====== 训练完成! ======")
            log(f"最终 mAP50: {d['mAP50']:.4f}")
            log(f"最终 mAP50-95: {d['mAP50_95']:.4f}")
            break

        if epoch >= 90:
            # 快结束了，5分钟检查一次
            time.sleep(300)
        else:
            time.sleep(1800)  # 30分钟

if __name__ == "__main__":
    main()
