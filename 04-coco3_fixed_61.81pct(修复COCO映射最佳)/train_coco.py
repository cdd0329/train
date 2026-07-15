import os
os.environ["YOLO_AMP_CHECK_SKIP"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
torch.multiprocessing.set_start_method("spawn", force=True)
"""
VOC2007 + VOC2012 合并训练 + 评估 + ONNX导出
基于 servers/scripts/train.py 改编，参数完全一致
"""
import os, sys
os.environ["YOLO_OFFLINE"] = "True"
ROOT = str(__import__('pathlib').Path(__file__).resolve().parent.parent)
os.chdir(ROOT)  # 保证相对路径正确

from ultralytics import YOLO
from ultralytics import settings

settings.update({
    "sync": False, "clearml": False, "comet": False,
    "dvc": False, "hub": False, "mlflow": False,
    "neptune": False, "raytune": False, "tensorboard": False,
    "wandb": False, "vscode_msg": False,
})

# ===== 配置 =====
MODEL_NAME = f"{ROOT}/models/pretrained/yolo11s.pt"
DATA_YAML = f"{ROOT}/configs/dataset_coco.yaml"
EPOCHS = 100
BATCH = 64
IMG_SIZE = 640
DEVICE = "0"
PROJECT = f"{ROOT}/results/runs"
NAME = "yolo11s_coco3"

# 尝试读取调优结果
import json
BEST_HP_PATH = f"{ROOT}/configs/best_hp.json"
TUNED_PARAMS = {
    "lr0": 0.001, "momentum": 0.937,
    "weight_decay": 0.0005, "warmup_epochs": 3,
}
try:
    with open(BEST_HP_PATH) as f:
        tuned = json.load(f)
    TUNED_PARAMS.update({
        "lr0": tuned.get("lr0", TUNED_PARAMS["lr0"]),
        "momentum": tuned.get("momentum", TUNED_PARAMS["momentum"]),
        "weight_decay": tuned.get("weight_decay", TUNED_PARAMS["weight_decay"]),
        "warmup_epochs": tuned.get("warmup_epochs", TUNED_PARAMS["warmup_epochs"]),
    })
    print(f"Loaded tuned params: {TUNED_PARAMS}")
except FileNotFoundError:
    print(f"No tuned params, using defaults: {TUNED_PARAMS}")

def verify_data():
    import glob
    base = f"{ROOT}/data/processed"
    n_tr = len(glob.glob(f"{base}/train/images/*.jpg"))
    n_tl = len(glob.glob(f"{base}/train/labels/*.txt"))
    n_vi = len(glob.glob(f"{base}/val/images/*.jpg"))
    n_vl = len(glob.glob(f"{base}/val/labels/*.txt"))
    n_te = len(glob.glob(f"{base}/test/images/*.jpg"))
    if n_tr < 1000 or n_tl < 1000:
        raise RuntimeError(f"Data insufficient! imgs:{n_tr} labels:{n_tl}")
    print(f"train: {n_tr}imgs {n_tl}labels  val: {n_vi}imgs {n_vl}labels  test: {n_te}imgs")

def main():
    verify_data()

    print(f"Model: {MODEL_NAME}")
    print(f"Data:  {DATA_YAML}")
    print(f"Epochs: {EPOCHS}, Batch: {BATCH}, Device: CUDA:{DEVICE}")
    print(f"超参数: {TUNED_PARAMS}")

    # 1. 训练
    model = YOLO(MODEL_NAME)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=TUNED_PARAMS["lr0"],
        momentum=TUNED_PARAMS["momentum"],
        weight_decay=TUNED_PARAMS["weight_decay"],
        warmup_epochs=TUNED_PARAMS["warmup_epochs"],
        amp=True,
        workers=4,
        cache=True,         # 缓存到RAM，加速且避免OpenCV崩溃
        augment=True,
        val=True,
    )

    # 2. 评估测试集
    best = YOLO(f"{PROJECT}/{NAME}/weights/best.pt")
    print("\n=== 评估测试集 (VOC2007 test) ===")
    metrics = best.val(data=DATA_YAML, split="test", device=DEVICE)
    print(f"mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")

    # 3. ONNX导出（与服务器流程一致）
    print("\n=== 导出 ONNX ===")
    best.export(format="onnx")
    print(f"训练完成! 模型: {PROJECT}/{NAME}/")

if __name__ == "__main__":
    main()
