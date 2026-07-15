# COCO 修复训练 — 61.81%（最佳）

| 项目 | 内容 |
|------|------|
| 模型 | YOLO11s |
| 数据 | COCO 2017（118,287张训练 / 5,000张验证） |
| mAP50 | **61.81%** |
| mAP50-95 | 44.92% |
| 最佳 epoch | 61 |
| 训练时长 | ~30h（85 epoch） |

🔧 **修复内容：** `prepare_coco.py` 改为 `C={cat["id"]:i for i,cat in enumerate(d["categories"])}`，
从JSON动态映射，修复类别ID断号问题。

### 训练参数
batch=64, AMP=True, AdamW, imgsz=640, workers=4
