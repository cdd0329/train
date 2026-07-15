# VOC 第二轮训练 — 78.12%

| 项目 | 内容 |
|------|------|
| 模型 | YOLO11s（COCO预训练→VOC finetune） |
| 数据 | VOC2007 trainval + VOC2012 trainval 90% |
| 验证 | VOC2012 trainval 10%（1,154张） |
| 测试 | VOC2007 test（4,952张，官方benchmark） |
| mAP50（test） | **78.12%** |
| 最佳 epoch | 86（val mAP50=75.15%） |
| 训练时长 | ~15h |

### 最佳超参
- lr0=0.0005, momentum=0.937, weight_decay=0.0005, warmup_epochs=5
- cos_lr=True, batch=64, AdamW
