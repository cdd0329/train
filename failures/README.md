# 训练翻车记录（答辩素材）

## VOC20 翻车

| # | 问题 | 表现 | 根因 | 解决 |
|---|------|------|------|------|
| 1 | batch=64 OOM | CUDA OOM | RTX 3090显存不够 | batch=16 |
| 2 | NaN loss | 训练loss炸到NaN | lr0=0.01梯度爆炸 | 调优后lr0=0.0005 |
| 3 | cos_lr不匹配 | 验证mAP只有76.32%(调优81.79%) | 50epoch调优→200epoch，余弦退火周期被拉长 | lrf=0.1 + 100epoch |
| 4 | AMP下载卡死 | 训练卡在AMP检查 | 服务器无外网 | 预下载yolo11n.pt或amp=False |
| 5 | 数据分法错 | train 16,406(超了VOC2012全量) | prepare脚本重复处理图片 | D盘脚本用90/10正确分法 |

## COCO 翻车

| # | 问题 | 表现 | 根因 | 解决 |
|---|------|------|------|------|
| 1 | 类别映射错 | 狗认成马、卧室标成斑马 | prepare_coco.py硬编码`C={i+1:i for i in range(80)}`，COCO类ID跳号 | 改用JSON动态映射 |
| 2 | CUBLAS OOM | workers>0时cublasCreate崩 | fork模式CUDA上下文损坏 | `torch.multiprocessing.set_start_method('spawn')` |
| 3 | 验证路径错 | val mAP一直显示13% | yaml用了相对路径 | 改为绝对路径 |
| 4 | NaN爆炸 | 续训到epoch 79 loss变NaN | lr0=0.001对已训模型太高 | lr0=0.0005 |
| 5 | AMP下载卡 | 同VOC | 无外网 | amp=False |
