# 通用工具

## plot_results.py
训练曲线可视化工具（需 matplotlib）。

用法：
```bash
python plot_results.py path/to/results.csv
python plot_results.py path/to/results.csv --output myplot.png
python plot_results.py results/runs/yolov8n_voc07_12/results.csv
```

## monitor_train.py
训练监控脚本，每30秒检查一次训练进度。

用法：
```bash
python scripts/monitor_train.py
```
