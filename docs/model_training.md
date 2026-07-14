# 模型训练

默认配置在 `configs/train_config.yaml`，标签表在 `configs/label_map.json`。训练脚本支持随机种子、类别权重、梯度累积、裁剪、warmup、混合精度、早停和断点恢复。每次运行写入 `checkpoints/runs/<run_name>`，保存实际配置、环境、数据摘要、训练历史和指标。

```powershell
python training/train_ner.py --config configs/train_config.yaml --run-name my_run
python scripts/list_runs.py
python scripts/select_best_run.py --run-name my_run
```

`scripts/run_experiments.py` 只运行 `configs/experiments.yaml` 中显式声明的项。正式运行 `macbert_pii_v1` 已在 CUDA 上完成，最佳 epoch 为 1；固定文本测试 F1 为 1.0，但样本高度模板化，必须通过真实业务集补充评估。存在有效 `checkpoints/best` 时不要为验证安装或界面而重复训练。
