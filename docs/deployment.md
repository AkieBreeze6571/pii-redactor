# 本地部署

推荐 Windows/Linux、Python 3.10 或 3.11、CUDA 13.0 对应驱动和至少 8 GB 显存。先安装 `requirements.txt`，再按硬件安装 Torch；当前 NVIDIA 环境可使用 `requirements-gpu.txt`。运行 `pip check` 和两组 pytest 后再启动 `python app.py`。

默认服务仅绑定 `127.0.0.1:7860`。数据目录、模型和端口可通过 `PII_DATA_DIR`、`PII_MODEL_PATH`、`PII_APP_PORT` 修改。不要将真实文档目录放入 Web 静态路径；不要记录 OCR 原文或完整实体值；定期清理 `data/outputs`、`data/cache`、`logs` 和数据库。

启动前确认 `checkpoints/best/model` 与 tokenizer 可读、PaddleOCR 模型可用、输出目录有写权限。应用日志位于 `logs/app.log`，训练日志位于 `logs/training.log`，均采用轮转并对常见结构化 PII 做遮蔽。
