# Git 协作

数据、模型、缓存、输出、数据库、日志和评测产物均由 `.gitignore` 排除。提交应集中于源码、配置、测试和文档，不要提交真实个人信息、下载模型或生成图片。

每个变更应先运行相关单测，再运行 `python -m pytest -v -m "not integration"`。涉及 OCR、模型加载或 UI 启动时还需运行集成测试。不要重写已冻结的数据 split 或固定图像清单；如确需版本升级，应新建有版本号的清单并记录哈希、生成器版本和迁移原因。

切换最佳模型使用 `scripts/select_best_run.py`，不要手工混合不同运行的 model、tokenizer 和 label map。实验必须在 `configs/experiments.yaml` 中显式列出，避免不可控的组合训练。
