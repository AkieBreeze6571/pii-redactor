# pii-redactor

本地运行的中文文档敏感信息检测与图像脱敏系统。项目组合 PaddleOCR、规则检测、上下文检测和微调后的 MacBERT NER，在图片或文本中识别个人信息，并输出黑框、马赛克或模糊脱敏结果、透明 Mask 和 JSON 报告。

## 当前能力

- 图片：OCR、混合实体检测、坐标映射、实体框预览、可编辑结果、三种脱敏方式。
- 文本：分别展示规则、NER、上下文和融合结果。
- 批量：单次最多 20 个文件，单文件失败隔离，导出 ZIP/CSV/JSON。
- 本地历史：SQLite 去重、分页检索、实体类型过滤和删除。
- 模型：训练、评估、预测、显式实验列表、运行对比和最佳模型切换。
- 隐私：默认仅监听 `127.0.0.1`，不上传文档；日志会遮蔽常见手机号、身份证号和邮箱。

## 架构与目录

处理链路为 `图片预处理 -> PaddleOCR 预训练模型 -> 阅读顺序重建 -> 规则/上下文/MacBERT NER 融合 -> 字符坐标映射 -> 图像脱敏 -> SQLite/导出`。NER 基于 `hfl/chinese-macbert-base` 中文预训练模型微调。

```text
configs/          训练、推理和实验配置
dataset_tools/    数据和图片标注检查
training/         数据准备、训练、评估和预测
evaluation/       图片指标与错误分析
services/         OCR、检测、映射、脱敏、数据库和批量服务
ui/               Gradio Blocks 页面
scripts/          文本检测与模型运行管理
tests/            单元测试和集成测试
docs/             设计、数据、训练、评测和部署文档
```

## 环境安装

推荐 Python 3.10 或 3.11。当前开发环境 Python 3.13.9 可以运行，但部分深度学习依赖对 3.10/3.11 的发布与兼容验证更完整。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
pip install -r requirements-gpu.txt
pip check
```

`requirements-gpu.txt` 固定为当前已验证的 `torch==2.13.0+cu130`。CPU 环境可在基础依赖后执行 `pip install torch`；其它 CUDA 版本应从 PyTorch 官方安装页选择匹配命令，不能使用该文件覆盖已有 CUDA Torch。

## 数据准备

PII Bench ZH 是文本实体数据，放置在 `data/raw` 后由项目统一为 BIO 训练数据；`data/generated` 中的合成图片由本项目脚本生成，并非外部真实文档。MultiPriv 只用于 `external_test` 非商业外部检查，不进入训练、验证或固定测试集。

```powershell
python dataset_tools/inspect_dataset.py
python training/prepare_ner_data.py
python dataset_tools/inspect_generated_images.py
```

已存在冻结 split 或固定图片清单时，不要重新生成或覆盖。原始数据的具体放置层级和统一格式见 `docs/data_pipeline.md`。

## 启动

```powershell
python app.py
```

浏览器访问 `http://127.0.0.1:7860`。可用环境变量：`PII_APP_HOST`、`PII_APP_PORT`、`PII_APP_SHARE`、`PII_DATA_DIR`、`PII_MODEL_PATH`。除非明确接受局域网或公网暴露风险，不要把监听地址改为 `0.0.0.0`，也不要启用分享链接。

## 常用命令

```powershell
# 混合文本检测
python scripts/detect_text.py --text "联系人张三，电话13800138000"

# 训练、评估和预测
python training/train_ner.py --config configs/train_config.yaml
python training/evaluate_ner.py --model-path checkpoints/best
python training/predict_ner.py --model-path checkpoints/best --text "张三住在四川省成都市武侯区"

# 查看运行并切换最佳模型（不会自动重训）
python scripts/list_runs.py
python scripts/select_best_run.py --run-name macbert_pii_v1

# 仅运行 configs/experiments.yaml 中显式列出的调参实验
python scripts/run_experiments.py --config configs/experiments.yaml

# 固定图像集评测
python evaluation/evaluate_images.py --annotations data/annotations/image_test_fixed.jsonl --output-dir reports

# 测试
python -m pytest -v -m "not integration"
python -m pytest -v -m integration
```

`configs/experiments.yaml` 只执行显式列出的实验，不生成参数笛卡尔积。已有 `checkpoints/best` 时无需重复正式训练。

批量处理在 Gradio 的“批量处理”页选择最多 20 张图片，可下载结果 ZIP、总报告 JSON 和 CSV；每张图片独立失败，不会中断整批。

## 已验证结果

- 正式 MacBERT 运行：固定文本测试集 macro/micro F1 均为 1.0。该数据来自模板化合成样本，不能推断真实业务泛化能力。
- 固定图像集 105 张：OCR 文本相似度 0.9164，实体 F1 0.8592，平均框 IoU 0.4124。
- 以 0.95 覆盖阈值计算的完整脱敏率为 0，说明字符级插值框仍不足以替代人工复核和更保守的边距设置。

固定测试集只用于最终评测，不得据此调参。MultiPriv 清单为 `metadata_only`、非伪标签，并受 `CC BY-NC-SA 4.0` 非商业限制。

## 数据与产物

原始数据、处理后数据、模型、数据库、输出图、缓存、日志和报告默认不提交 Git。主要位置：

- `data/raw`、`data/processed`、`data/annotations`
- `checkpoints/runs`、`checkpoints/best`
- `data/outputs`、`data/app.db`
- `reports`、`logs`

详细说明见 [项目答辩与技术原理详解](docs/project_defense_guide.md)、[项目学习与修改指南](docs/project_learning_guide.md)、[自有数据接入指南](docs/dataset_input_guide.md)、[架构](docs/architecture.md)、[数据流程](docs/data_pipeline.md)、[模型训练](docs/model_training.md)、[图像评测](docs/image_evaluation.md)、[部署](docs/deployment.md) 和 [Git 协作](docs/git_collaboration.md)。

## 许可证和责任边界

仓库代码许可不自动覆盖外部数据集或预训练模型。部署者必须分别核对 PII Bench ZH、MultiPriv、MacBERT 和 PaddleOCR 模型资产的许可。MultiPriv 的 `CC BY-NC-SA 4.0` 禁止商业使用。

系统可能漏检。模糊和马赛克不能保证敏感内容不可恢复，默认推荐实心黑框；低置信度、粗略映射和高风险业务文档必须人工复核。

## 协作与后续规划

提交前运行单元测试，涉及 OCR/UI 时再运行集成测试。数据集、权重、数据库、输出、缓存和日志不得提交 Git；固定评测集不得用于调参。后续重点是增加真实人工标注图片、改用字形级坐标或文本检测框细分、完善跨行实体映射和针对实际业务分布的阈值校准，而不是继续扩展合成模板。
