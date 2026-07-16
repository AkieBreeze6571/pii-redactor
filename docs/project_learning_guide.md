# pii-redactor 项目学习与修改指南

本文面向第一次阅读、二次开发和维护本项目的工程师。内容以当前仓库为准，说明每个源码、配置、测试和文档文件的用途，以及可以修改什么、修改时需要同步检查什么。

## 1. 项目目标

项目在本地完成中文敏感信息检测和图片脱敏：

1. PaddleOCR 从图片提取文本和 OCR block polygon。
2. 规则、上下文和本地微调 MacBERT NER 分别产生实体候选。
3. `HybridDetector` 合并候选并处理冲突。
4. `CoordinateMapper` 将实体字符位置映射回图片，必要时使用墨迹修正或完整 OCR block 降级。
5. `RedactionService` 生成预览图、实心黑框、马赛克、模糊和 Mask。
6. Gradio 提供图片、文本、批量、历史、模型和系统状态页面。
7. SQLite 保存处理记录，JSON/CSV/ZIP 保存处理产物。

## 2. 运行时调用链

```text
app.py
  -> DocumentProcessor
     -> ImageService
     -> OcrService
        -> PaddleOCR
        -> reconstruct_text + char_map
     -> HybridDetector
        -> RuleDetector
        -> NerDetector -> NerPredictor -> checkpoints/best
        -> ContextDetector
     -> CoordinateMapper
        -> normalize_with_mapping
        -> weighted span
        -> projection refinement
        -> connected-component refinement
        -> strict/full-block fallback
     -> RedactionService
  -> DatabaseService
  -> BatchService
  -> Gradio UI
```

文本训练链路：

```text
data/raw
  -> training/prepare_ner_data.py
  -> data/processed/{train,validation,test_fixed}.jsonl
  -> training/train_ner.py
  -> checkpoints/runs/<run_name>
  -> scripts/select_best_run.py
  -> checkpoints/best
```

## 3. 修改风险等级

| 等级 | 含义 | 典型内容 |
|---|---|---|
| 低 | 局部展示或默认值，接口不变 | UI 文案、日志级别、非关键默认路径 |
| 中 | 会改变局部行为，需要相关单测 | 规则、阈值、数据库查询、批量限制 |
| 高 | 会影响训练数据、实体边界或隐私覆盖 | 标签、数据划分、融合冲突、坐标映射、脱敏区域 |
| 冻结 | 不应就地修改 | 固定测试集、已有正式评测报告、已完成模型权重 |

任何高风险改动都应先增加测试，再修改实现。不要使用 `test_fixed` 或固定图片测试集调参。

## 4. 根目录文件

| 文件 | 内容与用途 | 可以修改什么 | 注意事项 |
|---|---|---|---|
| `.gitignore` | 排除虚拟环境、数据、权重、缓存、数据库、日志和输出 | 新增本地产物或隐私目录的忽略规则 | 不要取消 `data/raw`、`checkpoints`、`logs`、数据库和用户输出的忽略 |
| `README.md` | 项目介绍、安装、运行、指标和安全声明 | 安装命令、功能说明、已验证结果、文档链接 | 指标只能写实际运行结果；行为变化后同步更新 |
| `app.py` | 应用入口，初始化处理器、数据库、批量服务和 Gradio | 主机、端口、环境变量、服务装配 | 不要把 OCR、检测或数据库业务逻辑塞进入口；修改后构建 Gradio 并启动验收 |
| `requirements.txt` | 通用 Python 依赖，不包含 Torch | 已验证的兼容版本范围 | 不要加入会覆盖 CUDA Torch 的普通 `torch` 依赖 |
| `requirements-gpu.txt` | 当前 CUDA 13.0 GPU Torch 安装源和版本 | 仅在验证新 CUDA/Torch 组合后更新 | 不适用于 CPU 或其它 CUDA 版本 |
| `pytest.ini` | pytest marker 配置 | 增加测试 marker | 真实 OCR/模型测试继续标记为 `integration` |
| `download.py` | 从 Hugging Face 下载 PII Bench ZH 文本数据 | 数据集路径或下载目标 | 数据已存在时不要重复下载；下载内容不提交 Git |
| `download_multipriv.py` | 下载 MultiPriv 外部数据 | `allow_patterns` 和本地目录 | MultiPriv 受非商业许可限制，不能加入监督训练 |
| `inspect_datasets.py` | 简单打印原始文本样本的早期辅助脚本 | 临时检查字段或样本数 | 可能输出敏感文本；正式统计优先使用 `dataset_tools/inspect_dataset.py` |
| `generate_images.py` | 从文本和实体生成多模板合成图片及框标注 | 已有模板参数、字体、布局、开发样本量 | 不要为当前 MVP 无限扩模板；不要覆盖固定图片测试集 |
| `preview_annotations.py` | 在生成图上绘制标注框，便于人工检查 | 输入标注路径、预览数量、绘制样式 | 输出属于本地产物，不提交 Git |

## 5. 配置文件

| 文件 | 内容与用途 | 可以修改什么 | 修改后检查 |
|---|---|---|---|
| `configs/data_config.yaml` | 原始数据发现、字段别名、划分比例、去重和报告目录 | 新数据字段映射、非冻结 split 比例、去重策略 | `tests/test_dataset.py`；固定 split 已存在时不要直接重建 |
| `configs/train_config.yaml` | 模型、批次、学习率、epoch、早停、数据路径和输出目录 | 新 run 的训练超参数 | 只通过新 run 调参；不要覆盖已有正式 run |
| `configs/inference_config.yaml` | NER 阈值、融合、OCR、坐标映射和脱敏默认值 | 开发集验证后的阈值、安全模式和边距 | `tests/test_hybrid_detector.py`、映射测试、DocumentProcessor 测试；禁止依据固定测试集调参 |
| `configs/label_map.json` | 实体类型、别名和 BIO 标签表 | 增加标签别名；必要时扩充监督标签 | BIO 标签变化会破坏旧模型兼容性，需要重建数据并训练新模型 |
| `configs/experiments.yaml` | 显式实验列表 | 添加独立命名的实验 | 空列表不会训练；不要生成隐式参数笛卡尔积 |

## 6. 数据目录占位文件

这些 `.gitkeep` 仅用于让空目录进入 Git，不包含业务逻辑：

| 文件 | 目录用途 | 可以修改什么 |
|---|---|---|
| `data/annotations/.gitkeep` | 图片 manifest 和标注目录占位 | 通常不修改 |
| `data/cache/.gitkeep` | OCR 等缓存目录占位 | 通常不修改 |
| `data/images/.gitkeep` | 图片输入目录占位 | 通常不修改 |
| `data/outputs/.gitkeep` | 脱敏结果和导出目录占位 | 通常不修改 |
| `data/processed/.gitkeep` | 统一训练、验证和测试数据目录占位 | 通常不修改 |
| `reports/.gitkeep` | 评测和验收报告目录占位 | 通常不修改 |

以下运行时目录默认被忽略，不应提交：`data/raw`、`data/generated`、`data/local_regression`、`data/outputs`、`data/cache`、`checkpoints`、`logs`、`reports` 中的生成报告和 `data/app.db*`。

## 7. 数据工具 `dataset_tools`

| 文件 | 内容与用途 | 可以修改什么 | 修改后测试 |
|---|---|---|---|
| `dataset_tools/__init__.py` | 包标识 | 一般不修改；可放包级常量 | 导入检查 |
| `dataset_tools/inspect_dataset.py` | 扫描 JSON/JSONL，校验实体 span、重复、重叠和格式，并输出数据报告 | 新格式解析、更多校验项、报告字段 | `tests/test_inspect_dataset.py`、`tests/test_dataset.py` |
| `dataset_tools/build_manifest.py` | 建立固定合成图片清单和 MultiPriv metadata-only 清单 | 新 manifest 字段、来源元数据验证 | 不得覆盖已有固定清单；检查许可证、`pseudo_label` 和 split |
| `dataset_tools/inspect_generated_images.py` | 检查图片/标注对应、框越界、空框、模板和实体分布 | 新图片校验规则或分组统计 | 在少量开发图片上运行；不要自动改写标注 |

## 8. 训练模块 `training`

| 文件 | 内容与用途 | 可以修改什么 | 注意事项/测试 |
|---|---|---|---|
| `training/__init__.py` | 训练包标识 | 一般不修改 | 导入检查 |
| `training/prepare_ner_data.py` | 发现原始数据、统一 schema、标签映射、去重、分组划分、冻结测试和报告 | 新输入格式、去重方式、开发 split 策略 | 高风险；运行 `tests/test_dataset.py`，不得覆盖冻结测试集 |
| `training/label_alignment.py` | 把字符级实体 span 对齐到 tokenizer offset 和 BIO token 标签 | 子词边界和截断策略 | 高风险；运行 `tests/test_alignment.py` |
| `training/dataset.py` | 读取 JSONL，调用 tokenizer/对齐逻辑，构造 PyTorch Dataset | batch 字段、采样或缓存 | 保持训练循环字段契约；运行 alignment/dataset 测试 |
| `training/metrics.py` | BIO 序列转实体 span，计算实体级 precision/recall/F1 | 新指标或分组统计 | 不要把 token accuracy 当实体指标；运行 alignment/metrics 测试 |
| `training/callbacks.py` | EarlyStopping 状态 | patience 或改进判断方式 | 保持 validation 指标作为选择依据 |
| `training/train_ner.py` | 正式训练循环、CUDA/FP16、优化器、早停、checkpoint、报告和 best 同步 | 训练策略、优化器、损失、断点逻辑 | 高风险且耗时；使用新 run，不能覆盖已有模型；先 smoke run |
| `training/evaluate_ner.py` | 加载本地模型，在指定 split 上评估 | 新分组指标和输出格式 | 不用 test_fixed 反复调参 |
| `training/predict_ner.py` | `NerPredictor` 推理、滑窗和重叠窗口合并 | 阈值接口、窗口合并、输出字段 | `tests/test_ner_detector.py`；保持字符 start/end 正确 |

## 9. 服务层 `services`

服务层是主要业务逻辑。UI、CLI 和批量处理都应调用这里，而不是复制算法。

| 文件 | 内容与用途 | 可以修改什么 | 修改后测试 |
|---|---|---|---|
| `services/__init__.py` | 服务包标识 | 一般不修改 | 导入检查 |
| `services/image_service.py` | 读取 PIL/路径/numpy 图片，EXIF 纠正、RGBA 合成、暗图增强、缩放和坐标比例 | 输入格式、预处理开关、尺寸限制 | `tests/test_ocr_service.py`、DocumentProcessor 测试 |
| `services/ocr_service.py` | PaddleOCR 3.x/旧格式解析、延迟加载、SHA 缓存、阅读顺序和 `char_map` 重建 | PaddleOCR 参数、结果适配、排序策略、缓存版本 | 高风险；运行 OCR 单测和真实 integration；不要在日志输出 OCR 原文 |
| `services/rule_detector.py` | 手机、身份证、银行卡、邮箱、护照、车牌、IP、URL 等规则候选和冲突优先级 | 正则、上下文要求、置信度和优先级 | `tests/test_rules.py`、`tests/test_validators.py`；避免把身份证片段识别为手机 |
| `services/context_detector.py` | 根据“联系人、地址、公司”等上下文产生保守候选和组织子类 | 关键词、窗口长度、组织分类 | `tests/test_context.py`；避免跨句扩展 |
| `services/ner_detector.py` | 本地 NER 模型包装、缺失模型降级和 reload | 模型路径、加载错误处理、阈值传递 | `tests/test_ner_detector.py`；不要自动联网下载模型 |
| `services/hybrid_detector.py` | 合并规则、NER、上下文候选，处理重叠、来源和类型筛选 | 来源权重、冲突优先级、融合字段 | 高风险；运行 `tests/test_hybrid_detector.py` |
| `services/coordinate_mapper.py` | 把实体 span 对齐到 OCR block，编排加权、投影、连通域、覆盖校验、动态边距和 strict fallback | 安全阈值、置信度组成、fallback 条件、跨 block 策略 | 隐私高风险；运行 `tests/test_mapping.py` 和 `tests/test_mapping_refinement.py`；strict 不可靠时必须整块遮挡 |
| `services/redaction_service.py` | 根据实体 polygon 生成结果图、预览图和 Mask；支持 black/mosaic/blur | 新打码模式、颜色、滤镜、文件命名 | `tests/test_redaction.py`；Mask、预览和结果必须使用同一组 polygon |
| `services/document_processor.py` | 串联预处理、OCR、检测、映射、脱敏、耗时、warning 和 JSON 报告 | 阶段编排、报告字段、错误降级 | `tests/test_document_processor.py`；不能吞掉安全 warning |
| `services/database_service.py` | SQLite 建表、SHA 去重、参数化写入、分页查询、筛选和删除 | schema 迁移、查询条件、分页、保留策略 | `tests/test_database.py`；不要存图片二进制或拼接 SQL |
| `services/batch_service.py` | 最多 20 图的隔离处理、进度、ZIP/JSON/CSV 和过期清理 | 数量/大小限制、导出结构、清理周期 | `tests/test_batch_processing.py`；单图失败不能中断整批 |

## 10. 通用工具 `utils`

| 文件 | 内容与用途 | 可以修改什么 | 修改后测试 |
|---|---|---|---|
| `utils/__init__.py` | 工具包标识 | 一般不修改 | 导入检查 |
| `utils/validators.py` | 身份证 MOD 11-2、银行卡 Luhn、IP、手机规范化与校验 | 校验规则和标准化方式 | `tests/test_validators.py`、规则测试 |
| `utils/logger.py` | 日志轮转和手机号、身份证、邮箱、字段化姓名/地址遮蔽 | 新 PII 日志遮蔽模式、文件大小、备份数 | `tests/test_logger.py`；不得降低隐私遮蔽范围 |
| `utils/geometry.py` | 文本标准化索引、字符视觉权重、倾斜 polygon 插值、透视矫正、投影、连通域和墨迹覆盖校验 | 图像阈值、权重、边界吸附、组件聚类 | 隐私高风险；运行全部映射测试和本地回归脚本 |

## 11. UI `ui`

| 文件 | 内容与用途 | 可以修改什么 | 修改后检查 |
|---|---|---|---|
| `ui/__init__.py` | UI 包标识 | 一般不修改 | 导入检查 |
| `ui/app_ui.py` | 创建 Gradio Blocks 和六个标签页 | 页面顺序、标题、全局布局 | 构建 `build_application()` 并启动首页 |
| `ui/helpers.py` | UI 数据转换、回调、实体编辑、高亮、状态信息和数据库写入 | 表格列、回调返回值、状态字段 | `tests/test_ui_helpers.py`；修改返回数量时同步 Gradio outputs |
| `ui/image_tab.py` | 图片上传、类型、阈值、安全模式、边距、结果图、实体表和下载 | 图片页面控件、默认值、布局 | UI helper 测试和 Gradio 构建；strict 默认不能被静默改为 balanced |
| `ui/text_tab.py` | 文本输入、类型、阈值、四类检测结果、高亮和 JSON | 文本页布局和显示字段 | 实际文本回调检查 |
| `ui/batch_tab.py` | 多文件上传、类型、打码方式、批量结果和下载 | 批量页面控件和表格 | BatchService 测试和 Gradio 构建 |
| `ui/history_tab.py` | 文件名/实体筛选、记录列表和删除 | 查询控件、分页、删除确认 | DatabaseService 测试；默认不能删除原始图片 |
| `ui/model_tab.py` | 显示训练/推理配置、已有 runs 和重新加载 best | 展示字段、reload 操作 | 使用已有模型检查；不要从 UI 隐式启动训练 |
| `ui/status_tab.py` | 显示环境、CUDA、模型、OCR、数据库和输出状态 | 非敏感诊断字段 | 不得显示 API Key、完整用户名或用户文档内容 |

## 12. 图片评测 `evaluation`

| 文件 | 内容与用途 | 可以修改什么 | 注意事项 |
|---|---|---|---|
| `evaluation/__init__.py` | 评测包标识 | 一般不修改 | 导入检查 |
| `evaluation/image_metrics.py` | Levenshtein、文本相似度、矩形转换、IoU 和真值覆盖率 | 新独立指标或 polygon 指标 | 指标定义变化必须版本化，不能改写旧报告含义 |
| `evaluation/error_analysis.py` | 绘制真值框、预测框、打码结果等错误产物 | 颜色、标注内容、错误样本数量 | 产物可能含 PII，只保存在忽略目录 |
| `evaluation/evaluate_images.py` | 对固定图片清单运行真实 OCR/检测/映射，按类型/模板/来源汇总报告 | 新分组维度、报告字段 | 耗时且测试集冻结；不能用结果调参，也不能覆盖旧报告后冒充同一版本 |

## 13. 命令脚本 `scripts`

| 文件 | 内容与用途 | 可以修改什么 | 注意事项 |
|---|---|---|---|
| `scripts/__init__.py` | 脚本包标识 | 一般不修改 | 导入检查 |
| `scripts/detect_text.py` | CLI 混合文本检测 | 参数、输出格式、文件输入 | 输出可能含实体原文，不要写入公共日志 |
| `scripts/run_utils.py` | 读取单个训练 run 元数据并汇总 runs | 新 run 字段或兼容旧结构 | 缺失字段应降级，不删除 run |
| `scripts/list_runs.py` | 输出实验对比 CSV/JSON | 排序、列字段、输出路径 | 只读 checkpoint，不改模型 |
| `scripts/select_best_run.py` | 验证 run、备份当前 best、复制模型并写 metadata | 完整性检查、备份命名 | 主要依据 validation；不要用 test_fixed 自动选超参数 |
| `scripts/run_experiments.py` | 运行 `experiments.yaml` 中显式实验 | 命令参数和失败处理 | 不要隐式生成组合；已有正式模型时不要误触发训练 |
| `scripts/regression_test_mapping.py` | 对本地问题图输出 OCR block、加权/投影/连通域/最终框和脱敏图 | 新调试图、匿名报告字段 | 输入/输出被 Git 忽略；不得打印 OCR 原文、姓名、号码或敏感文件名 |

## 14. 测试 `tests`

| 文件 | 主要覆盖内容 | 实现变化时何时修改 |
|---|---|---|
| `tests/__init__.py` | 测试包标识 | 一般不修改 |
| `tests/test_dataset.py` | 数据读取、span、标签别名、去重、split 隔离、冻结测试 | 数据准备或 label map 变化 |
| `tests/test_inspect_dataset.py` | 文件发现和无效/重复/重叠统计 | 数据检查器变化 |
| `tests/test_alignment.py` | token 对齐、截断、相邻/重叠实体和实体指标 | tokenizer 对齐或 BIO 指标变化 |
| `tests/test_validators.py` | 身份证、Luhn、IP、手机校验 | validators 变化 |
| `tests/test_rules.py` | 所有格式规则、上下文和冲突优先级 | rule detector 变化 |
| `tests/test_context.py` | 姓名/地址上下文和组织细分 | context detector 变化 |
| `tests/test_ner_detector.py` | 模型缺失降级和本地 best 存在 | NER wrapper 或模型目录约定变化 |
| `tests/test_hybrid_detector.py` | 多来源融合、重叠、阈值和类型筛选 | fusion 逻辑变化 |
| `tests/test_ocr_service.py` | PaddleOCR 3/legacy 解析、阅读顺序、char_map、预处理、失败降级和真实 OCR integration | OCR 或图片预处理变化 |
| `tests/test_mapping.py` | 基础单 block、跨 block、倾斜、裁剪、重复文本和粗略降级 | mapper 基础契约变化 |
| `tests/test_mapping_refinement.py` | 三字姓名、相邻数字、空格差异、动态边距、投影、连通域、strict/balanced 和深色背景 | 几何、安全映射或配置变化 |
| `tests/test_redaction.py` | black/mosaic/blur、Mask、多框、越界和空实体 | redaction 变化 |
| `tests/test_document_processor.py` | 完整编排、空图、损坏图、输出隔离和安全 warning | DocumentProcessor 阶段或报告变化 |
| `tests/test_database.py` | 建表、JSON、去重、分页、查询、删除和不可写降级 | database schema/query 变化 |
| `tests/test_batch_processing.py` | 单图失败隔离、数量/大小限制、导出和清理 | batch service 变化 |
| `tests/test_ui_helpers.py` | 实体表编辑、HTML 转义和不重复 OCR/NER 的重新打码 | UI helper、表格列或 state 变化 |
| `tests/test_logger.py` | 日志 PII 遮蔽和格式化参数 | logger 隐私规则变化 |

测试原则：不要为了“通过”把真实 OCR 集成测试替换成 mock；不要把真实用户问题图片加入仓库测试。

## 15. 文档 `docs`

| 文件 | 内容与用途 | 何时更新 |
|---|---|---|
| `docs/architecture.md` | 模块边界、端到端架构和主要限制 | 服务关系或核心算法变化 |
| `docs/data_pipeline.md` | 文本、合成图片、固定集和 MultiPriv 数据流程 | schema、manifest、split 变化 |
| `docs/dataset_input_guide.md` | 自有文本、图片和外部数据的目录、JSONL schema、span/box 规范与接入流程 | 数据入口、字段契约或验证方式变化 |
| `docs/project_defense_guide.md` | 面向项目答辩的完整介绍、技术原理、实验解释、局限和常见问答 | 核心架构、算法、当前最佳模型或验收结果变化 |
| `docs/model_training.md` | 训练配置、run 管理和现有模型说明 | 训练流程或模型选择方式变化 |
| `docs/image_evaluation.md` | 图片指标、冻结基线和安全解释 | 新评测版本或指标定义变化 |
| `docs/deployment.md` | Python/CUDA、本地启动、路径和日志 | 依赖、环境变量或部署方式变化 |
| `docs/git_collaboration.md` | Git 提交边界、数据与模型排除规则 | 协作或版本管理规则变化 |
| `docs/project_learning_guide.md` | 本文，逐文件学习和修改索引 | 新增、删除、重命名模块或职责变化 |

## 16. 常见修改任务

### 16.1 增加一种格式型敏感信息

1. 在 `services/rule_detector.py` 增加候选和冲突优先级。
2. 如需校验，在 `utils/validators.py` 增加函数。
3. 在 `configs/label_map.json` 的 `target_labels`/aliases 中确认类型。
4. 在 UI 类型列表中增加该类型。
5. 增加 `tests/test_rules.py` 和 `tests/test_validators.py` 用例。

格式型类型通常不必加入 NER BIO 标签。

### 16.2 增加一种 NER 监督类型

1. 修改 `configs/label_map.json` 的 BIO labels。
2. 更新数据准备和对齐测试。
3. 生成新版本 processed 数据。
4. 使用新 run 训练模型。
5. 验证 `NerPredictor`、融合和 UI。

旧模型的分类头与新标签表不兼容，不能只改配置。

### 16.3 修改 OCR

1. 修改 `services/ocr_service.py` 或 `services/image_service.py`。
2. 增加新结果格式的解析 fixture。
3. 运行 OCR 单测和 `-m integration`。
4. OCR 配置变化时提高 cache 版本，避免读取旧缓存。

### 16.4 修改局部实体框

1. 数学和图像函数放在 `utils/geometry.py`。
2. 决策和 fallback 放在 `services/coordinate_mapper.py`。
3. 安全默认值放在 `configs/inference_config.yaml`。
4. 使用虚构合成图增加 `tests/test_mapping_refinement.py`。
5. 用被忽略的本地问题图运行 `scripts/regression_test_mapping.py`。

strict 模式的目标是防漏盖；不能为了减少多盖而取消低置信度完整 block 降级。

### 16.5 修改 UI

1. 组件声明放在对应 `ui/*_tab.py`。
2. 回调业务放在 `ui/helpers.py` 或 service，不放进 `app.py`。
3. 如果表格列变化，同步 headers、datatype、行转换和测试。
4. 构建 `build_application()`，再启动 `app.py` 检查首页。

### 16.6 修改脱敏方式

1. 在 `services/redaction_service.py` 实现。
2. 结果图和 Mask 必须使用同一 polygon。
3. 同步图片页和批量页选项。
4. 为新模式增加像素级测试。

隐私安全默认仍应使用实心覆盖；模糊和马赛克不能保证不可恢复。

### 16.7 修改数据库

1. 先设计向后兼容 schema migration。
2. 所有 SQL 使用参数化参数。
3. 不存图片二进制，不默认删除原图。
4. 增加不可写、重复 hash、并发和迁移测试。

## 17. 推荐学习顺序

1. `README.md`
2. `docs/architecture.md`
3. `app.py` 和 `ui/app_ui.py`
4. `services/document_processor.py`
5. `services/ocr_service.py` 和 `services/image_service.py`
6. `services/hybrid_detector.py` 及三个检测器
7. `services/coordinate_mapper.py` 和 `utils/geometry.py`
8. `services/redaction_service.py`
9. 对应 `tests/` 文件
10. 数据准备和训练模块
11. 图片评测和实验管理脚本

## 18. 修改后的最小验证命令

```powershell
# 局部测试：按改动选择对应文件
.\.venv\Scripts\python.exe -m pytest tests/test_mapping.py tests/test_mapping_refinement.py -v

# 全部非集成
.\.venv\Scripts\python.exe -m pytest -v -m "not integration"

# 涉及 OCR 时
.\.venv\Scripts\python.exe -m pytest -v -m integration

# 构建 UI
.\.venv\Scripts\python.exe -c "from app import build_application; print(type(build_application()).__name__)"

# 本地启动
.\.venv\Scripts\python.exe app.py
```

不要因为修改文档或 UI 而重复训练正式模型，也不要在普通回归中重新运行完整 105 张固定图片评测。

## 19. 不应直接修改的内容

- `data/raw` 中的原始数据。
- 已冻结的 `data/processed/test_fixed.jsonl`。
- `data/annotations/image_test_fixed.jsonl`。
- `checkpoints/best` 和已有 `checkpoints/runs` 内的权重。
- 已发布的固定图片评测报告。
- MultiPriv 的许可证、`metadata_only` 和非商业边界。
- 日志脱敏和 strict 安全降级原则。

需要升级这些内容时，应创建新版本、新 run 或新评测清单，记录来源、哈希、配置和迁移原因，不能就地覆盖。
