# 数据集详解与自有数据接入指南

本文说明项目使用哪些数据、每类数据应放到哪里、自有数据必须保持什么格式，以及数据准备、检查、训练和图片评测之间的关系。所有示例均为虚构内容。

## 1. 先判断你的数据属于哪一类

| 你的数据 | 是否需要标注 | 推荐位置 | 用途 |
|---|---:|---|---|
| 中文文本和实体字符位置 | 是 | `data/raw/user_text/` | 准备 NER 训练、验证数据 |
| 只想上传处理的普通图片 | 否 | `data/images/user/` 或直接通过 UI 上传 | OCR、检测和脱敏推理 |
| 图片、全文、实体字符位置和真值框 | 是 | 图片放 `data/generated/user/`，标注放 `data/generated/annotations_user.jsonl` | 图片数据检查或开发集评测 |
| 已统一好的图片评测 manifest | 是 | `data/annotations/image_dev_user.jsonl` | `evaluation/evaluate_images.py` |
| 本地坐标映射问题图片 | 否 | `data/local_regression/input/` | 单图映射诊断，不进入训练或正式评测 |
| 只有图片和人物/来源元数据 | 否 | 自定义外部目录；manifest 放 `data/annotations/` | 定性或 metadata-only 外部检查 |

这些目录默认受 `.gitignore` 保护。不要把真实姓名、电话、证件、地址、图片、数据库或生成报告提交 Git。

## 2. 当前数据目录结构

```text
data/
├── raw/
│   ├── formal.jsonl                 # 当前正式风格文本原始数据
│   ├── chat.jsonl                   # 当前聊天风格文本原始数据
│   ├── user_text/                   # 推荐放自有文本数据
│   │   ├── formal.jsonl
│   │   └── chat.jsonl
│   └── multipriv/                   # MultiPriv 原始外部数据
├── processed/
│   ├── train.jsonl                  # 数据准备脚本生成
│   ├── validation.jsonl             # 数据准备脚本生成
│   ├── test_fixed.jsonl             # 冻结测试集，不应覆盖
│   ├── external_test.jsonl          # metadata-only 外部清单
│   └── split_manifest.json          # split 数量、来源和源文件哈希
├── generated/
│   ├── images/                      # 合成或自有带框图片
│   ├── annotations_train.jsonl      # 生成图片训练/开发标注
│   ├── annotations_test.jsonl       # 生成器的测试候选标注
│   └── annotations_user.jsonl       # 推荐的自有图片标注文件名
├── annotations/
│   ├── image_test_fixed.jsonl       # 冻结图片测试集，不应覆盖
│   ├── image_dev_user.jsonl         # 推荐的自有图片开发集 manifest
│   └── multipriv_manifest.jsonl     # MultiPriv metadata-only 清单
├── images/user/                     # 仅推理图片，可自行建立子目录
├── local_regression/
│   ├── input/                       # 单张本地问题图片
│   ├── output/                      # 映射调试图
│   └── report.json                  # 不包含 OCR 原文的诊断报告
├── cache/                           # OCR 缓存
├── outputs/                         # 脱敏结果、Mask、JSON、ZIP
└── app.db                           # 本地历史记录数据库
```

`data/processed` 是脚本输出，不是自有原始数据入口。除非进行明确的数据迁移，不要手工编辑其中的 JSONL。

## 3. JSONL 基础规则

项目的文本数据和图片 manifest 使用 JSONL，而不是一个大 JSON 数组。

JSONL 要求：

1. 文件使用 UTF-8 或 UTF-8 with BOM 编码。
2. 每一行是一个完整 JSON object。
3. 行与行之间不使用逗号。
4. 文件中不能有注释。
5. `start`、`end`、宽高和坐标必须是 JSON number，不能写成字符串。
6. 布尔值使用小写 JSON 值 `true`、`false`。
7. 文本中的换行应写成 JSON 转义 `\n`，不能把一条记录拆成多行。

正确：

```json
{"id":"sample_0001","text":"赵明宇住在测试市","entities":[{"type":"person","text":"赵明宇","start":0,"end":3},{"type":"address","text":"测试市","start":5,"end":8}]}
{"id":"sample_0002","text":"这是一条无敏感实体的负样本。","entities":[]}
```

错误：

```json
[
  {"id": "sample_0001", "text": "..."},
  {"id": "sample_0002", "text": "..."}
]
```

错误原因是它是 JSON array，不是“一行一条”的 JSONL。

## 4. 自有文本训练数据

### 4.1 推荐放置位置

如果自有数据分为正式文档和聊天文本，可放置为：

```text
data/raw/user_text/formal.jsonl
data/raw/user_text/chat.jsonl
```

当前 `configs/data_config.yaml` 使用递归模式：

```yaml
formal_patterns:
  - "**/formal.jsonl"
  - "**/pii_bench_zh.jsonl"
chat_patterns:
  - "**/chat.jsonl"
  - "**/pii_bench_zh_chat.jsonl"
```

因此上述文件名可以被自动发现。其它文件名不会自动进入数据准备，除非：

1. 在 `configs/data_config.yaml` 增加 pattern；或
2. 使用 `--formal`、`--chat` 显式传入文件。

注意：使用默认 `configs/data_config.yaml` 时，程序会递归扫描 `data/raw`，因此会把当前已有的 `data/raw/formal.jsonl`、`data/raw/chat.jsonl` 与 `data/raw/user_text/` 下的同名文件一起合并。若只想处理自己的数据，请使用下面的 `--formal`/`--chat` 参数，或建立一份输入根目录独立的配置文件。

示例：

```powershell
.\.venv\Scripts\python.exe training/prepare_ner_data.py `
  --formal data/raw/user_text/my_formal.jsonl `
  --chat data/raw/user_text/my_chat.jsonl `
  --output-dir data/processed/user_v1
```

### 4.2 原始文本记录的标准格式

推荐每条记录包含：

```json
{
  "id": "user_formal_000001",
  "text": "赵明宇住在测试市",
  "entities": [
    {
      "type": "person",
      "text": "赵明宇",
      "start": 0,
      "end": 3
    },
    {
      "type": "address",
      "text": "测试市",
      "start": 5,
      "end": 8
    }
  ]
}
```

字段说明：

| 字段 | 必需 | 类型 | 含义 |
|---|---:|---|---|
| `id` | 推荐 | string | 样本唯一 ID；也接受字段名 `sample_id` |
| `text` | 是 | string | 完整原文；也接受字段名 `content` |
| `entities` | 是 | array | 实体列表；也接受字段名 `labels`；负样本使用空数组 |
| `entities[].type` | 是 | string | 实体类型；也接受字段名 `label` |
| `entities[].text` | 是 | string | 实体原文；也接受字段名 `value` |
| `entities[].start` | 是 | integer | 实体首字符的 0-based 下标；也接受 `start_offset` |
| `entities[].end` | 是 | integer | 实体末字符之后的 0-based 下标；也接受 `end_offset` |

### 4.3 start/end 的精确定义

项目使用 Python 半开区间：

```python
text[start:end] == entity_text
```

规则：

- `start` 从 0 开始。
- `end` 指向实体最后一个字符之后。
- 中文字符按 Python Unicode 字符下标计算，不按 UTF-8 字节数计算。
- 标点、空格、英文字母和数字各占一个 Python 字符位置。
- 实体前后的空格是否属于实体必须由标注规范明确决定。

可用下面的短脚本避免手算：

```powershell
@'
text = "赵明宇住在测试市"
value = "测试市"
start = text.index(value)
print(start, start + len(value), text[start:start + len(value)])
'@ | .\.venv\Scripts\python.exe -
```

如果 `text[start:end] != entity.text`，该实体会进入 `reports/invalid_annotations.jsonl`，不会作为有效监督 span 使用。

### 4.4 支持的字段别名

当前配置允许以下字段名：

```text
样本 ID:       id | sample_id
全文:          text | content
实体列表:      entities | labels
实体类型:      type | label
实体文字:      text | value
开始位置:      start | start_offset
结束位置:      end | end_offset
```

推荐始终使用标准字段 `id/text/entities/type/text/start/end`，减少后续工具兼容成本。

### 4.5 实体类型

`configs/label_map.json` 当前声明的目标类型包括：

```text
person
address
phone
id_number
email
bank_card
passport
license_plate
organization
```

常见别名会被转换，例如 `PER -> person`、`LOC -> address`、`ORG -> organization`。

重要区别：当前 NER 模型 BIO 分类头只监督：

```text
person
address
organization
```

手机号、身份证、银行卡、邮箱、护照、车牌等主要由规则检测器处理。把这些类型放进文本 JSONL 可以保留数据统计，但不会自动让现有 NER 分类头学习它们。若要把新类型加入 NER，必须同步修改 BIO labels、重新准备数据并训练一个新 run。

### 4.6 负样本

没有敏感实体的文本应保留，并使用空数组：

```json
{"id":"negative_000001","text":"今天是普通的测试通知。","entities":[]}
```

不要删除所有负样本，否则模型更容易把普通文本误判为实体。

### 4.7 重复和重叠

- 完全相同文本按文本 SHA-256 去重。
- 同文本但实体标注不同会记录到 `reports/conflicting_duplicates.jsonl`。
- 同一模板的记录会分组，尽量避免相似模板跨 split 泄漏。
- 监督实体不应互相重叠，否则 token 对齐可能无法生成唯一 BIO 标签。
- 同一个实体文字可以在全文出现多次，但每次都必须使用各自正确的 `start/end`。

### 4.8 自有数据的 source 和 license

当前 `prepare_ner_data.py` 最初为 PII Bench ZH 编写，自动发现的 formal/chat 文件会记录为：

```text
pii_bench_zh_formal
pii_bench_zh_chat
```

并从 `configs/data_config.yaml` 的单个 `raw.license` 字段读取许可证。

因此：

1. 只做本地实验时，可以复用 formal/chat 文件名和 schema。
2. 自有数据不能错误标记为 `Apache-2.0`。
3. 若自有数据与 PII Bench 混合，不能用一个全局 license 表示多个来源。
4. 需要正式发布、审计或可追溯训练时，应扩展 `discover_sources()`，让每个 source 分别配置 `name`、`path/pattern` 和 `license`，再生成新版本 processed 数据。

不要在没有确认授权、用途和保留期限时把真实业务 PII 用于训练。

## 5. 文本数据准备与输出

### 5.1 检查原始数据

```powershell
.\.venv\Scripts\python.exe dataset_tools/inspect_dataset.py --raw-dir data/raw
```

检查内容包括：

- JSON/JSONL 是否可解析；
- 实体 span 是否有效；
- 是否重复；
- 是否重叠；
- 字段和实体类型分布。

### 5.2 准备数据

```powershell
.\.venv\Scripts\python.exe training/prepare_ner_data.py `
  --config configs/data_config.yaml
```

脚本生成：

```text
data/processed/train.jsonl
data/processed/validation.jsonl
data/processed/test_fixed.jsonl
data/processed/split_manifest.json
reports/processed_dataset_summary.json
reports/processed_dataset_by_type.csv
reports/processed_dataset_by_source.csv
reports/split_distribution.csv
reports/invalid_annotations.jsonl
reports/conflicting_duplicates.jsonl
```

### 5.3 冻结测试集规则

当 `data/processed/test_fixed.jsonl` 已存在且 `freeze_test: true` 时：

- 脚本复用原测试集；
- 新数据只重新分配到 train/validation；
- 若冻结测试样本已不在原始来源中，脚本报错；
- 不应使用 `--force-resplit` 覆盖已有固定测试集。

如果确实要建立完全独立的自有数据版本，应使用新的输出目录，例如：

先复制 `configs/data_config.yaml` 为 `configs/user_data_config.yaml`，再在副本中修改 `raw.search_root`、`raw.license` 和 `processed.output_dir`。不要直接修改基线配置来伪装数据来源。

```powershell
.\.venv\Scripts\python.exe training/prepare_ner_data.py `
  --config configs/user_data_config.yaml `
  --output-dir data/processed/user_v1
```

不要把新版本冒充当前 `test_fixed`。

## 6. 只用于推理的自有图片

如果只想用现有系统识别和脱敏图片，不需要制作数据集标注。

可选方式：

1. 在 Gradio 图片页直接上传；或
2. 把本地图片放到 `data/images/user/`，再由代码或脚本读取。

推荐结构：

```text
data/images/user/
├── document_0001.png
├── screenshot_0002.jpg
└── scan_0003.jpeg
```

不要把真实姓名作为文件名。推荐使用随机 ID 或业务内部匿名 ID。

图片建议：

- PNG/JPEG，RGB、RGBA 或灰度均可；
- 保留原始分辨率，不要先进行有损压缩；
- EXIF 旋转信息应保留或提前正确旋转；
- 单文件大小应低于应用设置的 20 MB；
- 超大图片会按 `ocr.max_image_side` 缩放后识别，再映射回原图。

输出自动写入 `data/outputs`，不需要把结果移回输入目录。

## 7. 自有带框图片数据

带框图片数据同时包含：图片、完整真值文本、实体字符位置和实体图片框。

### 7.1 推荐目录

```text
data/generated/user/images/
├── user_000001.png
└── user_000002.png

data/generated/annotations_user.jsonl
```

`dataset_tools/inspect_generated_images.py` 会扫描：

```text
data/generated/annotations_*.jsonl
```

因此文件名必须以 `annotations_` 开头并以 `.jsonl` 结尾。

### 7.2 生成图片检查格式

注意这里的图片路径字段叫 `image`：

```json
{
  "id": "user_image_000001",
  "image": "data/generated/user/images/user_000001.png",
  "source": "user_synthetic",
  "template": "custom_form",
  "width": 800,
  "height": 400,
  "text": "赵明宇住在测试市",
  "entities": [
    {
      "type": "person",
      "text": "赵明宇",
      "start": 0,
      "end": 3,
      "boxes": [
        [40, 60, 150, 110]
      ]
    },
    {
      "type": "address",
      "text": "测试市",
      "start": 5,
      "end": 8,
      "boxes": [
        [220, 60, 330, 110]
      ]
    }
  ]
}
```

字段要求：

| 字段 | 必需 | 说明 |
|---|---:|---|
| `id` | 是 | 图片样本唯一 ID |
| `image` | 是 | 相对项目根目录的图片路径 |
| `source` | 推荐 | 数据来源，不要伪装成 PII Bench 或 MultiPriv |
| `template` | 推荐 | 版式/场景名称，用于分组统计 |
| `width`、`height` | 推荐 | 图片像素尺寸，应与实际文件一致 |
| `text` | 是 | 图片中文字的完整真值文本 |
| `entities` | 是 | 实体列表；负样本为空数组 |
| `boxes` | 是 | 每个实体的一个或多个图片框 |

### 7.3 图片框格式

生成图片检查器要求 axis-aligned rectangle：

```text
[x1, y1, x2, y2]
```

约束：

```text
0 <= x1 < x2 <= image_width
0 <= y1 < y2 <= image_height
```

坐标单位是原图像素：

- 原点 `(0, 0)` 位于图片左上角；
- x 向右增加；
- y 向下增加；
- `x2/y2` 是右下边界；
- 框应完整覆盖文字墨迹并保留合理安全边距。

跨行实体使用多个框：

```json
"boxes": [
  [500, 80, 760, 125],
  [40, 130, 180, 175]
]
```

不要用一个巨大矩形跨越两行无关内容。

### 7.4 字符标注和图片框必须一致

每个实体同时满足：

```python
row["text"][entity["start"]:entity["end"]] == entity["text"]
```

以及每个 box 都在图片尺寸内。运行：

```powershell
.\.venv\Scripts\python.exe dataset_tools/inspect_generated_images.py
```

脚本会报告：缺图、无标注图片、span 不匹配、空框、越界框、跨行实体、负样本和模板/类型分布。

## 8. 自有图片评测 manifest

图片评测脚本使用统一 manifest。与上一节不同，这里的路径字段叫 `image_path`，不是 `image`。

推荐放置：

```text
data/annotations/image_dev_user.jsonl
```

示例：

```json
{
  "id": "image_dev_000001",
  "source": "user_annotated_dev",
  "license": "private-internal",
  "annotation_type": "bounding_box",
  "pseudo_label": false,
  "split": "image_dev_user",
  "template": "custom_form",
  "image_path": "data/generated/user/images/user_000001.png",
  "width": 800,
  "height": 400,
  "text": "赵明宇住在测试市",
  "entities": [
    {
      "type": "person",
      "text": "赵明宇",
      "start": 0,
      "end": 3,
      "boxes": [[40, 60, 150, 110]]
    }
  ]
}
```

评测 manifest 的 box 可以是：

```json
[x1, y1, x2, y2]
```

或四边形：

```json
[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
```

当前 IoU/coverage 计算会把 polygon 转为外接矩形。若需要真正 polygon IoU，应新增指标版本，而不是静默改变旧指标。

运行自有开发集评测：

```powershell
.\.venv\Scripts\python.exe evaluation/evaluate_images.py `
  --annotations data/annotations/image_dev_user.jsonl `
  --output-dir reports/image_dev_user
```

不要把自有开发集命名为 `image_test_fixed`，也不要覆盖：

```text
data/annotations/image_test_fixed.jsonl
reports/image_evaluation_summary.json
```

建议始终使用新的 `--output-dir`，避免覆盖冻结基线。

## 9. pseudo label、metadata-only 与人工标注

### 人工框标注

```json
"annotation_type": "bounding_box",
"pseudo_label": false
```

表示实体和框经过人工确认，可用于正式评测，但仍需记录标注规范和质检过程。

### 伪标签

```json
"annotation_type": "pseudo_label",
"pseudo_label": true
```

表示 OCR/规则/模型自动产生的候选，不能与人工真值混合报告，也不能直接加入固定测试集。

### 只有元数据

```json
"annotation_type": "metadata_only",
"pseudo_label": false,
"text": "",
"entities": []
```

表示没有人工实体真值，只能做定性检查或元数据分析，不能计算正式实体 precision/recall/F1。

MultiPriv 当前使用这种模式，并保持：

```text
split: external_test
license: CC BY-NC-SA 4.0
usage_restriction: non-commercial
```

## 10. 本地问题图片回归

如果某张真实图片出现“实体识别正确但框覆盖不完整”，不要把它加入 Git 或仓库测试 fixture。

放置：

```text
data/local_regression/input/problem.png
```

运行：

```powershell
.\.venv\Scripts\python.exe scripts/regression_test_mapping.py `
  --input data/local_regression/input/problem.png `
  --mode strict
```

生成：

```text
data/local_regression/output/ocr_blocks.png
data/local_regression/output/initial_weighted.png
data/local_regression/output/projection_refined.png
data/local_regression/output/component_refined.png
data/local_regression/output/final_boxes.png
data/local_regression/output/redacted.png
data/local_regression/report.json
```

报告只保存输入哈希短 ID、文字长度、坐标、置信度和 fallback 原因，不保存完整 OCR 原文或实体值。

## 11. 数据版本和命名建议

建议每次正式数据更新使用版本目录：

```text
data/raw/user_text_v1/
data/raw/user_text_v2/
data/processed/user_v1/
data/processed/user_v2/
data/annotations/image_dev_user_v1.jsonl
data/annotations/image_dev_user_v2.jsonl
```

每个版本至少记录：

- 数据负责人；
- 来源和采集时间；
- 使用许可或内部授权；
- 标注规范版本；
- 样本数和实体分布；
- 原始文件 SHA-256；
- 去重方式；
- split seed；
- 是否含真实 PII；
- 保留期限和删除流程。

不要通过修改旧文件内容却保留同一版本号的方式更新数据。

## 12. 接入前检查清单

### 文本数据

- [ ] 文件是 JSONL，不是 JSON array、CSV 或 Excel。
- [ ] UTF-8 编码，每行一个 object。
- [ ] `id` 唯一且不含真实姓名/证件号。
- [ ] `text` 是完整字符串。
- [ ] `entities` 始终是 array，负样本使用 `[]`。
- [ ] `start/end` 是 0-based 半开区间 integer。
- [ ] 每个实体满足 `text[start:end] == entity.text`。
- [ ] 实体类型已在 label map 或有明确新增计划。
- [ ] 监督实体不重叠。
- [ ] source、license 和授权信息正确。
- [ ] 没有把固定测试样本复制进训练数据。

### 图片数据

- [ ] 图片可以正常打开，方向正确。
- [ ] manifest 路径相对项目根目录且文件存在。
- [ ] `width/height` 与实际图片一致。
- [ ] 完整真值文本与图片一致。
- [ ] 字符 span 与 entity text 一致。
- [ ] boxes 不为空且不越界。
- [ ] 跨行实体使用多个框。
- [ ] `image` 与 `image_path` 按对应工具使用。
- [ ] 人工标注、伪标签和 metadata-only 没有混合。
- [ ] 真实图片和错误可视化被 Git 忽略。

## 13. 常见错误

### 错误：脚本找不到自有文本文件

原因通常是文件名不匹配 `formal.jsonl`/`chat.jsonl` pattern。修改 pattern 或使用 `--formal`/`--chat`。

### 错误：invalid span or entity text

检查是否按字节数计算了中文位置、是否把 end 写成最后字符下标、是否漏算空格或标点。

### 错误：unknown label

在 `configs/label_map.json` 添加正确 alias，或把自有标注类型统一成当前标准类型。仅添加 `target_labels` 不会自动扩充 NER BIO 分类头。

### 错误：图片检查器提示 missing_image

生成图片标注使用 `image` 字段，路径应相对项目根目录，并使用 `/` 或合法 Windows 路径。

### 错误：图片评测提示 KeyError image_path

评测 manifest 必须使用 `image_path`；`image` 是生成图片检查格式。

### 错误：固定测试集拒绝覆盖

这是预期保护。为自有数据创建新的 manifest 文件名和新的报告目录，不要删除或覆盖固定集。

## 14. 推荐接入流程

```text
确认授权和许可证
  -> 匿名化文件名和样本 ID
  -> 写入 data/raw/user_text 或自有图片目录
  -> 校验 JSONL 和 span/box
  -> 运行 inspect 工具
  -> 检查 invalid/conflict 报告
  -> 输出到新版本 processed/dev manifest
  -> 运行相关单元测试
  -> 先做小规模 smoke train 或少量图片评测
  -> 人工抽样复核
  -> 再决定是否训练新模型或建立独立测试版本
```

数据接入的首要目标是可追溯、可复现和不污染冻结测试集，而不是尽快把所有文件混入现有训练目录。
