# 数据流程

文本原始数据位于 `data/raw/formal.jsonl` 和 `data/raw/chat.jsonl`。`training/prepare_ner_data.py` 负责校验 span、去重、分层划分并生成 `data/processed` 下的训练、验证和固定测试集。固定测试集及其 manifest 一旦生成，不应因调参而改写。

合成图片及标注通过 `dataset_tools/build_manifest.py` 建立统一清单，`dataset_tools/inspect_generated_images.py` 检查图片存在性、尺寸、空框、越界框、模板和实体分布。固定图片测试清单为 `data/annotations/image_test_fixed.jsonl`，包含生成器哈希和来源 ID，脚本遇到内容漂移会拒绝静默覆盖。

MultiPriv 仅建立外部测试元数据清单，不生成伪标签。其许可为 `CC BY-NC-SA 4.0`，不得用于商业用途；任何后续下载、标注和发布都必须保留来源及许可信息。

自有文本、图片和标注的目录、JSONL schema、span/box 规范及验证命令见 [数据集详解与自有数据接入指南](dataset_input_guide.md)。
