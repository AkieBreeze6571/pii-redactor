# 系统架构

系统采用本地模块化流水线：`ImageService` 规范化输入，`OcrService` 延迟加载 PaddleOCR 并缓存识别结果，`HybridDetector` 融合规则、MacBERT NER 和上下文候选，`CoordinateMapper` 将文本 span 映射回 OCR 多边形，`RedactionService` 生成预览、结果图和 Mask。

`DocumentProcessor` 负责端到端编排和分阶段耗时。Gradio UI 仅调用服务层，不在组件回调内实现检测算法。`DatabaseService` 使用 SQLite 保存可检索的处理元数据，`BatchService` 隔离单文件失败并打包产物。

主要边界：OCR 文本误差会传递到检测；字符坐标先按字符类别权重估算，再尝试透视矫正后的墨迹投影与连通域修正，但仍不等同于字形级检测框。严格模式在边界或覆盖置信度不足时遮挡完整 OCR block；最终结果保留来源、置信度、验证状态、映射策略和降级原因。
