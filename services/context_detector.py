"""Conservative context extraction and organization subtype refinement."""

from __future__ import annotations

import re
from typing import Any


NAME_KEYS = "姓名|联系人|收件人|寄件人|患者|学生|老师|负责人|申请人|法定代表人|持卡人|监护人"
ADDRESS_KEYS = "地址|住址|收货地址|寄送地址|联系地址|家庭住址|户籍地址|现居地址|单位地址|注册地址"
NAME_RE = re.compile(rf"(?:{NAME_KEYS})\s*[：:]?\s*([\u4e00-\u9fff]{{2,4}})")
ADDRESS_RE = re.compile(rf"(?:{ADDRESS_KEYS})\s*[：:]?\s*([^。；;\n]{{4,80}})")
BOUNDARY_RE = re.compile(r"[。；;\n]")
SCHOOL_SUFFIXES = ("大学", "学院", "中学", "小学", "学校", "培训中心", "研究院")
HOSPITAL_SUFFIXES = ("医院", "诊所", "卫生院", "医疗中心", "妇幼保健院")
COMPANY_SUFFIXES = ("公司", "集团", "企业", "厂", "有限责任公司", "股份有限公司")
ADDRESS_SUFFIXES = ("省", "自治区", "市", "自治州", "区", "县", "旗", "镇", "乡", "村", "街道", "路", "街", "巷", "弄", "号", "栋", "幢", "单元", "室", "小区", "社区", "大厦", "广场")


def classify_organization(text: str) -> str:
    if any(suffix in text for suffix in SCHOOL_SUFFIXES):
        return "school"
    if any(suffix in text for suffix in HOSPITAL_SUFFIXES):
        return "hospital"
    if any(suffix in text for suffix in COMPANY_SUFFIXES):
        return "company"
    return "organization"


class ContextDetector:
    def detect(self, text: str) -> list[dict[str, Any]]:
        if not text:
            return []
        output = []
        for match in NAME_RE.finditer(text):
            start, end = match.span(1)
            output.append(self._entity("person", text, start, end, 0.72))
        for match in ADDRESS_RE.finditer(text):
            value = match.group(1).rstrip("，, ")
            start = match.start(1); end = start + len(value)
            if any(suffix in value for suffix in ADDRESS_SUFFIXES):
                output.append(self._entity("address", text, start, end, 0.68))
        return output

    def enhance(self, text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enhanced = []
        for original in entities:
            item = dict(original)
            item.setdefault("sources", [item.get("source", "ner")])
            if item["type"] == "organization":
                refined = classify_organization(item["text"])
                if refined != "organization":
                    item["original_type"] = "organization"
                    item["type"] = refined
                    item["source"] = "context"
                    item["sources"] = ["ner", "context"]
                    item["confidence"] = float(item["confidence"]) * 0.95
            elif item["type"] == "address":
                extended_end = self._extend_address(text, item["end"])
                if extended_end > item["end"]:
                    item["end"] = extended_end
                    item["text"] = text[item["start"]:extended_end]
                    item["source"] = "context"
                    item["sources"] = ["ner", "context"]
                    item["confidence"] = float(item["confidence"]) * 0.95
            enhanced.append(item)
        return enhanced

    @staticmethod
    def _extend_address(text: str, end: int) -> int:
        boundary = BOUNDARY_RE.search(text, end)
        limit = boundary.start() if boundary else min(len(text), end + 40)
        candidate = text[end:limit]
        best = end
        for suffix in ADDRESS_SUFFIXES:
            for match in re.finditer(re.escape(suffix), candidate):
                best = max(best, end + match.end())
        return best

    @staticmethod
    def _entity(entity_type: str, text: str, start: int, end: int, confidence: float) -> dict[str, Any]:
        return {
            "type": entity_type, "text": text[start:end], "normalized_text": text[start:end],
            "start": start, "end": end, "confidence": confidence, "source": "context",
            "sources": ["context"], "validation": "not_applicable",
        }
