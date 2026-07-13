import json
from pathlib import Path

paths = [
    Path("data/raw/formal.jsonl"),
    Path("data/raw/chat.jsonl")
]

for path in paths:
    print("=" * 60)
    print(path)

    with path.open("r", encoding="utf-8") as file:
        for _ in range(2):
            row = json.loads(next(file))
            print(json.dumps(row, ensure_ascii=False, indent=2))