from datasets import load_dataset


def main():
    formal = load_dataset(
        "wan9yu/pii-bench-zh",
        data_files="data/pii_bench_zh.jsonl",
        split="train"
    )

    chat = load_dataset(
        "wan9yu/pii-bench-zh",
        data_files="data/pii_bench_zh_chat.jsonl",
        split="train"
    )

    formal.to_json(
        "data/raw/formal.jsonl",
        orient="records",
        lines=True,
        force_ascii=False
    )

    chat.to_json(
        "data/raw/chat.jsonl",
        orient="records",
        lines=True,
        force_ascii=False
    )

    print("formal:", len(formal))
    print("chat:", len(chat))
    print(formal[0])


if __name__ == "__main__":
    main()