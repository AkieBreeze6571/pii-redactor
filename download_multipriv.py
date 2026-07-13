from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "multipriv"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_path = snapshot_download(
        repo_id="CyberChangAn/MultiPriv",
        repo_type="dataset",
        local_dir=OUTPUT_DIR,
        allow_patterns=[
            "VLM/individual-level/**/*.png",
            "VLM/individual-level/**/*.jpg",
            "VLM/individual-level/**/*.jpeg",
            "VLM/individual-level/**/*.json",

            "VLM/attribute-level/Publicly_available/**/*",
            "VLM/attribute-level/Non_public_data_URL.xlsx",

            "README.md",
            "LICENSE"
        ]
    )

    print("MultiPriv 下载完成")
    print(f"保存位置：{downloaded_path}")


if __name__ == "__main__":
    main()