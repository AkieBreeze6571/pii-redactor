"""Fine-tune a Hugging Face token-classification model for Chinese NER."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.callbacks import EarlyStopping
from training.dataset import NerDataset, count_bio_classes, load_records
from training.metrics import bio_to_spans, compute_entity_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_labels(path: Path) -> tuple[list[str], dict[str, int], dict[int, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data["bio_labels"]
    if labels[0] != "O" or len(labels) != len(set(labels)):
        raise ValueError("configs/label_map.json 中的 bio_labels 无效")
    label2id = {label: index for index, label in enumerate(labels)}
    return labels, label2id, {index: label for label, index in label2id.items()}


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "pretrained_model_name": ("model", "pretrained_model_name"),
        "max_length": ("model", "max_length"), "dropout": ("model", "dropout"),
        "learning_rate": ("training", "learning_rate"), "batch_size": ("training", "batch_size"),
        "evaluation_batch_size": ("training", "evaluation_batch_size"), "num_epochs": ("training", "num_epochs"),
        "weight_decay": ("training", "weight_decay"), "warmup_ratio": ("training", "warmup_ratio"),
        "gradient_accumulation_steps": ("training", "gradient_accumulation_steps"),
        "max_grad_norm": ("training", "max_grad_norm"), "early_stopping_patience": ("training", "early_stopping_patience"),
        "seed": ("training", "seed"), "fp16": ("training", "fp16"),
        "resume_from_checkpoint": ("training", "resume_from_checkpoint"),
        "use_class_weights": ("loss", "use_class_weights"), "run_name": ("output", "run_name"),
        "max_train_samples": ("data", "max_train_samples"),
        "max_validation_samples": ("data", "max_validation_samples"), "max_test_samples": ("data", "max_test_samples"),
    }
    for argument, path in mapping.items():
        value = getattr(args, argument, None)
        if value is not None:
            config[path[0]][path[1]] = value
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"))
    parser.add_argument("--pretrained-model-name")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--evaluation-batch-size", type=int)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--run-name")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--overwrite-run", action="store_true")
    return parser.parse_args()


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    batch.pop("sample_index", None)
    labels = batch.pop("labels").to(device)
    inputs = {key: value.to(device) for key, value in batch.items()}
    return inputs, labels


def evaluate_model(model: Any, loader: DataLoader, device: torch.device, id2label: dict[int, str]) -> tuple[float, dict[str, Any], list[list[int]], list[list[int]]]:
    model.eval()
    losses: list[float] = []
    predictions: list[list[int]] = []
    references: list[list[int]] = []
    with torch.inference_mode():
        for batch in loader:
            inputs, labels = _move_batch(batch, device)
            outputs = model(**inputs, labels=labels)
            losses.append(float(outputs.loss.detach().cpu()))
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            references.extend(labels.cpu().tolist())
    metrics = compute_entity_metrics(predictions, references, id2label)
    return sum(losses) / max(len(losses), 1), metrics, predictions, references


def compute_class_weights(dataset: NerDataset, label_count: int, method: str) -> torch.Tensor:
    counts = Counter()
    for index in range(len(dataset)):
        counts.update(label for label in dataset[index]["labels"] if label != -100)
    values = []
    for label in range(label_count):
        count = counts[label]
        if method == "effective_number":
            beta = 0.999
            value = (1 - beta) / max(1 - beta ** max(count, 1), 1e-12)
        else:
            value = sum(counts.values()) / max(label_count * count, 1)
        values.append(value)
    weights = torch.tensor(values, dtype=torch.float32)
    return weights / weights.mean()


def save_error_cases(path: Path, records: list[dict[str, Any]], predictions: list[list[int]], references: list[list[int]], id2label: dict[int, str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record, pred_ids, ref_ids in zip(records, predictions, references):
            pairs = [(pred, ref) for pred, ref in zip(pred_ids, ref_ids) if ref != -100]
            truth = bio_to_spans([id2label[ref] for _, ref in pairs])
            prediction = bio_to_spans([id2label[pred] for pred, _ in pairs])
            if truth != prediction:
                error_type = "missed" if truth and not prediction else "false_positive" if prediction and not truth else "boundary/type"
                value = {"id": record.get("id"), "text": record.get("text"), "truth": truth, "prediction": prediction, "error_type": error_type}
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_plots(history: list[dict[str, Any]], metrics: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    epochs = [item["epoch"] for item in history]
    for filename, train_key, validation_key, ylabel in (
        ("loss_curve.png", "train_loss", "validation_loss", "Loss"),
        ("f1_curve.png", None, "validation_macro_f1", "Macro F1"),
    ):
        plt.figure(figsize=(7, 4))
        if train_key:
            plt.plot(epochs, [item[train_key] for item in history], label="train")
        plt.plot(epochs, [item[validation_key] for item in history], label="validation")
        plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.legend(); plt.tight_layout(); plt.savefig(report_dir / filename); plt.close()
    by_type = metrics.get("by_type", {})
    plt.figure(figsize=(7, 4))
    plt.bar(list(by_type), [item["f1"] for item in by_type.values()])
    plt.ylabel("F1"); plt.ylim(0, 1); plt.tight_layout(); plt.savefig(report_dir / "class_f1.png"); plt.close()


def sync_best(run_dir: Path, best_dir: Path) -> None:
    temporary = best_dir.with_name(best_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(run_dir / "model", temporary)
    for path in (run_dir / "tokenizer").iterdir():
        if path.is_file():
            shutil.copy2(path, temporary / path.name)
    shutil.copy2(run_dir / "label_map.json", temporary / "label_map.json")
    shutil.copy2(run_dir / "config.yaml", temporary / "train_config.yaml")
    if best_dir.exists():
        shutil.rmtree(best_dir)
    temporary.replace(best_dir)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parse_args()
    started = time.perf_counter()
    try:
        config = apply_overrides(yaml.safe_load(args.config.read_text(encoding="utf-8")), args)
        output = config["output"]
        checkpoint_root = Path(output["checkpoint_root"]).resolve()
        run_dir = checkpoint_root / output["run_name"]
        if run_dir.exists():
            if not args.overwrite_run:
                raise FileExistsError(f"训练目录已存在：{run_dir}，请更换 run_name 或显式使用 --overwrite-run")
            if checkpoint_root not in run_dir.resolve().parents:
                raise ValueError("拒绝删除 checkpoint_root 之外的目录")
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        report_dir = Path(output["report_dir"]); report_dir.mkdir(parents=True, exist_ok=True)
        labels, label2id, id2label = load_labels(Path("configs/label_map.json"))
        seed = int(config["training"]["seed"]); set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fp16 = bool(config["training"]["fp16"] and device.type == "cuda")
        model_name = config["model"]["pretrained_model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not tokenizer.is_fast:
            raise ValueError("NER 训练必须使用 fast tokenizer")
        model_config = AutoConfig.from_pretrained(model_name, num_labels=len(labels), label2id=label2id, id2label=id2label)
        dropout = float(config["model"]["dropout"])
        for attribute in ("hidden_dropout_prob", "classifier_dropout", "attention_probs_dropout_prob"):
            if hasattr(model_config, attribute): setattr(model_config, attribute, dropout)
        resume = config["training"].get("resume_from_checkpoint")
        model_source = Path(resume) / "model" if resume and (Path(resume) / "model").exists() else resume or model_name
        model = AutoModelForTokenClassification.from_pretrained(model_source, config=model_config, ignore_mismatched_sizes=True).to(device)

        data = config["data"]
        train_records = load_records(data["train_path"], data.get("max_train_samples"))
        validation_records = load_records(data["validation_path"], data.get("max_validation_samples"))
        test_records = load_records(data["test_path"], data.get("max_test_samples"))
        max_length = int(config["model"]["max_length"])
        train_dataset = NerDataset(train_records, tokenizer, label2id, max_length)
        validation_dataset = NerDataset(validation_records, tokenizer, label2id, max_length)
        test_dataset = NerDataset(test_records, tokenizer, label2id, max_length)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_dataset, batch_size=int(config["training"]["batch_size"]), shuffle=True, generator=generator)
        eval_batch = int(config["training"]["evaluation_batch_size"])
        validation_loader = DataLoader(validation_dataset, batch_size=eval_batch)
        test_loader = DataLoader(test_dataset, batch_size=eval_batch)

        optimizer = AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
        accumulation = int(config["training"]["gradient_accumulation_steps"])
        epochs = int(config["training"]["num_epochs"])
        update_steps = math.ceil(len(train_loader) / accumulation) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(update_steps * float(config["training"]["warmup_ratio"])), update_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=fp16)
        class_weights = None
        if config["loss"]["use_class_weights"]:
            class_weights = compute_class_weights(train_dataset, len(labels), config["loss"]["class_weight_method"]).to(device)
        start_epoch = 1
        state_path = Path(resume) / "training_state.pt" if resume else None
        if state_path and state_path.exists():
            state = torch.load(state_path, map_location=device, weights_only=False)
            optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); start_epoch = state["epoch"] + 1

        history: list[dict[str, Any]] = []
        stopper = EarlyStopping(int(config["training"]["early_stopping_patience"]))
        best_epoch = 0
        for epoch in range(start_epoch, epochs + 1):
            model.train(); optimizer.zero_grad(set_to_none=True); running_loss = 0.0
            for step, batch in enumerate(train_loader, start=1):
                inputs, batch_labels = _move_batch(batch, device)
                with torch.amp.autocast("cuda", enabled=fp16):
                    outputs = model(**inputs)
                    if class_weights is None:
                        loss = CrossEntropyLoss(ignore_index=-100)(outputs.logits.view(-1, len(labels)), batch_labels.view(-1))
                    else:
                        loss = CrossEntropyLoss(weight=class_weights, ignore_index=-100)(outputs.logits.view(-1, len(labels)), batch_labels.view(-1))
                    loss = loss / accumulation
                scaler.scale(loss).backward(); running_loss += float(loss.detach().cpu()) * accumulation
                if step % accumulation == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["max_grad_norm"]))
                    scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            validation_loss, validation_metrics, _, _ = evaluate_model(model, validation_loader, device, id2label)
            item = {"epoch": epoch, "train_loss": running_loss / max(len(train_loader), 1), "validation_loss": validation_loss, "validation_macro_f1": validation_metrics["macro_f1"], "validation_micro_f1": validation_metrics["micro_f1"]}
            history.append(item); print(json.dumps(item, ensure_ascii=False))
            improved = validation_metrics["macro_f1"] > stopper.best
            should_stop = stopper.update(validation_metrics["macro_f1"])
            if improved:
                best_epoch = epoch
                (run_dir / "model").mkdir(exist_ok=True); model.save_pretrained(run_dir / "model")
                (run_dir / "tokenizer").mkdir(exist_ok=True); tokenizer.save_pretrained(run_dir / "tokenizer")
                (run_dir / "validation_metrics.json").write_text(json.dumps(validation_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            checkpoint = run_dir / "checkpoint-last"; checkpoint.mkdir(exist_ok=True)
            model.save_pretrained(checkpoint / "model")
            torch.save({"epoch": epoch, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, checkpoint / "training_state.pt")
            if should_stop: break

        best_model = AutoModelForTokenClassification.from_pretrained(run_dir / "model").to(device)
        validation_loss, validation_metrics, _, _ = evaluate_model(best_model, validation_loader, device, id2label)
        test_loss, test_metrics, test_predictions, test_references = evaluate_model(best_model, test_loader, device, id2label)
        (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        actual_config = config | {"runtime": {"device": str(device), "fp16_effective": fp16, "best_epoch": best_epoch, "training_seconds": time.perf_counter() - started}}
        (run_dir / "config.yaml").write_text(yaml.safe_dump(actual_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        shutil.copy2("configs/label_map.json", run_dir / "label_map.json")
        manifest = Path("data/processed/split_manifest.json")
        data_summary = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        data_summary["used_counts"] = {"train": len(train_records), "validation": len(validation_records), "test_fixed": len(test_records)}
        data_summary["training_entity_counts"] = dict(count_bio_classes(train_records, label2id))
        (run_dir / "data_summary.json").write_text(json.dumps(data_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        environment = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "transformers": __import__("transformers").__version__, "device": str(device), "cuda_available": torch.cuda.is_available(), "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        (run_dir / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
        with (run_dir / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
        shutil.copy2(run_dir / "training_history.csv", report_dir / "training_history.csv")
        evaluation = {"validation": validation_metrics, "test_fixed": test_metrics, "best_epoch": best_epoch}
        (report_dir / "evaluation_summary.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
        with (report_dir / "evaluation_by_type.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(["split", "type", "precision", "recall", "f1", "tp", "fp", "fn"])
            for split, values in (("validation", validation_metrics), ("test_fixed", test_metrics)):
                for entity_type, item in values["by_type"].items(): writer.writerow([split, entity_type, item["precision"], item["recall"], item["f1"], item["tp"], item["fp"], item["fn"]])
        save_error_cases(report_dir / "error_cases.jsonl", test_records, test_predictions, test_references, id2label)
        write_plots(history, test_metrics, report_dir)
        sync_best(run_dir, Path(output["best_model_dir"]))
        print(json.dumps({"run_dir": str(run_dir), "best_epoch": best_epoch, "device": str(device), "validation_macro_f1": validation_metrics["macro_f1"], "test_macro_f1": test_metrics["macro_f1"], "training_seconds": actual_config["runtime"]["training_seconds"]}, ensure_ascii=False, indent=2))
        return 0
    except torch.OutOfMemoryError:
        print("CUDA 显存不足：请减小 batch_size 或 max_length，或增加 gradient_accumulation_steps。", file=sys.stderr)
        return 3
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"NER 训练失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
