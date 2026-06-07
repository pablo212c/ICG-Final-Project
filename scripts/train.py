from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import argparse
import csv
from copy import deepcopy
from contextlib import nullcontext
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.config import merged_defaults
from src.data import ATTRIBUTES, AADBDataset
from src.inference import load_checkpoint, resolve_device
from src.metrics import format_group_metrics, format_metrics, group_ranking_metrics, regression_metrics
from src.model import create_model
from src.transforms import EvalTTATransform, ImageTransform


TRAIN_DEFAULTS = {
    "images": "datasetImages_warp256",
    "labels": "imgListFiles_label",
    "output": "checkpoints/best.pt",
    "checkpoint_policy": "auto",
    "checkpoint": None,
    "test": False,
    "split": "test",
    "predictions_csv": None,
    "backbone": "convnext_small",
    "pretrained": True,
    "epochs": 30,
    "early_stop_patience": 0,
    "batch_size": 32,
    "eval_batch_size": None,
    "lr": 1e-4,
    "head_lr_multiplier": 5.0,
    "weight_decay": 1e-4,
    "attr_loss_weight": 0.2,
    "rank_loss_weight": 0.5,
    "rank_loss": "hinge",
    "rank_margin": 0.05,
    "rank_min_diff": 0.05,
    "rank_temperature": 0.05,
    "rank_weight_by_diff": False,
    "listwise_loss_weight": 0.0,
    "listwise_temperature": 0.08,
    "score_loss": "smooth_l1",
    "score_loss_beta": 0.05,
    "dropout": 0.2,
    "head_hidden_dim": 256,
    "stochastic_depth_scale": 1.0,
    "image_size": 224,
    "resize_size": 256,
    "num_workers": 0,
    "device": "auto",
    "seed": 42,
    "scheduler": "cosine",
    "warmup_epochs": 0,
    "min_lr": 1e-6,
    "max_grad_norm": 1.0,
    "tail_weight": 1.0,
    "tail_low": 0.35,
    "tail_high": 0.70,
    "freeze_backbone_epochs": 1,
    "trainable_backbone_from": None,
    "ema_decay": 0.0,
    "amp": True,
    "group_eval": False,
    "val_group_eval": False,
    "group_size": 5,
    "group_count": 1000,
    "selection_metric": "spearman",
    "tta_views": 1,
    "val_tta_views": 1,
    "log_every": 0,
    "limit_train": None,
    "limit_val": None,
    "limit_test": None,
}

TRAIN_CONFIG = "configs/ema_s.json"
TEST_CONFIG = "configs/test_tta.json"

GROUP_SELECTION_METRICS = {
    "group_spearman",
    "group_kendall",
    "pairwise_accuracy",
    "top1_accuracy",
    "group_rank_score",
}


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_parser.add_argument("--test", action="store_true")
    config_args, _ = pre_parser.parse_known_args()
    config_path = config_args.config or (TEST_CONFIG if config_args.test else TRAIN_CONFIG)
    defaults = merged_defaults(TRAIN_DEFAULTS, config_path, set(TRAIN_DEFAULTS))
    if config_args.test:
        defaults["test"] = True

    parser = argparse.ArgumentParser(
        description="Train or test the AADB photo aesthetic ranking model.",
    )
    parser.add_argument("--config", default=config_path, help="JSON config file.")
    parser.add_argument("--images", default=defaults["images"], help="Directory containing AADB images.")
    parser.add_argument("--labels", default=defaults["labels"], help="AADB imgListFiles_label directory.")
    parser.add_argument("--output", default=defaults["output"], help="Checkpoint path for training.")
    parser.add_argument(
        "--checkpoint-policy",
        default=defaults["checkpoint_policy"],
        choices=["auto", "overwrite", "error"],
        help="'auto' writes best2.pt/best3.pt when output exists; 'overwrite' reuses the path.",
    )
    parser.add_argument("--checkpoint", default=defaults["checkpoint"], help="Checkpoint path for --test.")
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        default=defaults["test"],
        help="Run evaluation only.",
    )
    parser.add_argument(
        "--split",
        default=defaults["split"],
        choices=["validation", "val", "test", "testnew", "test_new"],
    )
    parser.add_argument("--predictions-csv", default=defaults["predictions_csv"], help="Optional CSV path for test predictions.")
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=defaults["eval_batch_size"])
    parser.add_argument("--num-workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument(
        "--group-eval",
        action=argparse.BooleanOptionalAction,
        default=defaults["group_eval"],
        help="Report sampled 2-5 image group ranking metrics during test.",
    )
    parser.add_argument(
        "--val-group-eval",
        action=argparse.BooleanOptionalAction,
        default=defaults["val_group_eval"],
        help="Compute sampled group ranking metrics on validation during training.",
    )
    parser.add_argument("--group-size", type=int, default=defaults["group_size"])
    parser.add_argument("--group-count", type=int, default=defaults["group_count"])
    parser.add_argument(
        "--selection-metric",
        default=defaults["selection_metric"],
        choices=[
            "spearman",
            "kendall",
            "rank_mean",
            "group_spearman",
            "group_kendall",
            "pairwise_accuracy",
            "top1_accuracy",
            "group_rank_score",
        ],
        help="Validation metric used to save the best checkpoint.",
    )
    parser.add_argument("--tta-views", type=int, default=defaults["tta_views"], help="Use deterministic test-time augmentation views during evaluation.")
    parser.add_argument("--limit-train", type=int, default=defaults["limit_train"], help="Debug: limit train samples.")
    parser.add_argument("--limit-val", type=int, default=defaults["limit_val"], help="Debug: limit validation samples.")
    parser.add_argument("--limit-test", type=int, default=defaults["limit_test"], help="Debug: limit test samples.")
    args = parser.parse_args()
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    image_root: str,
    label_root: str,
    split: str,
    image_size: int,
    resize_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    limit: int | None = None,
    tta_views: int = 1,
    tail_weight: float = 1.0,
    tail_low: float = 0.35,
    tail_high: float = 0.70,
) -> DataLoader:
    transform = ImageTransform(image_size=image_size, resize_size=resize_size, train=train)
    if not train and tta_views > 1:
        transform = EvalTTATransform(image_size=image_size, resize_size=resize_size, views=tta_views)
    dataset = AADBDataset(
        image_root=image_root,
        label_root=label_root,
        split=split,
        transform=transform,
        limit=limit,
    )
    sampler = None
    shuffle = train
    if train and tail_weight > 1.0:
        weights = []
        tail_count = 0
        for record in dataset.records:
            is_tail = record.score <= tail_low or record.score >= tail_high
            weights.append(float(tail_weight if is_tail else 1.0))
            tail_count += int(is_tail)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
        print(
            f"tail-balanced sampler: tail_samples={tail_count}/{len(weights)}, "
            f"tail_weight={tail_weight}, low<={tail_low}, high>={tail_high}"
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def resolve_checkpoint_output_path(path: str | Path, policy: str) -> Path:
    output_path = Path(path)
    if policy == "overwrite" or not output_path.exists():
        return output_path
    if policy == "error":
        raise FileExistsError(
            f"Checkpoint already exists: {output_path}. Use --checkpoint-policy overwrite or auto."
        )
    if policy != "auto":
        raise ValueError(f"Unknown checkpoint policy: {policy}")

    parent = output_path.parent
    stem = output_path.stem
    suffix = output_path.suffix
    for index in range(2, 10000):
        candidate = parent / f"{stem}{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an unused checkpoint path for {output_path}")


def make_score_loss(name: str, beta: float) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    try:
        return nn.SmoothL1Loss(beta=beta)
    except TypeError:
        return nn.SmoothL1Loss()


def pairwise_rank_loss(
    pred_scores: torch.Tensor,
    true_scores: torch.Tensor,
    margin: float,
    min_diff: float,
    loss_type: str,
    temperature: float,
    weight_by_diff: bool,
) -> torch.Tensor:
    true_diff = true_scores.unsqueeze(1) - true_scores.unsqueeze(0)
    pred_diff = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(0)
    mask = true_diff > min_diff
    if not torch.any(mask):
        return pred_scores.new_tensor(0.0)

    selected_pred_diff = pred_diff[mask]
    if loss_type == "hinge":
        losses = torch.relu(margin - selected_pred_diff)
    elif loss_type == "logistic":
        scale = max(float(temperature), 1e-6)
        losses = F.softplus((margin - selected_pred_diff) / scale) * scale
    else:
        raise ValueError(f"Unknown rank loss: {loss_type}")

    if weight_by_diff:
        weights = torch.clamp(true_diff[mask] / max(float(min_diff), 1e-6), min=1.0, max=4.0)
        losses = losses * (weights / weights.mean().clamp_min(1e-6))
    return losses.mean()


def listwise_rank_loss(
    pred_scores: torch.Tensor,
    true_scores: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if pred_scores.numel() < 2 or torch.std(true_scores) == 0:
        return pred_scores.new_tensor(0.0)
    scale = max(float(temperature), 1e-6)
    target_distribution = torch.softmax(true_scores.detach() / scale, dim=0)
    log_pred_distribution = torch.log_softmax(pred_scores / scale, dim=0)
    return -(target_distribution * log_pred_distribution).sum()


def set_module_trainable(module: torch.nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def set_backbone_trainable(
    model: torch.nn.Module,
    trainable: bool,
    trainable_from: int | None = None,
) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    if not trainable:
        return

    if trainable_from is not None and int(trainable_from) < 0:
        return

    if trainable_from is None:
        set_module_trainable(model.backbone, True)
        return

    features = getattr(model.backbone, "features", None)
    if isinstance(features, nn.Sequential):
        start = max(0, min(int(trainable_from), len(features)))
        for module in features[start:]:
            set_module_trainable(module, True)
        for name in ("classifier", "fc", "head", "heads"):
            module = getattr(model.backbone, name, None)
            if isinstance(module, nn.Module):
                set_module_trainable(module, True)
        return

    set_module_trainable(model.backbone, True)


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.module = deepcopy(model).eval()
        self.decay = float(decay)
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        model_state = model.state_dict()
        for name, ema_value in self.module.state_dict().items():
            model_value = model_state[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


def make_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    head_params = list(model.score_head.parameters()) + list(model.attribute_head.parameters())
    head_param_ids = {id(parameter) for parameter in head_params}
    backbone_params = [parameter for parameter in model.parameters() if id(parameter) not in head_param_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * args.head_lr_multiplier},
        ],
        weight_decay=args.weight_decay,
    )


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        warmup_epochs: int,
        min_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.total_epochs = max(total_epochs, 1)
        self.warmup_epochs = max(warmup_epochs, 0)
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.epoch = 0
        if self.warmup_epochs > 0:
            self._set_scale(1.0 / max(self.warmup_epochs, 1))

    def step(self) -> None:
        self.epoch += 1
        if self.warmup_epochs > 0 and self.epoch < self.warmup_epochs:
            scale = float(self.epoch + 1) / float(self.warmup_epochs)
            self._set_scale(scale)
            return

        progress = (self.epoch - self.warmup_epochs + 1) / max(self.total_epochs - self.warmup_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self.min_lr + (base_lr - self.min_lr) * cosine

    def _set_scale(self, scale: float) -> None:
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = max(base_lr * scale, self.min_lr)


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.cuda.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type=device.type, enabled=enabled)
        except TypeError:
            pass
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    attr_loss_weight: float,
    rank_loss_weight: float,
    rank_loss_type: str,
    rank_margin: float,
    rank_min_diff: float,
    rank_temperature: float,
    rank_weight_by_diff: bool,
    listwise_loss_weight: float,
    listwise_temperature: float,
    score_loss_fn: nn.Module,
    scaler,
    max_grad_norm: float,
    log_every: int,
    ema: ModelEMA | None,
) -> float:
    model.train()
    attr_loss_fn = nn.MSELoss()
    total_loss = 0.0
    total_items = 0

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        scores = batch["score"].to(device)
        attributes = batch["attributes"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, scaler.is_enabled()):
            output = model(images)
            score_loss = score_loss_fn(output["score"], scores)
            attr_loss = attr_loss_fn(output["attributes"], attributes)
            rank_loss = pairwise_rank_loss(
                output["score"],
                scores,
                margin=rank_margin,
                min_diff=rank_min_diff,
                loss_type=rank_loss_type,
                temperature=rank_temperature,
                weight_by_diff=rank_weight_by_diff,
            )
            list_loss = listwise_rank_loss(output["score"], scores, listwise_temperature)
            loss = (
                score_loss
                + attr_loss_weight * attr_loss
                + rank_loss_weight * rank_loss
                + listwise_loss_weight * list_loss
            )

        scaler.scale(loss).backward()
        if max_grad_norm and max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        if log_every > 0 and step % log_every == 0:
            print(f"  step {step:04d}/{len(loader):04d}, loss={float(loss.item()):.4f}", flush=True)
    return total_loss / max(total_items, 1)


def forward_views(model: torch.nn.Module, images: torch.Tensor) -> dict[str, torch.Tensor]:
    if images.ndim == 5:
        batch_size, views, channels, height, width = images.shape
        flat_images = images.reshape(batch_size * views, channels, height, width)
        output = model(flat_images)
        score = output["score"].reshape(batch_size, views).mean(dim=1)
        attributes = output["attributes"].reshape(batch_size, views, -1).mean(dim=1)
        return {"score": score, "attributes": attributes}
    return model(images)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    y_true: list[float] = []
    y_pred: list[float] = []
    attr_abs_errors: list[float] = []
    rows: list[dict[str, object]] = []

    for batch in loader:
        images = batch["image"].to(device)
        scores = batch["score"].to(device)
        attributes = batch["attributes"].to(device)
        output = forward_views(model, images)

        pred_scores = output["score"].detach().cpu().numpy()
        true_scores = scores.detach().cpu().numpy()
        pred_attrs = output["attributes"].detach().cpu().numpy()
        true_attrs = attributes.detach().cpu().numpy()
        filenames = batch["filename"]

        y_true.extend(true_scores.tolist())
        y_pred.extend(pred_scores.tolist())
        attr_abs_errors.extend(np.abs(pred_attrs - true_attrs).reshape(-1).tolist())

        for index, filename in enumerate(filenames):
            row: dict[str, object] = {
                "filename": filename,
                "true_score": float(true_scores[index]),
                "pred_score": float(pred_scores[index]),
            }
            for attr_index, attr_name in enumerate(ATTRIBUTES):
                row[f"true_attr_{attr_name}"] = float(true_attrs[index, attr_index])
                row[f"pred_attr_{attr_name}"] = float(pred_attrs[index, attr_index])
            rows.append(row)

    metrics = regression_metrics(y_true, y_pred)
    metrics["attr_mae"] = float(np.mean(attr_abs_errors)) if attr_abs_errors else float("nan")
    return metrics, rows


def save_predictions(path: str | Path, rows: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_group_metrics(
    metrics: dict[str, float],
    rows: list[dict[str, object]],
    group_size: int,
    group_count: int,
    seed: int,
) -> dict[str, float]:
    group_metrics = group_ranking_metrics(
        rows,
        group_size=group_size,
        group_count=group_count,
        seed=seed,
    )
    metrics.update(group_metrics)
    metrics["group_rank_score"] = 0.5 * (
        float(group_metrics.get("pairwise_accuracy", float("nan")))
        + float(group_metrics.get("top1_accuracy", float("nan")))
    )
    return group_metrics


def uses_validation_group_metrics(args: argparse.Namespace) -> bool:
    return bool(args.val_group_eval or args.selection_metric in GROUP_SELECTION_METRICS)


def checkpoint_selection_value(metrics: dict[str, float], name: str) -> float:
    if name == "rank_mean":
        return 0.5 * (
            float(metrics.get("spearman", float("nan")))
            + float(metrics.get("kendall", float("nan")))
        )
    return float(metrics.get(name, float("nan")))


def run_test(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint or args.output
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    image_size = int(checkpoint.get("image_size", args.image_size))
    resize_size = int(checkpoint.get("resize_size", args.resize_size))
    eval_batch_size = args.eval_batch_size or args.batch_size

    loader = make_loader(
        args.images,
        args.labels,
        args.split,
        image_size,
        resize_size,
        eval_batch_size,
        args.num_workers,
        train=False,
        limit=args.limit_test,
        tta_views=args.tta_views,
    )
    metrics, rows = evaluate(model, loader, device)
    print(f"{args.split} metrics: {format_metrics(metrics)}")
    if args.group_eval:
        group_metrics = add_group_metrics(metrics, rows, args.group_size, args.group_count, args.seed)
        print(
            f"group metrics (size={args.group_size}, n={args.group_count}): "
            f"{format_group_metrics(group_metrics)}, "
            f"group_rank_score={metrics['group_rank_score']:.4f}"
        )
    if args.predictions_csv:
        save_predictions(args.predictions_csv, rows)
        print(f"wrote predictions: {args.predictions_csv}")


def run_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}")
    print(
        f"training config: epochs={args.epochs}, batch_size={args.batch_size}, "
        f"eval_batch_size={args.eval_batch_size or args.batch_size}, "
        f"backbone={args.backbone}, pretrained={args.pretrained}, output={args.output}, "
        f"checkpoint_policy={args.checkpoint_policy}, ema_decay={args.ema_decay}, "
        f"val_tta_views={args.val_tta_views}, selection_metric={args.selection_metric}, "
        f"rank_loss_weight={args.rank_loss_weight}, listwise_loss_weight={args.listwise_loss_weight}, "
        f"val_group_eval={uses_validation_group_metrics(args)}, "
        f"trainable_backbone_from={args.trainable_backbone_from}"
    )
    eval_batch_size = args.eval_batch_size or args.batch_size

    train_loader = make_loader(
        args.images,
        args.labels,
        "train",
        args.image_size,
        args.resize_size,
        args.batch_size,
        args.num_workers,
        train=True,
        limit=args.limit_train,
        tail_weight=args.tail_weight,
        tail_low=args.tail_low,
        tail_high=args.tail_high,
    )
    val_loader = make_loader(
        args.images,
        args.labels,
        "validation",
        args.image_size,
        args.resize_size,
        eval_batch_size,
        args.num_workers,
        train=False,
        limit=args.limit_val,
        tta_views=args.val_tta_views,
    )
    print(
        f"data: train_samples={len(train_loader.dataset)}, train_batches={len(train_loader)}, "
        f"val_samples={len(val_loader.dataset)}, val_batches={len(val_loader)}"
    )

    model = create_model(
        backbone=args.backbone,
        pretrained=args.pretrained,
        attr_count=len(ATTRIBUTES),
        dropout=args.dropout,
        head_hidden_dim=args.head_hidden_dim,
        stochastic_depth_scale=args.stochastic_depth_scale,
        image_size=args.image_size,
    ).to(device)
    print(f"pretrained backbone loaded: {getattr(model, 'pretrained_loaded', False)}")
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay and args.ema_decay > 0 else None
    if ema is not None:
        print(f"EMA enabled: decay={args.ema_decay}")
    optimizer = make_optimizer(args, model)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = WarmupCosineScheduler(
            optimizer,
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            min_lr=args.min_lr,
        )
    score_loss_fn = make_score_loss(args.score_loss, args.score_loss_beta)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = make_grad_scaler(amp_enabled)

    best_selection_value = float("-inf")
    epochs_without_improvement = 0
    output_path = resolve_checkpoint_output_path(args.output, args.checkpoint_policy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path != Path(args.output):
        print(f"checkpoint exists, using auto-versioned output: {output_path}")
    else:
        print(f"checkpoint output: {output_path}")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch:03d}/{args.epochs:03d}", flush=True)
        freeze_backbone = bool(
            getattr(model, "pretrained_loaded", False) and epoch <= args.freeze_backbone_epochs
        )
        set_backbone_trainable(model, not freeze_backbone, args.trainable_backbone_from)
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.attr_loss_weight,
            args.rank_loss_weight,
            args.rank_loss,
            args.rank_margin,
            args.rank_min_diff,
            args.rank_temperature,
            args.rank_weight_by_diff,
            args.listwise_loss_weight,
            args.listwise_temperature,
            score_loss_fn,
            scaler,
            args.max_grad_norm,
            args.log_every,
            ema,
        )
        eval_model = ema.module if ema is not None else model
        val_metrics, val_rows = evaluate(eval_model, val_loader, device)
        val_group_metrics = None
        if uses_validation_group_metrics(args):
            val_group_metrics = add_group_metrics(
                val_metrics,
                val_rows,
                args.group_size,
                args.group_count,
                args.seed,
            )
        current_lr = optimizer.param_groups[0]["lr"]
        log_line = (
            f"train_loss={train_loss:.4f}, lr={current_lr:.2e}, "
            f"backbone={'frozen' if freeze_backbone else 'trainable'}, "
            f"val {format_metrics(val_metrics)}"
        )
        if val_group_metrics is not None:
            log_line += (
                f", val_group {format_group_metrics(val_group_metrics)}, "
                f"group_rank_score={val_metrics['group_rank_score']:.4f}"
            )
        print(log_line)
        if scheduler is not None:
            scheduler.step()

        current_selection_value = checkpoint_selection_value(val_metrics, args.selection_metric)
        if np.isnan(current_selection_value):
            current_selection_value = float("-inf")
        if current_selection_value >= best_selection_value:
            best_selection_value = current_selection_value
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": eval_model.state_dict(),
                    "backbone": args.backbone,
                    "attr_names": ATTRIBUTES,
                    "dropout": args.dropout,
                    "head_hidden_dim": args.head_hidden_dim,
                    "stochastic_depth_scale": args.stochastic_depth_scale,
                    "pretrained_loaded": getattr(model, "pretrained_loaded", False),
                    "ema_decay": args.ema_decay,
                    "checkpoint_uses_ema": ema is not None,
                    "image_size": args.image_size,
                    "resize_size": args.resize_size,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "selection_metric": args.selection_metric,
                    "selection_value": current_selection_value,
                    "args": vars(args),
                },
                output_path,
            )
            print(f" * best checkpoint saved")
        else:
            epochs_without_improvement += 1
            if args.early_stop_patience and epochs_without_improvement >= args.early_stop_patience:
                print(
                    f"early stopping: no validation {args.selection_metric} improvement "
                    f"for {epochs_without_improvement} epochs"
                )
                break


def main() -> None:
    args = parse_args()
    if args.test:
        run_test(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
