from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import argparse
import json
from pathlib import Path

from src.color_harmony import analyze_image
from src.config import merged_defaults
from src.data import ATTRIBUTE_DISPLAY_NAMES, ATTRIBUTES
from src.inference import load_checkpoint, predict_image, resolve_device, resolve_image_path

RANK_DEFAULTS = {
    "images": None,
    "demo": None,
    "demo_dir": "demo_img",
    "checkpoint": "checkpoints/best.pt",
    "image_root": "img",
    "device": "auto",
    "tta_views": 10,
    "json": False,
    "llm_input": "outputs/rank_llm_input.json",
}


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="JSON config file.")
    config_args, _ = config_parser.parse_known_args()
    defaults = merged_defaults(RANK_DEFAULTS, config_args.config, set(RANK_DEFAULTS))

    parser = argparse.ArgumentParser(
        description="Rank 2 to 5 AADB photos with aesthetic and harmony features.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=defaults["images"],
        required=False,
        help="Image filenames or paths, min 2 and max 5.",
    )
    parser.add_argument(
        "--demo",
        nargs="+",
        default=defaults["demo"],
        help="Use demo image aliases: all, img1, or 1. Example: --demo img1 img2 img5.",
    )
    parser.add_argument("--demo-dir", default=defaults["demo_dir"])
    parser.add_argument("--checkpoint", default=defaults["checkpoint"])
    parser.add_argument("--image-root", default=defaults["image_root"])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--tta-views", type=int, default=defaults["tta_views"])
    parser.add_argument("--json", action=argparse.BooleanOptionalAction, default=defaults["json"], help="Print JSON instead of text.")
    parser.add_argument(
        "--llm-input",
        default=defaults["llm_input"],
        help="Path to write structured JSON for downstream LLM explanation.",
    )
    args = parser.parse_args()
    if args.demo and args.images:
        parser.error("Use either --demo or --images, not both.")
    if args.demo:
        args.images = resolve_demo_images(args.demo, args.demo_dir)
        args.image_root = "."
    if not args.images:
        parser.error("Provide --images or --demo.")
    return args


def resolve_demo_images(tokens: list[str], demo_dir: str) -> list[str]:
    aliases = [token.lower().strip() for token in tokens]
    demo_path = Path(demo_dir)
    if len(aliases) == 1 and aliases[0] == "all":
        images = sorted(demo_path.glob("img*.jpg"))
        if not images:
            raise SystemExit(f"No demo images found under {demo_dir}.")
        return [str(path) for path in images]

    resolved: list[str] = []
    for alias in aliases:
        if alias.startswith("image"):
            suffix = alias.removeprefix("image")
        elif alias.startswith("img"):
            suffix = alias.removeprefix("img")
        else:
            suffix = alias
        if not suffix.isdigit():
            raise SystemExit(f"Invalid demo image alias: {alias}. Use img1, 1, or all.")
        image_path = demo_path / f"img{int(suffix)}.jpg"
        if not image_path.exists():
            raise SystemExit(f"Demo image not found: {image_path}")
        resolved.append(str(image_path))

    if len(set(resolved)) != len(resolved):
        raise SystemExit("Duplicate demo images are not allowed.")
    return resolved


def _attr_dict(values: list[float]) -> dict[str, float]:
    return {name: float(values[index]) for index, name in enumerate(ATTRIBUTES)}


def _top_attr_text(attrs: dict[str, float], limit: int = 3) -> str:
    items = sorted(attrs.items(), key=lambda item: item[1], reverse=True)[:limit]
    return ", ".join(f"{ATTRIBUTE_DISPLAY_NAMES.get(name, name)}={value:.2f}" for name, value in items)


def _attribute_level(value: float) -> str:
    if value >= 0.50:
        return "strong_positive"
    if value >= 0.15:
        return "positive"
    if value <= -0.50:
        return "strong_negative"
    if value <= -0.15:
        return "negative"
    return "neutral"


def _score_margin_level(margin: float) -> str:
    if margin >= 0.10:
        return "clear"
    if margin >= 0.03:
        return "moderate"
    return "close"


def _attribute_summary(attrs: dict[str, float], limit: int = 4) -> dict[str, object]:
    items = [
        {
            "name": name,
            "display_name": ATTRIBUTE_DISPLAY_NAMES.get(name, name),
            "value": float(value),
            "level": _attribute_level(float(value)),
        }
        for name, value in attrs.items()
    ]
    positives = sorted(
        [item for item in items if float(item["value"]) >= 0.15],
        key=lambda item: float(item["value"]),
        reverse=True,
    )[:limit]
    negatives = sorted(
        [item for item in items if float(item["value"]) <= -0.15],
        key=lambda item: float(item["value"]),
    )[:limit]
    return {
        "strongest_positive": positives,
        "strongest_negative": negatives,
        "all": items,
    }


def _attribute_differences(
    top_record: dict[str, object],
    other_record: dict[str, object],
    limit: int = 4,
) -> dict[str, list[dict[str, object]]]:
    top_attrs = top_record["attributes"]
    other_attrs = other_record["attributes"]
    if not isinstance(top_attrs, dict) or not isinstance(other_attrs, dict):
        return {"top_advantages": [], "top_disadvantages": []}

    diffs = []
    for name in ATTRIBUTES:
        diff = float(top_attrs.get(name, 0.0)) - float(other_attrs.get(name, 0.0))
        diffs.append(
            {
                "name": name,
                "display_name": ATTRIBUTE_DISPLAY_NAMES.get(name, name),
                "top_value": float(top_attrs.get(name, 0.0)),
                "other_value": float(other_attrs.get(name, 0.0)),
                "difference": diff,
            }
        )

    advantages = [item for item in sorted(diffs, key=lambda item: item["difference"], reverse=True) if item["difference"] > 0.05]
    disadvantages = [item for item in sorted(diffs, key=lambda item: item["difference"]) if item["difference"] < -0.05]
    return {
        "top_advantages": advantages[:limit],
        "top_disadvantages": disadvantages[:limit],
    }


def build_structured_payload(
    ranked: list[dict[str, object]],
    checkpoint_path: str,
    image_size: int,
    resize_size: int,
    tta_views: int,
) -> dict[str, object]:
    enriched: list[dict[str, object]] = []
    for index, record in enumerate(ranked):
        item = dict(record)
        previous_score = float(ranked[index - 1]["rank_score"]) if index > 0 else None
        next_score = float(ranked[index + 1]["rank_score"]) if index + 1 < len(ranked) else None
        rank_score = float(record["rank_score"])
        item["rank"] = index + 1
        item["score_gaps"] = {
            "behind_previous": None if previous_score is None else previous_score - rank_score,
            "ahead_of_next": None if next_score is None else rank_score - next_score,
        }
        item["attribute_summary"] = _attribute_summary(record["attributes"])
        item["color_harmony"] = {
            "role": "explanation_only_not_used_for_ranking",
            "score": float(record["harmony_score"]),
            "distance_degrees": float(record["harmony_distance"]),
            "level": record["harmony_level"],
            "percentile": float(record["harmony_percentile"]),
            "best_template": record["best_template"],
            "best_rotation_degrees": record["best_rotation_degrees"],
            "dominant_hues": record["dominant_hues"],
        }
        enriched.append(item)

    top = enriched[0]
    runner_up = enriched[1] if len(enriched) > 1 else None
    top_margin = (
        float(top["rank_score"]) - float(runner_up["rank_score"])
        if runner_up is not None
        else None
    )
    comparisons = []
    for other in enriched[1:]:
        score_margin = float(top["rank_score"]) - float(other["rank_score"])
        comparisons.append(
            {
                "top_filename": top["filename"],
                "other_filename": other["filename"],
                "score_margin": score_margin,
                "margin_level": _score_margin_level(score_margin),
                "top_harmony_distance_minus_other": float(top["harmony_distance"]) - float(other["harmony_distance"]),
                "top_harmony_score_minus_other": float(top["harmony_score"]) - float(other["harmony_score"]),
                "attribute_differences": _attribute_differences(top, other),
            }
        )

    return {
        "metadata": {
            "task": "photo_src",
            "input_count": len(enriched),
            "checkpoint": checkpoint_path,
            "image_size": image_size,
            "resize_size": resize_size,
            "tta_views": tta_views,
            "ranking_method": "sort_descending_by_rank_score",
            "rank_score_field": "aesthetic_score",
            "color_harmony_role": "explanation_only_not_used_for_ranking",
            "attribute_role": "auxiliary_model_outputs_for_explanation",
            "llm_rules": [
                "Use only fields in this JSON.",
                "Do not claim the model saw facts that are not represented by computed attributes.",
                "Explain that ranking is determined by aesthetic_score/rank_score only.",
                "Use color harmony as supporting evidence, not as the reason the score changed.",
            ],
        },
        "summary": {
            "top_filename": top["filename"],
            "top_rank_score": float(top["rank_score"]),
            "runner_up_filename": None if runner_up is None else runner_up["filename"],
            "top_margin_over_runner_up": top_margin,
            "top_margin_level": None if top_margin is None else _score_margin_level(top_margin),
        },
        "ranked": enriched,
        "comparisons": comparisons,
    }


def write_json_payload(path: str | Path, payload: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 2 <= len(args.images) <= 5:
        raise SystemExit("--images requires 2 to 5 images.")

    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    image_size = int(checkpoint.get("image_size", 224))
    resize_size = int(checkpoint.get("resize_size", 256))

    records: list[dict[str, object]] = []
    for image_arg in args.images:
        path = resolve_image_path(image_arg, args.image_root)
        prediction = predict_image(
            model,
            path,
            device=device,
            image_size=image_size,
            resize_size=resize_size,
            tta_views=args.tta_views,
        )
        harmony = analyze_image(path)
        aesthetic_score = float(prediction["aesthetic_score"])
        attrs = _attr_dict(prediction["attributes"])
        harmony_score = float(harmony["harmony_score"])

        records.append(
            {
                "filename": Path(path).name,
                "path": str(path),
                "rank_score": aesthetic_score,
                "aesthetic_score": aesthetic_score,
                "harmony_score": harmony_score,
                "harmony_distance": float(harmony["harmony_distance"]),
                "harmony_level": harmony["harmony_level"],
                "harmony_percentile": float(harmony["harmony_percentile"]),
                "best_template": harmony["best_template"],
                "best_rotation_degrees": harmony["best_rotation_degrees"],
                "dominant_hues": harmony["dominant_hues"],
                "attributes": attrs,
            }
        )

    ranked = sorted(records, key=lambda item: float(item["rank_score"]), reverse=True)
    payload = build_structured_payload(
        ranked,
        checkpoint_path=args.checkpoint,
        image_size=image_size,
        resize_size=resize_size,
        tta_views=args.tta_views,
    )

    if args.llm_input:
        write_json_payload(args.llm_input, payload)

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    for index, record in enumerate(ranked, start=1):
        print(f"Rank {index}: {record['filename']}")
        print(f"  Ranking score: {float(record['rank_score']):.3f}")
        print(f"  Harmony score: {float(record['harmony_score']):.3f}")
        print(f"  Harmony distance: {float(record['harmony_distance']):.2f} deg")
        print(f"  Harmony level: {record['harmony_level']} ({float(record['harmony_percentile']):.1f} percentile)")
        print(f"  Best harmonic template: {record['best_template']}")
        print(f"  Top attributes: {_top_attr_text(record['attributes'])}")
    if args.llm_input:
        print(f"Wrote structured LLM input: {args.llm_input}")


if __name__ == "__main__":
    main()
