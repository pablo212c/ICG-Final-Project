from __future__ import annotations

import base64
import sys
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

# Make sibling folders (src/, scripts/) importable.
_PROJECT_ROOT = Path(__file__).resolve().parent
for _sub in ("", "scripts"):
    _p = str(_PROJECT_ROOT / _sub) if _sub else str(_PROJECT_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gradio as gr

from src.color_harmony import analyze_image
from src.inference import load_checkpoint, predict_image, resolve_device
from rank import _attr_dict, _top_attr_text, build_structured_payload


CHECKPOINT_PATH = "checkpoints/best.pt"
DEVICE = "auto"
TTA_VIEWS = 10

APP_CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    color: #111827;
    font-family: "Segoe UI", "Noto Sans TC", "Microsoft JhengHei", Arial, sans-serif !important;
}
body,
button,
input,
textarea,
select {
    font-family: "Segoe UI", "Noto Sans TC", "Microsoft JhengHei", Arial, sans-serif !important;
}
footer {
    display: none !important;
}
.hero {
    text-align: center;
    padding: 26px 8px 14px;
}
.hero h1 {
    font-size: 38px;
    line-height: 1.15;
    margin: 0 0 8px;
    font-weight: 800;
    letter-spacing: 0;
}
.hero p {
    font-size: 17px;
    color: #4b5563;
    margin: 0;
}
.upload-dropzone {
    border: 2px dashed #93c5fd !important;
    border-radius: 14px !important;
    background: #f8fbff !important;
    padding: 16px !important;
}
.upload-dropzone:hover {
    border-color: #2563eb !important;
    background: #eff6ff !important;
}
.upload-hint {
    font-size: 15px;
    color: #4b5563;
    margin-top: -4px;
}
.upload-status {
    font-size: 15px;
    color: #374151;
    margin-top: -6px;
}
.upload-status p {
    margin: 0;
}
.upload-status-ready {
    color: #166534;
    font-weight: 700;
}
.upload-status-error {
    color: #b91c1c;
    font-weight: 700;
}
.section-title h2 {
    font-size: 24px;
    margin: 14px 0 8px;
    font-weight: 800;
    letter-spacing: 0;
}
.selected-summary {
    font-size: 18px;
    line-height: 1.6;
}
.winner-frame {
    border: 3px solid #2563eb !important;
    border-radius: 12px !important;
    padding: 8px !important;
    background: #eff6ff !important;
}
.winner-frame img {
    border-radius: 8px !important;
}
.ranked-cards {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 12px;
    margin: 4px 0 14px;
}
.rank-card {
    border: 1px solid #d1d5db;
    border-radius: 10px;
    overflow: hidden;
    background: #ffffff;
    min-width: 0;
}
.rank-card-top {
    border: 2px solid #2563eb;
    box-shadow: 0 8px 18px rgba(37, 99, 235, 0.14);
}
.rank-card img {
    width: 100%;
    height: 126px;
    object-fit: cover;
    display: block;
    background: #f3f4f6;
}
.rank-card-body {
    padding: 9px 10px 10px;
}
.rank-card-title {
    font-size: 14px;
    font-weight: 800;
    color: #111827;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.rank-card-score {
    font-size: 13px;
    color: #4b5563;
    margin-top: 3px;
}
.ranking-table {
    border: 1px solid #d1d5db;
    border-radius: 10px;
    overflow: hidden;
    background: #ffffff;
    margin-top: 8px;
}
.ranking-table table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 15px;
    margin: 0 !important;
}
.ranking-table th {
    background: #f3f4f6;
    color: #111827;
    font-weight: 700;
}
.ranking-table th,
.ranking-table td {
    border-bottom: 1px solid #e5e7eb;
    padding: 10px 12px;
    text-align: left;
}
.ranking-table tr:first-child td {
    font-weight: 700;
    background: #eff6ff;
}
.ranking-table tr:last-child td {
    border-bottom: 0;
}
.explanation-box {
    font-size: 17px;
    line-height: 1.72;
    color: #1f2937;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    background: #ffffff;
    padding: 14px 16px;
}
.explanation-box h3 {
    font-size: 22px;
    line-height: 1.3;
    margin-top: 0;
    font-weight: 800;
}
@media (max-width: 900px) {
    .ranked-cards {
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }
}
@media (max-width: 640px) {
    .ranked-cards {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
"""


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, dict[str, object], Any]:
    device = resolve_device(DEVICE)
    model, checkpoint = load_checkpoint(CHECKPOINT_PATH, device=device)
    return model, checkpoint, device


def _uploaded_path(file: Any) -> Path:
    if isinstance(file, (str, Path)):
        return Path(file)
    if hasattr(file, "name"):
        return Path(file.name)
    raise ValueError(f"Unsupported uploaded file object: {type(file)}")


def _file_count(files: list[Any] | None) -> int:
    return len(files or [])


def _upload_state(files: list[Any] | None) -> tuple[Any, str]:
    count = _file_count(files)
    if 2 <= count <= 5:
        status = f'<span class="upload-status-ready">{count} images ready. Click Rank Photos.</span>'
        return gr.update(interactive=True), status
    if count == 0:
        status = "Drag 2-5 images into the upload area, or use the upload button."
    elif count == 1:
        status = '<span class="upload-status-error">Add at least one more image to rank.</span>'
    else:
        status = '<span class="upload-status-error">Use at most 5 images.</span>'
    return gr.update(interactive=False), status


def _format_attrs(attrs: dict[str, float], limit: int = 3) -> str:
    return _top_attr_text(attrs, limit=limit) or "No strong attribute signal"


def _harmony_text(record: dict[str, object]) -> str:
    return (
        f"{record['harmony_level']} "
        f"({float(record['harmony_distance']):.2f} deg, "
        f"{float(record['harmony_percentile']):.1f} pct)"
    )


def _markdown_table(ranked: list[dict[str, object]]) -> str:
    lines = [
        "| Rank | Image | Ranking score | Color harmony fit | Template | Top attributes |",
        "|---:|---|---:|---|---|---|",
    ]
    for index, record in enumerate(ranked, start=1):
        lines.append(
            "| "
            f"{index} | "
            f"{record['filename']} | "
            f"{float(record['rank_score']):.3f} | "
            f"{_harmony_text(record)} | "
            f"{record['best_template']} | "
            f"{_format_attrs(record['attributes'])} |"
        )
    return "\n".join(lines)


def _image_data_uri(path: Path) -> str:
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime = mime_types.get(path.suffix.lower(), "image/jpeg")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _gallery_html(ranked: list[dict[str, object]]) -> str:
    cards: list[str] = []
    for index, record in enumerate(ranked, start=1):
        path = Path(str(record["path"]))
        card_class = "rank-card rank-card-top" if index == 1 else "rank-card"
        cards.append(
            f"""
            <div class="{card_class}">
              <img src="{_image_data_uri(path)}" alt="{escape(str(record['filename']))}">
              <div class="rank-card-body">
                <div class="rank-card-title">Rank {index}: {escape(str(record['filename']))}</div>
                <div class="rank-card-score">Ranking score {float(record['rank_score']):.3f}</div>
              </div>
            </div>
            """
        )
    return f"<div class=\"ranked-cards\">{''.join(cards)}</div>"


def _make_explanation(payload: dict[str, object]) -> str:
    ranked = payload["ranked"]
    summary = payload["summary"]
    comparisons = payload.get("comparisons", [])
    if not isinstance(ranked, list) or not ranked:
        return "No ranking result."

    top = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    top_name = str(top["filename"])
    top_score = float(top["rank_score"])
    color = top.get("color_harmony", {})

    lines = [
        f"### Why `{top_name}` was selected",
        "",
        f"`{top_name}` is selected because its predicted aesthetic ranking is strongest in this set.",
    ]

    if runner_up is not None:
        margin = summary.get("top_margin_over_runner_up")
        margin_level = summary.get("top_margin_level")
        lines.append(
            f"It is ahead of `{runner_up['filename']}` by **{float(margin):.3f}** "
            f"({margin_level} margin)."
        )

    advantage_phrases: list[str] = []
    seen: set[str] = set()
    if isinstance(comparisons, list):
        for comparison in comparisons:
            if not isinstance(comparison, dict):
                continue
            differences = comparison.get("attribute_differences", {})
            if not isinstance(differences, dict):
                continue
            advantages = differences.get("top_advantages", [])
            if not isinstance(advantages, list):
                continue
            for item in advantages:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", ""))
                if name in seen:
                    continue
                seen.add(name)
                display = str(item.get("display_name", name))
                diff = float(item.get("difference", 0.0))
                if diff < 0.08:
                    continue
                phrase = _attribute_explanation(display, diff)
                if phrase:
                    advantage_phrases.append(phrase)
                if len(advantage_phrases) >= 4:
                    break
            if len(advantage_phrases) >= 4:
                break

    if advantage_phrases:
        lines.append("Compared with the other candidates, it looks stronger because:")
        for phrase in advantage_phrases:
            lines.append(f"- {phrase}")
    else:
        top_attrs = top.get("attribute_summary", {})
        top_positive = top_attrs.get("strongest_positive", []) if isinstance(top_attrs, dict) else []
        if top_positive:
            positives = ", ".join(
                f"{item['display_name']}"
                for item in top_positive[:3]
            )
            lines.append(f"Its clearest visual strengths are **{positives}**.")

    if isinstance(color, dict):
        distance = float(color["distance_degrees"])
        percentile = float(color["percentile"])
        lines.append(
            "Its color-harmony fit is "
            f"**{color['level']}** by template distance "
            f"(**{distance:.2f} deg**, {percentile:.1f} percentile), "
            f"closest to template **{color['best_template']}**."
        )

    return "\n".join(lines)


def _attribute_explanation(display_name: str, diff: float) -> str:
    strength = "clearly" if diff >= 0.20 else "slightly"
    mapping = {
        "Balancing elements": f"the composition feels {strength} more balanced.",
        "Color harmony": f"the colors appear {strength} more coordinated.",
        "Content": f"the scene/content is {strength} more visually meaningful.",
        "Depth of field": f"the focus separation and depth of field are {strength} more appealing.",
        "Light": f"the lighting looks {strength} better controlled.",
        "Motion blur": f"there is {strength} less distracting blur or motion softness.",
        "Object emphasis": f"the main subject is {strength} easier to identify.",
        "Repetition": f"repeated shapes or visual rhythm are {strength} more effective.",
        "Rule of thirds": f"the framing follows a {strength} stronger rule-of-thirds layout.",
        "Symmetry": f"the image feels {strength} more symmetrical or orderly.",
        "Vivid color": f"the colors are {strength} more vivid and engaging.",
    }
    return mapping.get(display_name, f"{display_name} is {strength} stronger.")


def _winner_summary(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    top_name = summary["top_filename"]
    top_score = float(summary["top_rank_score"])
    runner_up = summary.get("runner_up_filename")
    margin = summary.get("top_margin_over_runner_up")
    if runner_up is None or margin is None:
        return f"### Selected: `{top_name}`\n\nRanking score: **{top_score:.3f}**"
    return (
        f"### Selected: `{top_name}`\n\n"
        f"Ranking score: **{top_score:.3f}**  \n"
        f"Margin over `{runner_up}`: **{float(margin):.3f}**"
    )


def rank_uploaded_images(files: list[Any]) -> tuple[str, str, str, str, str]:
    if not files:
        raise gr.Error("Upload 2 to 5 images.")
    if not 2 <= len(files) <= 5:
        raise gr.Error("Please upload 2 to 5 images.")

    model, checkpoint, device = _load_model()
    image_size = int(checkpoint.get("image_size", 256))
    resize_size = int(checkpoint.get("resize_size", 256))

    records: list[dict[str, object]] = []
    for file in files:
        path = _uploaded_path(file)
        prediction = predict_image(
            model,
            path,
            device=device,
            image_size=image_size,
            resize_size=resize_size,
            tta_views=TTA_VIEWS,
        )
        harmony = analyze_image(path)
        score = float(prediction["aesthetic_score"])
        attrs = _attr_dict(prediction["attributes"])

        records.append(
            {
                "filename": path.name,
                "path": str(path),
                "rank_score": score,
                "aesthetic_score": score,
                "harmony_score": float(harmony["harmony_score"]),
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
        checkpoint_path=CHECKPOINT_PATH,
        image_size=image_size,
        resize_size=resize_size,
        tta_views=TTA_VIEWS,
    )

    winner_path = str(ranked[0]["path"])
    return (
        winner_path,
        _winner_summary(payload),
        _gallery_html(ranked),
        _markdown_table(ranked),
        _make_explanation(payload),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="ICG Photo Ranking") as app:
        gr.HTML(
            """
            <div class="hero">
              <h1>ICG Photo Ranking</h1>
              <p>Upload 2-5 photos. The model selects the strongest image and explains the ranking with aesthetic attributes and color harmony.</p>
            </div>
            """
        )
        gr.Markdown(
            "Drag images into the upload area for automatic upload, or use the upload button.",
            elem_classes=["upload-hint"],
        )
        files = gr.Files(
            label="Upload 2-5 Images",
            file_count="multiple",
            file_types=["image"],
            type="filepath",
            elem_classes=["upload-dropzone"],
        )
        upload_status = gr.HTML(
            "Drag 2-5 images into the upload area, or use the upload button.",
            elem_classes=["upload-status"],
        )
        run_button = gr.Button("Rank Photos", variant="primary", size="lg", interactive=False)

        gr.HTML('<div class="section-title"><h2>Selected Photo</h2></div>')
        with gr.Row():
            selected_image = gr.Image(
                label="Rank 1",
                type="filepath",
                height=300,
                interactive=False,
                elem_classes=["winner-frame"],
            )
            selected_summary = gr.Markdown(elem_classes=["selected-summary"])

        gr.HTML('<div class="section-title"><h2>Ranking Results</h2></div>')
        ranked_gallery = gr.HTML()
        ranking_table = gr.Markdown(elem_classes=["ranking-table"])

        gr.HTML('<div class="section-title"><h2>Explanation</h2></div>')
        explanation = gr.Markdown(elem_classes=["explanation-box"])

        files.change(
            _upload_state,
            inputs=[files],
            outputs=[run_button, upload_status],
        )
        run_button.click(
            rank_uploaded_images,
            inputs=[files],
            outputs=[selected_image, selected_summary, ranked_gallery, ranking_table, explanation],
        )
    return app


if __name__ == "__main__":
    build_app().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=APP_CSS,
        footer_links=[],
    )
