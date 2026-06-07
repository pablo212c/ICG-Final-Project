from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


ATTRIBUTES = [
    "BalacingElements",
    "ColorHarmony",
    "Content",
    "DoF",
    "Light",
    "MotionBlur",
    "Object",
    "Repetition",
    "RuleOfThirds",
    "Symmetry",
    "VividColor",
]

ATTRIBUTE_DISPLAY_NAMES = {
    "BalacingElements": "Balancing elements",
    "ColorHarmony": "Color harmony",
    "Content": "Content",
    "DoF": "Depth of field",
    "Light": "Light",
    "MotionBlur": "Motion blur",
    "Object": "Object emphasis",
    "Repetition": "Repetition",
    "RuleOfThirds": "Rule of thirds",
    "Symmetry": "Symmetry",
    "VividColor": "Vivid color",
}

SPLIT_PREFIXES = {
    "train": "Train",
    "validation": "Validation",
    "val": "Validation",
    "test": "Test",
    "testnew": "TestNew",
    "test_new": "TestNew",
}


@dataclass(frozen=True)
class SampleRecord:
    filename: str
    score: float
    attributes: tuple[float, ...]


def resolve_label_root(label_root: str | Path) -> Path:
    """Resolve either imgListFiles_label or the nested label directory."""
    root = Path(label_root)
    if (root / "imgListTrainRegression_score.txt").exists():
        return root

    nested = root / "imgListFiles_label"
    if (nested / "imgListTrainRegression_score.txt").exists():
        return nested

    raise FileNotFoundError(
        "Could not find AADB label txt files. Pass either "
        "'imgListFiles_label' or 'imgListFiles_label/imgListFiles_label'."
    )


def read_regression_file(path: str | Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError(f"Malformed label row at {path}:{line_no}: {line!r}")
            rows.append((parts[0], float(parts[1])))
    return rows


def load_records(
    split: str,
    label_root: str | Path = "imgListFiles_label",
    include_attributes: bool = True,
) -> list[SampleRecord]:
    label_dir = resolve_label_root(label_root)
    split_key = split.lower()
    if split_key not in SPLIT_PREFIXES:
        raise ValueError(f"Unknown split '{split}'. Expected one of {sorted(SPLIT_PREFIXES)}")
    prefix = SPLIT_PREFIXES[split_key]

    score_rows = read_regression_file(label_dir / f"imgList{prefix}Regression_score.txt")
    attr_maps: dict[str, dict[str, float]] = {}
    if include_attributes:
        for attr in ATTRIBUTES:
            attr_maps[attr] = dict(
                read_regression_file(label_dir / f"imgList{prefix}Regression_{attr}.txt")
            )

    records: list[SampleRecord] = []
    for filename, score in score_rows:
        attrs = (
            tuple(attr_maps[attr].get(filename, 0.0) for attr in ATTRIBUTES)
            if include_attributes
            else tuple(0.0 for _ in ATTRIBUTES)
        )
        records.append(SampleRecord(filename=filename, score=score, attributes=attrs))
    return records


class AADBDataset(Dataset):
    def __init__(
        self,
        image_root: str | Path,
        label_root: str | Path,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        limit: int | None = None,
        check_files: bool = True,
    ) -> None:
        self.image_root = Path(image_root)
        self.records = load_records(split, label_root=label_root, include_attributes=True)
        if limit is not None:
            self.records = self.records[:limit]
        self.transform = transform

        if check_files:
            missing = [r.filename for r in self.records if not (self.image_root / r.filename).exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} images from split '{split}' were not found under "
                    f"{self.image_root}. First missing file: {missing[0]}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image_path = self.image_root / record.filename
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if self.transform is not None:
                image_tensor = self.transform(image)
            else:
                raise RuntimeError("AADBDataset requires a transform returning a tensor.")

        return {
            "image": image_tensor,
            "score": torch.tensor(record.score, dtype=torch.float32),
            "attributes": torch.tensor(record.attributes, dtype=torch.float32),
            "filename": record.filename,
        }
