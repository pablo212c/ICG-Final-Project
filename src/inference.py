from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageOps

from src.data import ATTRIBUTES
from src.model import create_model
from src.transforms import EvalTTATransform, ImageTransform


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, object]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    attr_names = checkpoint.get("attr_names", ATTRIBUTES)
    model = create_model(
        backbone=str(checkpoint.get("backbone", "resnet18")),
        pretrained=False,
        attr_count=len(attr_names),
        dropout=float(checkpoint.get("dropout", 0.2)),
        head_hidden_dim=int(checkpoint.get("head_hidden_dim", 0)),
        stochastic_depth_scale=float(checkpoint.get("stochastic_depth_scale", 1.0)),
        image_size=int(checkpoint.get("image_size", 224)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def resolve_image_path(image: str | Path, image_root: str | Path | None = None) -> Path:
    path = Path(image)
    if path.exists():
        return path
    if image_root is not None:
        rooted = Path(image_root) / path.name
        if rooted.exists():
            return rooted
    raise FileNotFoundError(f"Could not find image '{image}'.")


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image_path: str | Path,
    device: torch.device,
    image_size: int = 224,
    resize_size: int | None = 256,
    tta_views: int = 1,
) -> dict[str, object]:
    if tta_views > 1:
        transform = EvalTTATransform(image_size=image_size, resize_size=resize_size, views=tta_views)
    else:
        transform = ImageTransform(image_size=image_size, resize_size=resize_size, train=False)
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = transform(image)
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(0).to(device)
            views = tensor.shape[1]
            flat = tensor.reshape(views, *tensor.shape[2:])
            output = model(flat)
            score = output["score"].mean()
            attributes = output["attributes"].mean(dim=0)
        else:
            tensor = tensor.unsqueeze(0).to(device)
            output = model(tensor)
            score = output["score"].squeeze(0)
            attributes = output["attributes"].squeeze(0)
    return {
        "aesthetic_score": float(score.detach().cpu().item()),
        "attributes": attributes.detach().cpu().tolist(),
    }
