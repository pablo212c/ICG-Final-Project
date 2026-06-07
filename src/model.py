from __future__ import annotations

import warnings

import torch
from torch import nn
from torchvision import models


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int, bool]:
    if name != "convnext_small":
        raise ValueError("This project is configured for the ConvNeXt-Small backbone.")
    pretrained_loaded = False
    try:
        weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_small(weights=weights)
        pretrained_loaded = weights is not None
    except Exception as exc:
        if not pretrained:
            raise
        warnings.warn(
            f"Could not load pretrained ConvNeXt-Small weights ({exc}). "
            "Falling back to random initialization.",
            RuntimeWarning,
        )
        backbone = models.convnext_small(weights=None)

    feature_dim = backbone.classifier[-1].in_features
    backbone.classifier[-1] = nn.Identity()
    return backbone, feature_dim, pretrained_loaded


def _scale_stochastic_depth(module: nn.Module, scale: float) -> None:
    if scale < 0:
        raise ValueError("stochastic_depth_scale must be >= 0.")
    for child in module.modules():
        if child.__class__.__name__ == "StochasticDepth" and hasattr(child, "p"):
            child.p *= scale


class AestheticAttributeModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "convnext_small",
        pretrained: bool = False,
        attr_count: int = 11,
        dropout: float = 0.2,
        head_hidden_dim: int = 0,
        stochastic_depth_scale: float = 1.0,
        image_size: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, feature_dim, self.pretrained_loaded = _build_backbone(
            backbone_name,
            pretrained=pretrained,
        )
        self.stochastic_depth_scale = stochastic_depth_scale
        if stochastic_depth_scale != 1.0:
            _scale_stochastic_depth(self.backbone, stochastic_depth_scale)
        self.head_hidden_dim = head_hidden_dim
        self.score_head = self._make_head(feature_dim, 1, dropout, head_hidden_dim)
        self.attribute_head = self._make_head(feature_dim, attr_count, dropout, head_hidden_dim)

    @staticmethod
    def _make_head(
        feature_dim: int,
        output_dim: int,
        dropout: float,
        head_hidden_dim: int,
    ) -> nn.Module:
        if head_hidden_dim <= 0:
            return nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, output_dim))
        return nn.Sequential(
            nn.Linear(feature_dim, head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, output_dim),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        score = torch.sigmoid(self.score_head(features)).squeeze(1)
        attributes = torch.tanh(self.attribute_head(features))
        return {"score": score, "attributes": attributes}


def create_model(
    backbone: str = "convnext_small",
    pretrained: bool = False,
    attr_count: int = 11,
    dropout: float = 0.2,
    head_hidden_dim: int = 0,
    stochastic_depth_scale: float = 1.0,
    image_size: int | None = None,
) -> AestheticAttributeModel:
    return AestheticAttributeModel(
        backbone_name=backbone,
        pretrained=pretrained,
        attr_count=attr_count,
        dropout=dropout,
        head_hidden_dim=head_hidden_dim,
        stochastic_depth_scale=stochastic_depth_scale,
        image_size=image_size,
    )
