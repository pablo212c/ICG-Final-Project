from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image


try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9
    RESAMPLE_BILINEAR = Image.BILINEAR


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class ImageTransform:
    def __init__(
        self,
        image_size: int = 224,
        resize_size: int | None = None,
        train: bool = False,
        hflip_prob: float = 0.5,
    ) -> None:
        self.image_size = image_size
        self.resize_size = resize_size or (256 if image_size <= 224 else image_size)
        self.train = train
        self.hflip_prob = hflip_prob

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.resize_size, self.resize_size), RESAMPLE_BILINEAR)
        image = self._crop(image)
        if self.train and random.random() < self.hflip_prob:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD

    def _crop(self, image: Image.Image) -> Image.Image:
        if self.resize_size == self.image_size:
            return image

        max_offset = self.resize_size - self.image_size
        if max_offset < 0:
            return image.resize((self.image_size, self.image_size), RESAMPLE_BILINEAR)

        if self.train:
            left = random.randint(0, max_offset)
            top = random.randint(0, max_offset)
        else:
            left = max_offset // 2
            top = max_offset // 2

        return image.crop((left, top, left + self.image_size, top + self.image_size))


class EvalTTATransform:
    def __init__(
        self,
        image_size: int = 224,
        resize_size: int | None = 256,
        views: int = 10,
    ) -> None:
        self.image_size = image_size
        self.resize_size = resize_size or (256 if image_size <= 224 else image_size)
        self.views = max(1, views)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.resize_size, self.resize_size), RESAMPLE_BILINEAR)
        crops = self._crops(image)
        tensors = [self._to_tensor(crop) for crop in crops]
        return torch.stack(tensors, dim=0)

    def _crops(self, image: Image.Image) -> list[Image.Image]:
        if self.resize_size <= self.image_size:
            base = [image.resize((self.image_size, self.image_size), RESAMPLE_BILINEAR)]
        else:
            max_offset = self.resize_size - self.image_size
            center = max_offset // 2
            positions = [
                (center, center),
                (0, 0),
                (max_offset, 0),
                (0, max_offset),
                (max_offset, max_offset),
            ]
            base = [
                image.crop((left, top, left + self.image_size, top + self.image_size))
                for left, top in positions
            ]

        crops: list[Image.Image] = []
        for crop in base:
            crops.append(crop)
            crops.append(crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
        return crops[: self.views]

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD
