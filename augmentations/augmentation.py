from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    ToGray,
    RandomCrop,
    CenterCrop,
    RandomSizedCrop,
    PadIfNeeded,
    RandomContrast,
    RandomBrightness,
    CLAHE,
    ShiftScaleRotate,
    Transpose,
    RandomGamma,
    RandomRotate90,
    Cutout,
    GaussNoise,
    Blur,
)

__all__ = ['get_strong_augmentations', 'get_medium_augmentations', 'get_light_augmentations']


def get_strong_augmentations(width, height):
    # TODO maybe consider more augmentations and play with params
    return [
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        ToGray(p=0.5),
        ShiftScaleRotate(p=0.5),
        RandomCrop(height, width, p=0.5),
        CenterCrop(height, width, p=0.5),
        RandomSizedCrop((height - 6, height - 2), height, width, p=0.5),
        RandomContrast(p=0.5),
        RandomBrightness(p=0.5),
        RandomGamma(p=0.5),
        CLAHE(p=0.5),
        Blur(blur_limit=3, p=0.5),
        GaussNoise(p=0.5)
    ]


def get_medium_augmentations(width, height):
    return [
        HorizontalFlip(p=1),
        VerticalFlip(p=1),
        RandomSizedCrop((height - 4, height - 2), height, width, p=1),
        RandomContrast(p=1),
        RandomBrightness(p=1),
        RandomGamma(p=1),
        ShiftScaleRotate(p=1),
        Blur(blur_limit=3, p=1)
    ]


def get_light_augmentations(width, height):
    return [
        HorizontalFlip(p=1),
        VerticalFlip(p=1),
        Transpose(p=1),
        RandomSizedCrop((height - 4, height - 2), height, width),
    ]
