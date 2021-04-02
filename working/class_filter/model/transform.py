
from albumentations import (
    PadIfNeeded, HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,     IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize,ToGray
)

from albumentations.pytorch import ToTensorV2

def get_train_transforms(input_shape, way="resize", crop_rate=0.98):
    if way == "resize":
        return Compose([
                Resize(input_shape[0], input_shape[1]),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(scale_limit=0.05, rotate_limit=5, p=0.5),
                Cutout(p=0.5),
                ToTensorV2(p=1.0),
            ], p=1.)

def get_valid_transforms(input_shape, way="pad", crop_rate=0.98):
    if way == "resize":
        return Compose([
                Resize(input_shape[0], input_shape[1]),
                ToTensorV2(p=1.0),
            ], p=1.)


def get_inference_transforms(input_shape, way="pad", crop_rate=0.98):
    if way == "resize":
        return Compose([
                Resize(input_shape[0], input_shape[1]),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(scale_limit=0.05, rotate_limit=5, p=0.5),
                Cutout(p=0.5),
                ToTensorV2(p=1.0),
            ], p=1.)
