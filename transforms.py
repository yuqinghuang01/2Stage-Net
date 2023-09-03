import numpy as np
from monai.transforms import (
    Compose,
    ToTensord,
    RandFlipd,
    Spacingd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    AddChanneld,
    DivisiblePadd,
    RandCropByLabelClassesd,
)

from config import (BACKGROUND_AS_CLASS, NUM_CLASSES, CROP_RATIO, TRAIN_CROP_SAMPLES, TASK_ID, PATCH_SIZE_X, PATCH_SIZE_Y, PATCH_SIZE_Z)

if BACKGROUND_AS_CLASS: NUM_CLASSES += 1

#Transforms to be applied on training instances
train_transform = Compose(
    [   
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=['image', 'label'], pixdim=(1., 1., 1.), mode=("bilinear", "nearest")),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
        RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
        DivisiblePadd(k=16, keys=["image", "label"]),
        RandCropByLabelClassesd(
            keys=['image', 'label'], 
            label_key='label', 
            spatial_size=(PATCH_SIZE_X, PATCH_SIZE_Y, PATCH_SIZE_Z), 
            ratios=CROP_RATIO, 
            num_classes=NUM_CLASSES, 
            num_samples=TRAIN_CROP_SAMPLES,
        ),
        ToTensord(keys=['image', 'label'])
    ]
)

#Cuda version of "train_transform"
train_transform_cuda = Compose(
    [   
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=['image', 'label'], pixdim=(1., 1., 1.), mode=("bilinear", "nearest")),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
        RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
        DivisiblePadd(k=16, keys=["image", "label"]),
        RandCropByLabelClassesd(
            keys=['image', 'label'], 
            label_key='label', 
            spatial_size=(PATCH_SIZE_X, PATCH_SIZE_Y, PATCH_SIZE_Z), 
            ratios=CROP_RATIO, 
            num_classes=NUM_CLASSES, 
            num_samples=TRAIN_CROP_SAMPLES,
        ),
        ToTensord(keys=['image', 'label'], device='cuda')
    ]
)

#Transforms to be applied on validation instances
val_transform = Compose(
    [   
        AddChanneld(keys=["image", "label"]),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        DivisiblePadd(k=16, keys=["image", "label"]),
        ToTensord(keys=['image', 'label'])
    ]
)

#Cuda version of "val_transform"
val_transform_cuda = Compose(
    [   
        AddChanneld(keys=["image", "label"]),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        DivisiblePadd(k=16, keys=["image", "label"]),
        ToTensord(keys=['image', 'label'], device='cuda')
    ]
)