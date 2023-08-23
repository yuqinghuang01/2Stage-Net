import os
import torch
from config import (
    NUM_CLASSES, IN_CHANNELS, BACKGROUND_AS_CLASS, TRAIN_CUDA
)
from dataset import get_train_val_test_Dataloaders
from torch.utils.tensorboard import SummaryWriter
from unet3d import UNet3D
from transforms import (train_transform, train_transform_cuda,
                        val_transform, val_transform_cuda)
from monai.visualize import plot_2d_or_3d_image

if BACKGROUND_AS_CLASS: NUM_CLASSES += 1

writer = SummaryWriter("log/")

model = UNet3D(in_channels=IN_CHANNELS , num_classes=NUM_CLASSES, level_channels=[4, 8, 16], bottleneck_channel=32)
train_transforms = train_transform
val_transforms = val_transform

if torch.cuda.is_available() and TRAIN_CUDA:
    model = model.cuda()
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda
elif not torch.cuda.is_available() and TRAIN_CUDA:
    print('cuda not available! Training initialized on cpu ...')

# Load best model
root_dir = "checkpoints/"
model.load_state_dict(torch.load(
    os.path.join(root_dir, "epoch12_valLoss0.7113795280456543.pth")))
model.eval()

train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)

with torch.no_grad():
    for data in val_dataloader:
        image, ground_truth = data['image'], data['label']
        print(image.size())
        plot_2d_or_3d_image(data=image, step=0, writer=writer, frame_dim=-1, tag="image")
        print(ground_truth.size())
        plot_2d_or_3d_image(data=ground_truth, step=0, writer=writer, frame_dim=-1, tag="label")
        ground_truth = torch.squeeze(ground_truth, dim=1).long()

        target = model(image)
        print(target.size())
        # Visualize the model output with the input label
        plot_2d_or_3d_image(data=target, step=0, writer=writer, max_channels=3, frame_dim=-1, tag="predict")

writer.flush()
writer.close()
