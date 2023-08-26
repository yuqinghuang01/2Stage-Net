import math
import torch
torch.cuda.empty_cache()
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BCE_WEIGHTS, BACKGROUND_AS_CLASS, TRAIN_CUDA
)
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torchmetrics.functional import dice
from torchmetrics import Precision, Recall, AUROC
from dataset import get_train_val_test_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from models.unet3d import UNet3D
from transforms import (train_transform, train_transform_cuda,
                        val_transform, val_transform_cuda)

if BACKGROUND_AS_CLASS: NUM_CLASSES += 1

writer = SummaryWriter("log")

model = UNet3D(in_channels=IN_CHANNELS , num_classes=NUM_CLASSES, level_channels=[4, 8, 16], bottleneck_channel=32)
train_transforms = train_transform
val_transforms = val_transform
criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS))

if torch.cuda.is_available() and TRAIN_CUDA:
    model = model.cuda()
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda
    criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS).to(device='cuda'))
elif not torch.cuda.is_available() and TRAIN_CUDA:
    print('cuda not available! Training initialized on cpu ...')


train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)


optimizer = Adam(params=model.parameters())

min_valid_loss = math.inf

for epoch in range(TRAINING_EPOCH):
    
    train_loss = 0.0
    train_metric = 0.0
    model.train()
    for data in train_dataloader:
        image, ground_truth = data['image'], data['label']
        ground_truth = torch.squeeze(ground_truth, dim=1).long()
        
        optimizer.zero_grad()

        target = model(image)
        print(target.size())
        print(ground_truth.size())
        loss = criterion(target, ground_truth)
        metric = dice(target, ground_truth, ignore_index=0)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_metric += metric.item()
    
    valid_loss = 0.0
    valid_metric = 0.0
    model.eval()
    for data in val_dataloader:
        image, ground_truth = data['image'], data['label']
        ground_truth = torch.squeeze(ground_truth, dim=1).long()

        target = model(image)
        loss = criterion(target,ground_truth)
        metric = dice(target, ground_truth, ignore_index=0)
        valid_loss += loss.item()
        valid_metric += metric.item()
        
    writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
    writer.add_scalar("Loss/Validation", valid_loss / len(val_dataloader), epoch)
    writer.add_scalar("Metric/Train", train_metric / len(train_dataloader), epoch)
    writer.add_scalar("Metric/Validation", valid_metric / len(val_dataloader), epoch)
    
    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(val_dataloader)}')
    print(f'Epoch {epoch+1} \t\t Training Metric: {train_metric / len(train_dataloader)} \t\t Validation Metric: {valid_metric / len(val_dataloader)}')

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), f'checkpoints/best_model.pth')

writer.flush()
writer.close()

