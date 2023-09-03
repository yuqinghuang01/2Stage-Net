import math
import torch
import torchio as tio
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BOTTLENECK_CHANNEL, BACKGROUND_AS_CLASS, TRAIN_CUDA,
    VAL_BATCH_SIZE, PATCH_SIZE_X, PATCH_SIZE_Y, PATCH_SIZE_Z, BCE_WEIGHTS
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

writer = SummaryWriter("logs")

model = UNet3D(in_channels=IN_CHANNELS , num_classes=NUM_CLASSES, level_channels=[int(BOTTLENECK_CHANNEL/8), int(BOTTLENECK_CHANNEL/4), int(BOTTLENECK_CHANNEL/2)], bottleneck_channel=BOTTLENECK_CHANNEL)
train_transforms = train_transform
val_transforms = val_transform
criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS))
device = 'cpu'

if torch.cuda.is_available() and TRAIN_CUDA:
    model = model.cuda()
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda
    criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS).to(device='cuda'))
    device = 'cuda'
elif not torch.cuda.is_available() and TRAIN_CUDA:
    print('cuda not available! Training initialized on cpu ...')


train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)


optimizer = Adam(params=model.parameters())

min_valid_loss = math.inf

for epoch in range(TRAINING_EPOCH):
    
    train_loss = 0.0
    train_metric = 0.0
    model.train()
    for crops in train_dataloader:
        crops_loss = 0.0
        metric = 0.0
        for data in crops:
            img, gt = data['image'], data['label']
            gt = torch.squeeze(gt, dim=1).long()
        
            optimizer.zero_grad()

            output = model(img)
            #print(output.size()) #[1, C, 64, 64, 64]
            #print(gt.size()) #[1, 64, 64, 64]
            loss = criterion(output, gt)
            crops_loss += loss.item()

            output = torch.softmax(output, dim=1)
            metric += dice(output, gt, ignore_index=0)
            print(metric)
            loss.backward()
            optimizer.step()

        train_loss += crops_loss / len(crops)
        train_metric += metric.item() / len(crops)
        
    #validation starts
    valid_loss = 0.0
    valid_metric = 0.0
    model.eval() #switch on evaluation mode
    with torch.no_grad():
        for data in val_dataloader:
            img, gt = data['image'], data['label']

            patch_subject = tio.Subject(data=tio.ScalarImage(tensor=img.squeeze(0)), target=tio.LabelMap(tensor=gt.squeeze(0)))
            grid_sampler = tio.inference.GridSampler(subject=patch_subject, patch_size=(PATCH_SIZE_X, PATCH_SIZE_Y, PATCH_SIZE_Z), patch_overlap=0)
            aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='crop')
            patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=VAL_BATCH_SIZE)
            patch_loss = 0.0
            gt = torch.squeeze(gt, dim=1).long()

            for patches_batch in patch_loader:
                data_patch = patches_batch['data'][tio.DATA].to(device)
                locations_patch = patches_batch[tio.LOCATION]
                #print(data_patch.size()) [1, 1, 64, 64, 64]
                patch_output = model(data_patch)
                #print(patch_output.size()) [1, C, 64, 64, 64]
                patch_gt = torch.squeeze(patches_batch['target'][tio.DATA].to(device), dim=1).long()
                patch_loss += criterion(patch_output, patch_gt)
                patch_output = torch.softmax(patch_output, dim=1)
                aggregator.add_batch(patch_output, locations_patch)
        
            output = aggregator.get_output_tensor().unsqueeze(0).to(device)
            patch_loss = patch_loss / len(patch_loader)
            valid_loss += patch_loss.item()
            print(output.size())
            #print(gt.size())
            valid_metric += dice(output, gt, ignore_index=0)
        
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

