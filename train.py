import math
import numpy as np
import torch
import torchio as tio
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BOTTLENECK_CHANNEL, BACKGROUND_AS_CLASS, TRAIN_CUDA,
    VAL_BATCH_SIZE, PATCH_SIZE_X, PATCH_SIZE_Y, PATCH_SIZE_Z, BCE_WEIGHTS,
    DROPOUT, ALPHA, HIDDEN, NUM_ATTEN_HEADS, WEIGHT_DECAY
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
from utils import load_data, build_graph, accuracy
from models.gat import GAT, SpGAT

if BACKGROUND_AS_CLASS: NUM_CLASSES += 1

NUM_FEATURES = int(BOTTLENECK_CHANNEL/8)

writer = SummaryWriter("logs")

unet = UNet3D(in_channels=IN_CHANNELS , 
              num_classes=NUM_CLASSES, 
              level_channels=[int(BOTTLENECK_CHANNEL/8), int(BOTTLENECK_CHANNEL/4), int(BOTTLENECK_CHANNEL/2)], bottleneck_channel=BOTTLENECK_CHANNEL)
gat = GAT(nfeat=NUM_FEATURES, 
          nhid=HIDDEN, 
          nclass=NUM_CLASSES, 
          dropout=DROPOUT, 
          nheads=NUM_ATTEN_HEADS, 
          alpha=ALPHA)
train_transforms = train_transform
val_transforms = val_transform
criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS))
device = 'cpu'

if torch.cuda.is_available() and TRAIN_CUDA:
    unet = unet.cuda()
    gat = gat.cuda()
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda
    criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS).to(device='cuda'))
    device = 'cuda'
elif not torch.cuda.is_available() and TRAIN_CUDA:
    print('cuda not available! Training initialized on cpu ...')


train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)


optimizer_unet = Adam(params=unet.parameters())
optimizer_gat = Adam(params=gat.parameters(), weight_decay=WEIGHT_DECAY)

min_valid_loss = math.inf

for epoch in range(TRAINING_EPOCH):
    
    train_loss = 0.0
    train_metric = 0.0
    unet.train()
    for crops in train_dataloader:
        crops_loss = 0.0
        metric = 0.0
        for data in crops:
            img, gt = data['image'], data['label']
            gt = torch.squeeze(gt, dim=1).long()
        
            optimizer_unet.zero_grad()

            output, feature = unet(img) #model returns last layer features so we can insert GCCM plug-in here (based on GAT)
            #print(output.size()) #[1, C, 64, 64, 64]
            #print(gt.size()) #[1, 64, 64, 64]

            #construct graph from gt, cnn features as node features
            vesselness = gt.squeeze(0).detach().cpu().numpy().astype(np.double)
            feature = torch.moveaxis(feature, 1, -1) #print(feature.size()) #[1, 64, 64, 64, 8]
            adj, features, labels, idx_train = build_graph(vesselness, feature.squeeze(0))
            features, adj, labels = Variable(features), Variable(adj), Variable(labels)

            if device == 'cuda':
                features = features.cuda()
                adj = adj.cuda()
                labels = labels.cuda()
                idx_train = idx_train.cuda()
            
            #feed graph to GAT network
            gat.train()
            optimizer_gat.zero_grad()
            output_gat = gat(features, adj)
            loss_train = F.nll_loss(output_gat[idx_train], labels[idx_train])
            acc_train = accuracy(output_gat[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer_gat.step()

            print('GAT:',
                  'loss_train: {:.4f}'.format(loss_train.data.item()),
                  'acc_train: {:.4f}'.format(acc_train.data.item()))

            #sum up unet loss and connectivity constraints loss
            loss = loss_train + criterion(output, gt)
            crops_loss += loss.item()

            output = torch.softmax(output, dim=1)
            metric += dice(output, gt, ignore_index=0)
            print(metric)
            loss.backward()
            optimizer_unet.step()

        train_loss += crops_loss / len(crops)
        train_metric += metric.item() / len(crops)
        
    #validation starts
    valid_loss = 0.0
    valid_metric = 0.0
    unet.eval() #switch on evaluation mode
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
                patch_output, feature = unet(data_patch)
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
        torch.save(unet.state_dict(), f'checkpoints/best_model.pth')

writer.flush()
writer.close()
