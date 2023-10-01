import math
import os
import numpy as np
import torch
import torchio as tio
import torch.nn.functional as F
import torch.optim as optim
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BOTTLENECK_CHANNEL, BACKGROUND_AS_CLASS, TRAIN_CUDA, TRAIN_BATCH_SIZE, KFOLD, LOG_DIR, MODEL_NAME,
    VAL_BATCH_SIZE, PATCH_SIZE_X, PATCH_SIZE_Y, PATCH_SIZE_Z, NLL_WEIGHT, WINDOW_SIZE, PATIENCE, GCCM,
    FEATURE_SAMPLING, DROPOUT, ALPHA, HIDDEN, NUM_ATTEN_HEADS, WEIGHT_DECAY_UNET, WEIGHT_DECAY_GAT, ALPHA_WEIGHT, BETA_WEIGHT
)
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torchmetrics.functional import dice
from torchmetrics import Precision, Recall, AUROC, Accuracy, Specificity
from monai.metrics import compute_hausdorff_distance
from dataset import get_train_val_test_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from models.unet3d import UNet3D
from transforms import (train_transform, train_transform_cuda,
                        val_transform, val_transform_cuda)
from utils import build_graph, accuracy
from models.gat import GAT
from monai.losses import DiceLoss

if BACKGROUND_AS_CLASS: NUM_CLASSES += 1

NUM_FEATURES = int(BOTTLENECK_CHANNEL/8)

train_transforms = train_transform
val_transforms = val_transform
#criterion = CrossEntropyLoss(weight=torch.Tensor(NLL_WEIGHTS))
criterion = DiceLoss()
device = 'cpu'

for fold in range(1,KFOLD):

    count_no_improve = 0

    unet = UNet3D(in_channels=IN_CHANNELS, 
              num_classes=NUM_CLASSES, 
              level_channels=[int(BOTTLENECK_CHANNEL/8), int(BOTTLENECK_CHANNEL/4), int(BOTTLENECK_CHANNEL/2)], bottleneck_channel=BOTTLENECK_CHANNEL)
    gat = GAT(nfeat=NUM_FEATURES, 
          nhid=HIDDEN, 
          nclass=NUM_CLASSES, 
          dropout=DROPOUT, 
          nheads=NUM_ATTEN_HEADS, 
          alpha=ALPHA)
    #total_params = sum(p.numel() for p in unet.parameters())
    #print(total_params)
    #total_params = sum(p.numel() for p in gat.parameters())
    #print(total_params)
    #break

    if torch.cuda.is_available() and TRAIN_CUDA:
        unet = unet.cuda()
        gat = gat.cuda()
        train_transforms = train_transform_cuda
        val_transforms = val_transform_cuda
        #criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS).to(device='cuda'))
        device = 'cuda'
    elif not torch.cuda.is_available() and TRAIN_CUDA:
        print('cuda not available! Training initialized on cpu ...')


    optimizer_unet = Adam(params=unet.parameters(), weight_decay=WEIGHT_DECAY_UNET)
    optimizer_gat = Adam(params=gat.parameters(), weight_decay=WEIGHT_DECAY_GAT)

    precision = Precision(task='binary').to(device)
    acc = Accuracy(task='binary').to(device)
    specificity = Specificity(task='binary').to(device)
    recall = Recall(task='binary').to(device)
    
    min_valid_loss = math.inf

    writer = SummaryWriter(os.path.join(LOG_DIR, f"runs_split{fold+1}"))
    print(f"Fold {fold+1}")
    print("----------------------")

    train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms= val_transforms, training_fold=fold)

    for epoch in range(TRAINING_EPOCH):
    
        train_loss = 0.0
        train_loss_gat = 0.0
        train_loss_unet = 0.0
        train_metric = 0.0
        unet.train()
        for crops in train_dataloader:
            crops_loss = 0.0
            crops_loss_gat = 0.0
            crops_loss_unet = 0.0
            metric = 0.0
            for data in crops:
                img, gt = data['image'], data['label']
                gt = torch.squeeze(gt, dim=1).long()

                optimizer_unet.zero_grad()

                output, features = unet(img) #model returns last layer features so we can insert GCCM plug-in here (based on GAT)
                output = torch.softmax(output, dim=1)
                #print(output.size()) #[1, C, 64, 64, 64]
                #print(gt.size()) #[1, 64, 64, 64]

                loss_gat = 0.0
                if GCCM:
                    #construct graph from gt, sample cnn features as node features
                    vesselness = gt.squeeze(0).cpu().detach().numpy()
                    adj, feature_mask, labels, idx_train = build_graph(vesselness, FEATURE_SAMPLING)

                    #apply feature mask to feature
                    if FEATURE_SAMPLING == 'avg':
                        features = torch.moveaxis(features, 1, -1).squeeze(0) #print(features.size()) #[64, 64, 64, 8]
                        feature_mask = torch.from_numpy(feature_mask).to(device).unsqueeze(-1)
                        pool = torch.nn.AvgPool3d(WINDOW_SIZE)
                        features = torch.moveaxis(features * feature_mask, -1, 0)
                        features = pool(features)
                        features = features * (WINDOW_SIZE ** 3)
                        features = torch.moveaxis(features, 0, -1)
                        features = F.normalize(features, dim=3)
                        features = torch.flatten(features, 0, 2).float()
                        #print(features.size()) #[512, 8]
                    else:
                        features = torch.moveaxis(features, 1, -1).squeeze(0) #print(features.size()) #[64, 64, 64, 8]
                        features = F.normalize(features, dim=3)
                        feature_mask = torch.from_numpy(feature_mask).bool().to(device)
                        features = features[feature_mask,:]

                    if device == 'cuda':
                        adj = adj.cuda()
                        labels = labels.cuda()
                        idx_train = idx_train.cuda()
            
                    #feed graph to GAT network
                    gat.train()
                    optimizer_gat.zero_grad()
                    output_gat = gat(features, adj)
                    loss_gat = F.nll_loss(output_gat[idx_train], labels[idx_train], weight=torch.Tensor(NLL_WEIGHT).to(device))
                    crops_loss_gat += loss_gat.item()
                    acc_gat = accuracy(output_gat[idx_train], labels[idx_train])
                    #print('GAT:',
                    #    'loss_train: {:.4f}'.format(loss_gat.data.item()),
                    #    'acc_train: {:.4f}'.format(acc_gat.data.item()))

                #sum up unet loss and connectivity constraints loss
                metric += dice(output, gt, ignore_index=0)
                gt = gt.unsqueeze(1)
                output = output[:,1].unsqueeze(1)
                loss_unet = criterion(output, gt)
                #print(metric)
                #print(loss_unet)

                loss = loss_unet
                crops_loss_unet += loss_unet.item()
                if GCCM:
                    loss = BETA_WEIGHT * loss_gat + ALPHA_WEIGHT * loss_unet
                    crops_loss += loss.item()
                    loss.backward()
                else:
                    crops_loss += loss.item()
                    loss_unet.backward()

                #print(unet.s_block1.conv2.weight.grad)
                #for param in gat.parameters():
                #    print(param.grad)

                optimizer_gat.step()
                optimizer_unet.step()

            train_loss += crops_loss / len(crops)
            train_loss_gat += crops_loss_gat / len(crops)
            train_loss_unet += crops_loss_unet / len(crops)
            train_metric += metric.item() / len(crops)
        
        #validation starts
        valid_loss = 0.0
        valid_dice = 0.0 # 2TP/(2TP+FN+FP)
        valid_acc = 0.0 # (TP+TN)/(TP+TN+FP+FN)
        valid_recall = 0.0 # TP/(TP+FN), also called sensitivity
        valid_pre = 0.0 # TP/(TP+FP)
        valid_spe = 0.0 # TN/(TN+FP)
        valid_hausdorff_dist = 0.0 #deviation between predicted mask and gt
        unet.eval() #switch on evaluation mode
        with torch.no_grad():
            for data in val_dataloader:
                img, gt = data['image'], data['label']

                patch_subject = tio.Subject(data=tio.ScalarImage(tensor=img.squeeze(0)), target=tio.LabelMap(tensor=gt.squeeze(0)))
                grid_sampler = tio.inference.GridSampler(subject=patch_subject, patch_size=(PATCH_SIZE_X, PATCH_SIZE_Y, PATCH_SIZE_Z), patch_overlap=0)
                aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='crop')
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=VAL_BATCH_SIZE)
                #patch_loss = 0.0
                gt = torch.squeeze(gt, dim=1).long()

                for patches_batch in patch_loader:
                    data_patch = patches_batch['data'][tio.DATA].to(device)
                    locations_patch = patches_batch[tio.LOCATION]
                    #print(data_patch.size()) [1, 1, 64, 64, 64]
                    patch_output, features = unet(data_patch)
                    #print(patch_output.size()) [1, C, 64, 64, 64]
                    #patch_gt = torch.squeeze(patches_batch['target'][tio.DATA].to(device), dim=1).long()
                    #patch_loss += criterion(patch_output, patch_gt)
                    patch_output = torch.softmax(patch_output, dim=1)
                    aggregator.add_batch(patch_output, locations_patch)

                output = aggregator.get_output_tensor().unsqueeze(0).to(device)
                valid_dice += dice(output, gt, ignore_index=0)
                output = output[:,1]
                valid_pre += precision(output, gt)
                valid_acc += acc(output, gt)
                valid_spe += specificity(output, gt)
                valid_recall += recall(output, gt) #sensitivity
                gt = gt.unsqueeze(1)
                output = output.unsqueeze(1)
                valid_loss += criterion(output, gt)
                output = torch.where(output > 0.5, 1, 0)
                valid_hausdorff_dist += compute_hausdorff_distance(output, gt, percentile=95)
        
        writer.add_scalar('loss/train', train_loss / len(train_dataloader), epoch+1)
        writer.add_scalar('loss/val', valid_loss / len(val_dataloader), epoch+1)
        writer.add_scalar('loss/train_unet', train_loss_unet / len(train_dataloader), epoch+1)
        writer.add_scalar('loss/train_gat', train_loss_gat / len(train_dataloader), epoch+1)

        writer.add_scalar('metric/train_dice', train_metric / len(train_dataloader), epoch+1)
        writer.add_scalar('metric/val_dice', valid_dice / len(val_dataloader), epoch+1)
        writer.add_scalar('metric/val_precision', valid_pre / len(val_dataloader), epoch+1)
        writer.add_scalar('metric/val_accuracy', valid_acc / len(val_dataloader), epoch+1)
        writer.add_scalar('metric/val_specificity', valid_spe / len(val_dataloader), epoch+1)
        writer.add_scalar('metric/val_recall', valid_recall / len(val_dataloader), epoch+1)
        writer.add_scalar('metric/val_hausdorff', valid_hausdorff_dist / len(val_dataloader), epoch+1)
    
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(val_dataloader)}')
        print(f'Epoch {epoch+1} \t\t Training Metric: {train_metric / len(train_dataloader)} \t\t Validation Metric: {valid_dice / len(val_dataloader)}')

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(unet.state_dict(), os.path.join('checkpoints/', MODEL_NAME + f'_fold{fold+1}.pth'))
            count_no_improve = 0
        else:
            count_no_improve += 1
        
        if count_no_improve >= PATIENCE: break
        print(count_no_improve)


    writer.flush()
    writer.close()

print('done----------')
