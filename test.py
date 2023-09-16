import os
import numpy as np
import torch
import torchio as tio
from config import (
    NUM_CLASSES, IN_CHANNELS, BOTTLENECK_CHANNEL, BACKGROUND_AS_CLASS, TRAIN_CUDA, TEST_BATCH_SIZE,
    PATCH_SIZE_X, PATCH_SIZE_Y, PATCH_SIZE_Z
)
from dataset import get_train_val_test_Dataloaders
from torch.utils.tensorboard import SummaryWriter
from models.unet3d import UNet3D
from transforms import (train_transform, train_transform_cuda,
                        val_transform, val_transform_cuda)
from monai.visualize import plot_2d_or_3d_image
from utils import np_to_stl
from torchmetrics.functional import dice
from torchmetrics import Precision, Recall, AUROC, Accuracy, Specificity
from monai.metrics import compute_hausdorff_distance
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

if BACKGROUND_AS_CLASS: NUM_CLASSES += 1

#writer = SummaryWriter("log/")

model = UNet3D(in_channels=IN_CHANNELS , num_classes=NUM_CLASSES, level_channels=[int(BOTTLENECK_CHANNEL/8), int(BOTTLENECK_CHANNEL/4), int(BOTTLENECK_CHANNEL/2)], bottleneck_channel=BOTTLENECK_CHANNEL)
train_transforms = train_transform
val_transforms = val_transform
test_transforms = val_transform
device = 'cpu'

if torch.cuda.is_available() and TRAIN_CUDA:
    model = model.cuda()
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda
    test_transforms = val_transform_cuda
    device = 'cuda'
elif not torch.cuda.is_available() and TRAIN_CUDA:
    print('cuda not available! Training initialized on cpu ...')

# Load best model
root_dir = "checkpoints/"
model.load_state_dict(torch.load(
    os.path.join(root_dir, "vessel_ica_bottleneck64_unet_gccm.pth")))
model.eval()

_, _, test_dataloader = get_train_val_test_Dataloaders(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms, training_fold=-1)
precision = Precision(task='multiclass', num_classes=2, ignore_index=0).to(device)
accuracy = Accuracy(task='multiclass', num_classes=2, ignore_index=0).to(device)
specificity = Specificity(task='multiclass', num_classes=2, ignore_index=0).to(device)
recall = Recall(task='multiclass', num_classes=2, ignore_index=0).to(device)

#test starts
test_dice = 0.0 # 2TP/(2TP+FN+FP)
test_acc = 0.0 # (TP+TN)/(TP+TN+FP+FN)
test_recall = 0.0 # TP/(TP+FN), also called sensitivity
test_pre = 0.0 # TP/(TP+FP)
test_spe = 0.0 # TN/(TN+FP)
test_hausdorff_dist = 0.0 #deviation between predicted mask and gt

model.eval() #switch on evaluation mode
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        img, gt = data['image'], data['label']

        patch_subject = tio.Subject(data=tio.ScalarImage(tensor=img.squeeze(0)), target=tio.LabelMap(tensor=gt.squeeze(0)))
        grid_sampler = tio.inference.GridSampler(subject=patch_subject, patch_size=(PATCH_SIZE_X, PATCH_SIZE_Y, PATCH_SIZE_Z), patch_overlap=0)
        aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='crop')
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=TEST_BATCH_SIZE)
        patch_loss = 0.0
        gt = torch.squeeze(gt, dim=1).long()

        for patches_batch in patch_loader:
            data_patch = patches_batch['data'][tio.DATA].to(device)
            locations_patch = patches_batch[tio.LOCATION]
            #print(data_patch.size()) [1, 1, 64, 64, 64]
            patch_output, features = model(data_patch)
            #print(patch_output.size()) [1, C, 64, 64, 64]
            patch_output = torch.softmax(patch_output, dim=1)
            aggregator.add_batch(patch_output, locations_patch)
        
        output = aggregator.get_output_tensor().unsqueeze(0).to(device)
        test_dice += dice(output, gt, ignore_index=0)
        test_pre += precision(output, gt)
        test_acc += accuracy(output, gt)
        test_spe += specificity(output, gt)
        test_recall += recall(output, gt) #sensitivity
        output = torch.where(output > 0.5, 1, 0)
        gt = gt.unsqueeze(1)
        #print(output.size()) #[1, 2, 160, 560, 640]
        #print(gt.size()) #[1, 1, 160, 560, 640]
        test_hausdorff_dist += compute_hausdorff_distance(output, gt, percentile=95)

        output = output.squeeze(0)[1].detach().cpu().numpy().astype(int)
        gt = gt.squeeze(0)
        gt = gt.squeeze(0).detach().cpu().numpy().astype(int)

        # Visualize the model output with the input label
        #plot_2d_or_3d_image(data=target, step=0, writer=writer, max_channels=3, frame_dim=-1, tag="predict")
        print(output.shape)
        np_to_stl(output, f'./test_stl/test{i}_pred.stl')
        print(gt.shape)
        np_to_stl(gt, f'./test_stl/test{i}_gt.stl')

        '''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        gt = np.where(gt == 1)
        ax.scatter(gt[0], gt[1], gt[2], c='black')
        ax.grid(False) #turn gridlines off
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([]) #suppress tick labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        fig.tight_layout()
        plt.savefig('./test_stl/tmp.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        '''

print(f'Test Dice: {test_dice / len(test_dataloader)}')
print(f'Test Precision: {test_pre / len(test_dataloader)}')
print(f'Test Accuracy: {test_acc / len(test_dataloader)}')
print(f'Test Specificity: {test_spe / len(test_dataloader)}')
print(f'Test Recall: {test_recall / len(test_dataloader)}')
print(f'Test Hausdorff Distance: {test_hausdorff_dist / len(test_dataloader)}')

#writer.flush()
#writer.close()
