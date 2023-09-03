import os
import torch
import torchio as tio
from config import (
    NUM_CLASSES, IN_CHANNELS, BOTTLENECK_CHANNEL, BACKGROUND_AS_CLASS, TRAIN_CUDA,
    VAL_BATCH_SIZE, PATCH_SIZE_X, PATCH_SIZE_Y, PATCH_SIZE_Z, TASK_ID, EDGE_DIST_THRESH
)
from dataset import get_train_val_test_Dataloaders
from models.unet3d import UNet3D
from transforms import (train_transform, train_transform_cuda,
                        val_transform, val_transform_cuda)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import skfmm
from tqdm import tqdm

if BACKGROUND_AS_CLASS: NUM_CLASSES += 1

vessels = []
if TASK_ID == 3:
    vessels = [1]
elif TASK_ID == 4:
    vessels = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

model = UNet3D(in_channels=IN_CHANNELS , num_classes=NUM_CLASSES, level_channels=[int(BOTTLENECK_CHANNEL/8), int(BOTTLENECK_CHANNEL/4), int(BOTTLENECK_CHANNEL/2)], bottleneck_channel=BOTTLENECK_CHANNEL)
train_transforms = train_transform
val_transforms = val_transform
device = 'cpu'

if torch.cuda.is_available() and TRAIN_CUDA:
    model = model.cuda()
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda
    device = 'cuda'
elif not torch.cuda.is_available() and TRAIN_CUDA:
    print('cuda not available! Training initialized on cpu ...')

# Load best model
root_dir = "checkpoints/"
model.load_state_dict(torch.load(
    os.path.join(root_dir, "best_model.pth")))
model.eval()

train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)

with torch.no_grad():
    for data in val_dataloader:
        img, gt = data['image'], data['label']

        patch_subject = tio.Subject(data=tio.ScalarImage(tensor=img.squeeze(0)), target=tio.LabelMap(tensor=gt.squeeze(0)))
        grid_sampler = tio.inference.GridSampler(subject=patch_subject, patch_size=(PATCH_SIZE_X, PATCH_SIZE_Y, PATCH_SIZE_Z), patch_overlap=0)
        aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='crop')
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=VAL_BATCH_SIZE)
        gt = torch.squeeze(gt, dim=1).long()

        for patches_batch in patch_loader:
            data_patch = patches_batch['data'][tio.DATA].to(device)
            locations_patch = patches_batch[tio.LOCATION]
            patch_output = model(data_patch)
            patch_gt = torch.squeeze(patches_batch['target'][tio.DATA].to(device), dim=1).long()
            if NUM_CLASSES > 1:
                patch_output = torch.softmax(patch_output, dim=1).squeeze(0)
            else:
                patch_output = torch.sigmoid(patch_output).squeeze(0)
            
            #construct graph
            graph = nx.Graph()
            pos = {}

            #add nodes
            for index in np.ndindex(PATCH_SIZE_X, PATCH_SIZE_Y, PATCH_SIZE_Z):
                node_idx = index[0]*PATCH_SIZE_Y*PATCH_SIZE_Z + index[1]*PATCH_SIZE_Z + index[2]
                pos[node_idx] = (index[1], index[2])
                graph.add_node(node_idx, x=index[0], y=index[1], z=index[2], label=node_idx)
                print('node label', node_idx, 'pos', (index[0], index[1], index[2]), 'added')

            #add edges 
            node_list = list(graph.nodes)
            for vessel in vessels:
                pred = patch_output[vessel].detach().cpu().numpy().astype(np.double)
                for i, n in enumerate(tqdm(node_list)):
                    if n == 1:
                        break
                    if pred[graph.nodes[n]['x'], graph.nodes[n]['y'], graph.nodes[n]['z']] == 0:
                        continue
                    neighbor = pred[max(0, graph.nodes[n]['x']-1):min(PATCH_SIZE_X, graph.nodes[n]['x']+2), \
                                    max(0, graph.nodes[n]['y']-1):min(PATCH_SIZE_Y, graph.nodes[n]['y']+2), \
                                    max(0, graph.nodes[n]['z']-1):min(PATCH_SIZE_Z, graph.nodes[n]['z']+2)]

                    if np.mean(neighbor) < 0.1:
                        continue

                    phi = np.ones_like(pred)
                    phi[graph.nodes[n]['x'], graph.nodes[n]['y'], graph.nodes[n]['z']] = -1
                    tt = skfmm.travel_time(phi, pred, narrow=EDGE_DIST_THRESH)

                    for n_comp in node_list[i+1:]:
                        geo_dist = tt[graph.nodes[n_comp]['x'], graph.nodes[n_comp]['y'], graph.nodes[n_comp]['z']] #travel time
                        if geo_dist < EDGE_DIST_THRESH:
                            #print(geo_dist)
                            graph.add_edge(n, n_comp, weight=EDGE_DIST_THRESH/(EDGE_DIST_THRESH+geo_dist), label=vessel)
                            print('An edge between', 'node', n, '&', n_comp, 'with label', vessel, 'is constructed')

            #visualize graph                
            nx.draw(graph, pos)
            plt.savefig("graph.png")
            break
