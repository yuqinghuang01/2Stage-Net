import os
import torch
import torchio as tio
from config import (
    NUM_CLASSES, IN_CHANNELS, BOTTLENECK_CHANNEL, BACKGROUND_AS_CLASS, TRAIN_CUDA,
    VAL_BATCH_SIZE, PATCH_SIZE_X, PATCH_SIZE_Y, PATCH_SIZE_Z, TASK_ID, EDGE_DIST_THRESH, WINDOW_SIZE
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
    os.path.join(root_dir, "vessel_ica_bottleneck64_unet.pth")))
model.eval()

train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms= train_transforms, val_transforms=val_transforms, test_transforms= val_transforms)

def save_graph_viz(vesselness, save_path):
    #find local maxima
    max_val = []
    max_pos = []
    x_quan = range(0, vesselness.shape[0], WINDOW_SIZE)
    y_quan = range(0, vesselness.shape[1], WINDOW_SIZE)
    z_quan = range(0, vesselness.shape[2], WINDOW_SIZE)
    for x_idx in x_quan:
        for y_idx in y_quan:
            for z_idx in z_quan:
                cur_win = vesselness[x_idx:x_idx+WINDOW_SIZE, y_idx:y_idx+WINDOW_SIZE, z_idx:z_idx+WINDOW_SIZE]
                if np.sum(cur_win) == 0:
                    max_val.append(0)
                    max_pos.append((int(x_idx+WINDOW_SIZE/2), int(y_idx+WINDOW_SIZE/2), int(z_idx+WINDOW_SIZE/2)))
                else:
                    #print(np.max(cur_win))
                    max_val.append(np.max(cur_win))
                    temp = np.unravel_index(cur_win.argmax(), cur_win.shape)
                    max_pos.append((int(x_idx+temp[0]), int(y_idx+temp[0]), int(z_idx+temp[0])))
        
    #construct graph
    graph = nx.Graph()
    pos = {}

    #add nodes
    for node_idx, (node_x, node_y, node_z) in enumerate(max_pos):
        pos[node_idx] = [node_x, node_y, node_z]
        graph.add_node(node_idx, x=node_x, y=node_y, z=node_z, label=node_idx)
        #print('node label', node_idx, 'pos', (node_x, node_y, node_z), 'added')

    #add edges 
    node_list = list(graph.nodes)
    for i, n in enumerate(tqdm(node_list)):
        if vesselness[graph.nodes[n]['x'], graph.nodes[n]['y'], graph.nodes[n]['z']] == 0:
            continue
        neighbor = vesselness[max(0, graph.nodes[n]['x']-1):min(PATCH_SIZE_X, graph.nodes[n]['x']+2), \
                              max(0, graph.nodes[n]['y']-1):min(PATCH_SIZE_Y, graph.nodes[n]['y']+2), \
                              max(0, graph.nodes[n]['z']-1):min(PATCH_SIZE_Z, graph.nodes[n]['z']+2)]

        if np.mean(neighbor) < 0.1:
            continue

        phi = np.ones_like(vesselness)
        phi[graph.nodes[n]['x'], graph.nodes[n]['y'], graph.nodes[n]['z']] = -1
        print(np.unique(phi))
        tt = skfmm.travel_time(phi, vesselness, narrow=EDGE_DIST_THRESH)

        for n_comp in node_list[i+1:]:
            geo_dist = tt[graph.nodes[n_comp]['x'], graph.nodes[n_comp]['y'], graph.nodes[n_comp]['z']] #travel time
            if geo_dist < EDGE_DIST_THRESH:
                #print(geo_dist)
                graph.add_edge(n, n_comp, weight=EDGE_DIST_THRESH/(EDGE_DIST_THRESH+geo_dist))
                #print('An edge between', 'node', n, '&', n_comp, 'is constructed')
        
    #extract node and edge positions
    node_xyz = np.array([pos[v] for v in sorted(graph)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in graph.edges()])

    #visualize graph
    #create 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(*node_xyz.T, s=50, ec='w', color='green')  
    #plot the degs
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color='blue')        
        
    ax.grid(False) #turn gridlines off
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([]) #suppress tick labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

with torch.no_grad():
    for crops in train_dataloader:
        data = crops[0]
        img, gt = data['image'], data['label']
        gt = torch.squeeze(gt, dim=1).long()

        vesselness = gt.squeeze(0).detach().cpu().numpy().astype(np.double)
        save_path = os.path.join('graphs', data['name'][0] + '_graph_gt.png')
        save_graph_viz(vesselness, save_path)

        output = model(img)
        output = torch.softmax(output, dim=1).squeeze(0)[1]
        vesselness = output.detach().cpu().numpy().astype(np.double)
        save_path = os.path.join('graphs', data['name'][0] + '_graph_pred.png')
        save_graph_viz(vesselness, save_path)

plt.cla()
plt.clf()
plt.close()