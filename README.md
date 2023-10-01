# Pytorch implementation of [3D U-Net](https://arxiv.org/pdf/1606.06650v1.pdf) with [GCCM](https://pubmed.ncbi.nlm.nih.gov/34613925/)

This implementation is an extension of the method proposed in the paper [3D Graph-Connectivity Constrained Network for Hepatic Vessel Segmentation](https://pubmed.ncbi.nlm.nih.gov/34613925/). We provided an implementation of U-Net with GAT-based GCCM. Then we proposed a new feature sampling strategy of integrating GAT with U-Net during the training stage. Experiments of binary brain artery segmentation on a synthetic dataset and an in-house real dataset show that incorporating graph neural networks during the training stage can improve test-time accuracy as well as help with reducing false positives and segmenting thinner vessels. Therefore, the results confirm the potential of combining CNNs with GNNs to improve the accuracy and connectivity of vessel segmentation.

## Model Architecture

GCCM based on GAT is integrated with U-Net during the training stage to add connectivity constraints. Two feature sampling strategies (circled in red) are implemented in this repository, as well as the baseline method U-Net.

![UNet-GCCM](https://github.com/AghdamAmir/3D-UNet/blob/main/UNet-GCCM.png)

## Dataset

The synthetic data used in the experiment is from [DeepVesselNet Datasets](https://github.com/giesekow/deepvesselnet/wiki/Datasets#time-of-flight-tof-magnetic-resonance-angiography-mra-data). The dataset is in nifti format, and nibabel library is used to read in the data.

The real dataset cannot be shared, unfortunately. However, the dataset is in format of hdf5. Ground truth segmentation masks are named "*_seg.h5", and TOF-MRAs are named "*_tof.h5". All files are stored in the same directory.

The Dataset class uses Monai package for applying augmentations on them in the transform.py file. You can modify the applied transformation in this file according to your preferences.

## Configure the network

All the configurations and hyperparameters are set in the config.py file. Please refer to the documentation inside the file to set the corresponding parameters. In particular, 
- To use the model U-Net, ensure to set GCCM as False;
- To use the model UNet-GCCM-max, set GCCM as True and FEATURE_SAMPLING as 'max';
- To use the model UNet-GCCM-avg, set GCCM as True and FEATURE_SAMPLING as 'avg'.

## Training

After configure config.py, you can start to train by running

`python train.py`

We also employ tensorboard to visualize the training process.

`tensorboard --logdir=logs/`

## Testing

After training is done, there will be files of model weights stored in the checkpoints/ directory. Use these checkpoints for testing. Note that the evaluation metrics will be printed out when testing is done. In addition, for qualitative analysis, the .stl files representing the mesh extracted from ground truth and predictions will be stored in the specified directory.
