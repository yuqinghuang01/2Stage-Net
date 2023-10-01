import copy
import nibabel as nib
import numpy as np
import os
import tarfile
import json
import h5py
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import random_split
from config import (
    DATASET_PATH_TRAIN, DATASET_PATH_TEST, DATASET_TYPE, VESSEL_LABEL, SHUFFLE_SEED, KFOLD,
    TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE
)

class AneurysmSegmentationUSZ(Dataset):
    """
    The base dataset class for USZ Brain Aneurysm segmentation tasks
    -- __init__()
    :param vessel_label -> represent the target vessel label
    :param dir_path -> the dataset directory path to .h5 files
    :param training_fold -> 0...(k-1) indicate which fold during training; -1 indicate testing phase
    :param transform -> optional - transforms to be applied on each instance
    """
    def __init__(self, vessel_label, dir_path, training_fold, transforms = None, mode = None) -> None:
        super(AneurysmSegmentationUSZ, self).__init__()
        #Specify the task type
        self.vessel_label = vessel_label
        #Path to extracted dataset
        self.dir = dir_path
        # 0-4 for 5-fold cross validation during training, -1 to indicate test set
        self.training_fold = training_fold
        #Meta data about the dataset
        samples = [file[:-7] for file in os.listdir(self.dir) if file.endswith("tof.h5")]
        shuffle(samples, random_state=SHUFFLE_SEED)
        self.transform = transforms
        #Calculating split number of images
        num_training_imgs =  len(samples)
        self.mode = mode

        if training_fold == -1:
            self.test = samples
        else:
            fold = num_training_imgs // KFOLD
            self.train = samples[0:fold*training_fold] + samples[fold*(training_fold+1):]
            self.val = samples[fold*training_fold:fold*(training_fold+1)]

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == "train":
            return len(self.train)
        elif self.mode == "val":
            return len(self.val)
        elif self.mode == "test":
            return len(self.test)
        else:
            raise ValueError("invalid mode.")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #Obtaining image name by given index and the mode using meta data
        if self.mode == "train":
            name = self.train[idx]
        elif self.mode == "val":
            name = self.val[idx]
        elif self.mode == "test":
            name = self.test[idx]
        else:
            raise ValueError("invalid mode.")
        img_path = os.path.join(self.dir, name+'_tof.h5')
        label_path = os.path.join(self.dir, name+'_seg.h5')
        img_object = h5py.File(img_path, 'r')
        label_object = h5py.File(label_path, 'r')
        #Converting to channel-first numpy array
        img_array = np.array(img_object['data'])
        #print("before: ", img_array.shape)
        img_array = np.moveaxis(img_array, -1, 0)
        #print("after: ", img_array.shape)
        label_array = np.array(label_object['data'])
        label_array = np.moveaxis(label_array, -1, 0)
        #Modify label array based on task type
        vessel_mask = (label_array == VESSEL_LABEL)
        label_array = np.zeros_like(label_array)
        label_array[vessel_mask] = 1
        #else use all available labels
        proccessed_out = {'name': name, 'image': img_array, 'label': label_array} 
        if self.transform:
            if self.mode == "train":
                proccessed_out = self.transform[0](proccessed_out)
            elif self.mode == "val":
                proccessed_out = self.transform[1](proccessed_out)
            elif self.mode == "test":
                proccessed_out = self.transform[2](proccessed_out)
            else:
                raise ValueError("invalid mode.")
        img_object.close()
        label_object.close()
        #The output numpy array is in channel-first format
        return proccessed_out


class SyntheticDataset(Dataset):
    def __init__(self, vessel_label, dir_path, training_fold, transforms = None, mode = None) -> None:
        super(SyntheticDataset, self).__init__()
        #Specify the task type
        self.vessel_label = vessel_label
        #Path to extracted dataset
        self.dir = dir_path
        # 0-4 for 5-fold cross validation during training, -1 to indicate test set
        self.training_fold = training_fold
        #Meta data about the dataset
        samples = [file for file in os.listdir(os.path.join(self.dir, 'raw/'))]
        shuffle(samples, random_state=SHUFFLE_SEED)
        self.transform = transforms
        #Calculating split number of images
        num_training_imgs =  len(samples)
        self.mode = mode

        if training_fold == -1:
            self.test = samples
        else:
            fold = num_training_imgs // KFOLD
            self.train = samples[0:fold*training_fold] + samples[fold*(training_fold+1):]
            self.val = samples[fold*training_fold:fold*(training_fold+1)]

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == "train":
            return len(self.train)
        elif self.mode == "val":
            return len(self.val)
        elif self.mode == "test":
            return len(self.test)
        else:
            raise ValueError("invalid mode.")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #Obtaining image name by given index and the mode using meta data
        if self.mode == "train":
            name = self.train[idx]
        elif self.mode == "val":
            name = self.val[idx]
        elif self.mode == "test":
            name = self.test[idx]
        else:
            raise ValueError("invalid mode.")
        img_path = os.path.join(self.dir, 'raw', name)
        label_path = os.path.join(self.dir, 'seg', name)
        img_object  = nib.load(img_path)
        label_object = nib.load(label_path)
        #Converting to channel-first numpy array
        img_array = img_object.get_fdata()
        #print("before: ", img_array.shape)
        img_array = np.moveaxis(img_array, -1, 0)
        #print("after: ", img_array.shape)
        label_array = label_object.get_fdata()
        label_array = np.moveaxis(label_array, -1, 0)
        #Modify label array based on task type
        assert(len(np.unique(label_array)) == 2)
        vessel_mask = (label_array == VESSEL_LABEL)
        label_array = np.zeros_like(label_array)
        label_array[vessel_mask] = 1
        #else use all available labels
        proccessed_out = {'name': name, 'image': img_array, 'label': label_array} 
        if self.transform:
            if self.mode == "train":
                proccessed_out = self.transform[0](proccessed_out)
            elif self.mode == "val":
                proccessed_out = self.transform[1](proccessed_out)
            elif self.mode == "test":
                proccessed_out = self.transform[2](proccessed_out)
            else:
                raise ValueError("invalid mode.")
        #The output numpy array is in channel-first format
        return proccessed_out


def get_train_val_test_Dataloaders(train_transforms, val_transforms, test_transforms, training_fold):
    """
    The utility function to generate splitted train, validation and test dataloaders
    
    Note: all the configs to generate dataloaders in included in "config.py"
    """

    if DATASET_TYPE == 'hdf5' and training_fold == -1:
        dataset = AneurysmSegmentationUSZ(vessel_label=VESSEL_LABEL, dir_path=DATASET_PATH_TEST, training_fold=training_fold, transforms=[train_transforms, val_transforms, test_transforms])
    elif DATASET_TYPE == 'hdf5':
        dataset = AneurysmSegmentationUSZ(vessel_label=VESSEL_LABEL, dir_path=DATASET_PATH_TRAIN, training_fold=training_fold, transforms=[train_transforms, val_transforms, test_transforms])
    elif DATASET_TYPE == 'nifti' and training_fold == -1:
        dataset = SyntheticDataset(vessel_label=VESSEL_LABEL, dir_path=DATASET_PATH_TEST, training_fold=training_fold, transforms=[train_transforms, val_transforms, test_transforms])
    elif DATASET_TYPE == 'nifti':
        dataset = SyntheticDataset(vessel_label=VESSEL_LABEL, dir_path=DATASET_PATH_TRAIN, training_fold=training_fold, transforms=[train_transforms, val_transforms, test_transforms])
    else:
        raise ValueError("Dataset type not supported currently.")
    
    #Spliting dataset and building their respective DataLoaders
    train_set, val_set, test_set = copy.deepcopy(dataset), copy.deepcopy(dataset), copy.deepcopy(dataset)
    train_set.set_mode('train')
    val_set.set_mode('val')
    test_set.set_mode('test')

    if training_fold == -1:
        test_dataloader = DataLoader(dataset=test_set, batch_size=TEST_BATCH_SIZE, shuffle=False)
        return None, None, test_dataloader
    else:
        train_dataloader = DataLoader(dataset=train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(dataset=val_set, batch_size=VAL_BATCH_SIZE, shuffle=False)
        return train_dataloader, val_dataloader, None
