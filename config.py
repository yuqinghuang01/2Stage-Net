
"""""
Dataset configurations:
    :param DATASET_PATH -> the directory path to dataset .tar files or 'hdf5' files
    :param DATASET_TYPE -> support 'tar' (Medical Decathlon challenge) or 'hdf5' (USZ brain artery data)
    :param TASK_ID -> specifies the the segmentation task ID (see the dict below for hints)
    :param IN_CHANNELS -> number of input channels
    :param NUM_CLASSES -> specifies the number of output channels for dispirate classes
    :param BACKGROUND_AS_CLASS -> if True, the model treats background as a class

"""""
DATASET_PATH_TRAIN = 'Datasets/Synthetic/train/'
DATASET_PATH_TEST = 'Datasets/Synthetic/test/'
LOG_DIR = 'logs_synthetic_ours'
MODEL_NAME = 'synthetic_ours'
DATASET_TYPE = 'nifti' #'hdf5' 'nifti'
VESSEL_LABEL = 1
IN_CHANNELS = 1
NUM_CLASSES = 1
BOTTLENECK_CHANNEL = 64
BACKGROUND_AS_CLASS = True
SHUFFLE_SEED = 42

"""""
U-Net training configurations:
    :param TRAIN_VAL_TEST_SPLIT -> delineates the ratios in which the dataset shoud be splitted. The length of the array should be 3.
    :param SPLIT_SEED -> the random seed with which the dataset is splitted
    :param TRAINING_EPOCH -> number of training epochs
    :param VAL_BATCH_SIZE -> specifies the batch size of the training DataLoader
    :param TEST_BATCH_SIZE -> specifies the batch size of the test DataLoader
    :param TRAIN_CUDA -> if True, moves the model and inference onto GPU
    :param BCE_WEIGHTS -> the class weights for the Binary Cross Entropy loss
"""""
KFOLD = 3
CROP_RATIO = [0.1, 0.9]
TRAIN_CROP_SAMPLES = 1
PATCH_SIZE_X = 64
PATCH_SIZE_Y = 64
PATCH_SIZE_Z = 64
TRAINING_EPOCH = 100
TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
NLL_WEIGHT = [0.2, 0.8]
TRAIN_CUDA = True
WEIGHT_DECAY_UNET = 1e-3
ALPHA_WEIGHT = 1.0
PATIENCE = 10

"""""
GAT training configurations:
    :param DROPOUT -> dropout rate (1 - keep probability)
    :param ALPHA -> Alpha value for the leakyrelu
    :param HIDDEN -> number of hidden units in GAT
    :param NUM_ATTEN_HEADS -> number of head attentions
    :param WEIGHT_DECAY -> weight decay (L2 loss on parameters)
"""""
GCCM = True
FEATURE_SAMPLING = 'avg' #'max'
EDGE_DIST_THRESH = 10
WINDOW_SIZE = 8
DROPOUT = 0.5
ALPHA = 0.2
HIDDEN = 8
NUM_ATTEN_HEADS = 8
WEIGHT_DECAY_GAT = 5e-3
BETA_WEIGHT = 0.2
