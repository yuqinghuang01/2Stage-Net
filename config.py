
"""""
Dataset configurations:
    :param DATASET_PATH_TRAIN -> the directory path to training dataset
    :param DATASET_PATH_TEST -> the directory path to test dataset
    :param LOG_DIR -> directory to store tensorboard log files
    :param MODEL_NAME -> model name to store in checkpoints/
    :param DATASET_TYPE -> 'hdf5' (USZ brain artery data) or 'nifti' (synthetic data)
    :param VESSEL_LABEL -> label of the target vessel in ground truth
    :param IN_CHANNELS -> number of input channels
    :param NUM_CLASSES -> specifies the number of output channels for dispirate classes
    :param BOTTLENECK_CHANNEL -> size of the bottleneck channel in U-Net
    :param BACKGROUND_AS_CLASS -> if True, the model treats background as a class
    :param SHUFFLE_SEED -> seed to randomly shuffle training images
"""""
DATASET_PATH_TRAIN = 'Datasets/Synthetic/train/'
DATASET_PATH_TEST = 'Datasets/Synthetic/test/'
LOG_DIR = 'logs_synthetic_unet'
MODEL_NAME = 'synthetic_unet'
DATASET_TYPE = 'nifti' #'hdf5' 'nifti'
VESSEL_LABEL = 1
IN_CHANNELS = 1
NUM_CLASSES = 1
BOTTLENECK_CHANNEL = 64
BACKGROUND_AS_CLASS = True
SHUFFLE_SEED = 42

"""""
U-Net training configurations:
    :param KFOLD -> specify k-fold cross-validation
    :param CROP_RATIO -> ratio to crop the training image by classes
    :param TRAIN_CROP_SAMPLES -> number of samples to crop per training image
    :param PATCH_SIZE_X, PATCH_SIZE_Y, PATCH_SIZE_Z -> training patch size
    :param TRAINING_EPOCH -> number of training epochs
    :param VAL_BATCH_SIZE -> specifies the batch size of the training DataLoader
    :param TEST_BATCH_SIZE -> specifies the batch size of the test DataLoader
    :param TRAIN_CUDA -> if True, moves the model and inference onto GPU
    :param WEIGHT_DECAY_UNET -> weight decay used in U-Net optimizer
    :param ALPHA_WEIGHT -> weight for segmentation loss in the overall loss function
    :param PATIENCE -> early stopping if validation loss does not decrease for more than (PATIENCE) epochs
"""""
KFOLD = 3
CROP_RATIO = [0.1, 0.9]
TRAIN_CROP_SAMPLES = 1
PATCH_SIZE_X = 64
PATCH_SIZE_Y = 64
PATCH_SIZE_Z = 64
TRAINING_EPOCH = 50
TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
TRAIN_CUDA = True
WEIGHT_DECAY_UNET = 1e-3
ALPHA_WEIGHT = 1.0
PATIENCE = 5

"""""
GAT training configurations:
    :param GCCM -> whether to use GCCM or not (reduce to U-Net)
    :param FEATURE_SAMPLING -> which feature sampling strategy
    :param EDGE_DIST_THRESH -> edge distance threshold to include edge in the constructed graph
    :param WINDOW_SIZE -> node sampling interval
    :param DROPOUT -> dropout rate (1 - keep probability)
    :param ALPHA -> Alpha value for the leakyrelu
    :param HIDDEN -> number of hidden units in GAT
    :param NUM_ATTEN_HEADS -> number of head attentions
    :param WEIGHT_DECAY_GAT -> weight decay (L2 loss on parameters) used in GAT optimizer
    :param BETA_WEIGHT -> weight for connectivity constraint loss in overall loss function
    :param NLL_WEIGHT -> weight used in nll loss for GAT
"""""
GCCM = False
FEATURE_SAMPLING = 'max' #'avg'
EDGE_DIST_THRESH = 10
WINDOW_SIZE = 8
DROPOUT = 0.5
ALPHA = 0.2
HIDDEN = 8
NUM_ATTEN_HEADS = 8
WEIGHT_DECAY_GAT = 5e-3
BETA_WEIGHT = 0.2
NLL_WEIGHT = [0.2, 0.8]
