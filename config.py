from models import HybridModel_1, HybridModel_2, HybridModel_3, HybridModel_4

BASE_PATH = '/home/ggous/Documents/Hyperspectral_internet/1-DATA WITH GROUND-TRUTH LABELS/'
TEST_PATH = '/home/ggous/Documents/TEST_DATA/'
OUTPUT_DIR = './saved_data/'

BATCH_SIZE = 32
WEIGHT_DECAY = 1e-3
DROPOUT_RATE = 0.3
LEARNING_RATE = 1e-4
SCHEDULER_PATIENCE = 4
SCHEDULER_FACTOR = 0.5
EARLY_STOPPING = 10

ORIG_HEIGHT = 956
ORIG_WIDTH = 684

NB_CHANNELS_ORIG = 120
NB_CLASSES = 3 # sea, land, cloud
PATCH_SIZE = 48

CHANNELS = [7, 29, 64, 100, 103]

if len(CHANNELS) == 3:
    NB_CHANNELS_REDUCED = 3
else:
    NB_CHANNELS_REDUCED = 5

# 704 with 32 patch size and 960 with 48 patch size
if PATCH_SIZE == 32:
    IMG_WIDTH = 704
    IMG_HEIGHT = 704
else:
    IMG_WIDTH = 960
    IMG_HEIGHT = 960


OVERLAP_STEP = PATCH_SIZE // 2

PREFETCH_FACTOR = 8

EPOCHS = 100
VAL_SPLIT = 0.2

DEBUG_MODE = False
PROFILE = 0 # Set to 0 if you don’t want profiling
USE_AMP = True # use mixed precision
NUM_WORKERS = 4

KFOLD = False
NFOLDS = 5

APPLY_AUGMENTATIONS = True
APPLY_MIXUP = False

CLASS_NAMES = ["Sea", "Land", "Cloud"]

# Used in models to train and models to inference
MODEL_CLASSES = {
        "HybridModel_1": HybridModel_1,
        "HybridModel_2": HybridModel_2,
        "Unet": HybridModel_3,
        "HybridModel_4": HybridModel_4,
    }
