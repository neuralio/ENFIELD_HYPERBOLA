import os
import torch
import logging
from augmentations import transformations
from utils import SplitPatchDataDataset, calculate_class_weights
from train import train_multiple_models
from kfold_training import train_with_kfold
from rich import print
from config import BASE_PATH, BATCH_SIZE, NB_CLASSES, \
    EPOCHS, PATCH_SIZE, VAL_SPLIT, DEBUG_MODE, \
    APPLY_AUGMENTATIONS, OVERLAP_STEP, OUTPUT_DIR, \
    IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES, CHANNELS, \
    KFOLD, NFOLDS, MODEL_CLASSES

# Set log level dynamically (e.g., using an environment variable)
# Run with LOG_LEVEL=DEBUG python train.py 
# log_level = os.getenv("LOG_LEVEL", "INFO").upper()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler("debug.log") 
    ]
)

if DEBUG_MODE:
    logging.getLogger().setLevel(logging.DEBUG)
else:
    logging.getLogger().setLevel(logging.INFO)


# Define the models you want to train
model_classes = MODEL_CLASSES

if APPLY_AUGMENTATIONS:
    augmentations = transformations
else:
    augmentations = None

# Create the split dataset object
split_dataset = SplitPatchDataDataset(
    base_path=BASE_PATH,
    patch_size=PATCH_SIZE,
    validation_split=VAL_SPLIT,
    num_classes=NB_CLASSES,
    overlap_step=OVERLAP_STEP,
    target_height=IMG_HEIGHT,
    target_width=IMG_WIDTH,
    selected_channels=CHANNELS,
    shuffle=True, 
    transform=augmentations) # applies transforms only in train dataset

# Get all label files from the split dataset
all_label_files = split_dataset.label_files

# Calculate class weights
class_weights = calculate_class_weights(all_label_files, NB_CLASSES)

logging.info('Starting training...\n')
# Train all models and get their histories
if KFOLD:
    logging.info(f"Using {NFOLDS}-fold cross-validation")
    # Train with k-fold cross-validation
    results = train_with_kfold(
        split_dataset=split_dataset,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        device=device,
        outdir=OUTPUT_DIR,
        class_names=CLASS_NAMES,
        model_classes=model_classes,
        class_weights=class_weights,
        n_folds=NFOLDS
    )
else:
    logging.info("Using standard train/val split")
    results = train_multiple_models(
        split_dataset=split_dataset,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        device=device,
        outdir=OUTPUT_DIR,
        class_names=CLASS_NAMES,
        class_weights=class_weights,
        model_classes=model_classes
    )
