import os
import tensorrt as trt
import numpy as np
from calibrator import MyEntropyCalibrator
from utils import SplitPatchDataDataset, normalize_spectral_signature
from config import PATCH_SIZE, OUTPUT_DIR, BASE_PATH, VAL_SPLIT,\
     IMG_HEIGHT, IMG_WIDTH, CHANNELS, OVERLAP_STEP, NB_CLASSES

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

split_dataset = SplitPatchDataDataset(
    base_path=BASE_PATH,
    patch_size=PATCH_SIZE,
    validation_split=VAL_SPLIT,
    num_classes=NB_CLASSES,
    overlap_step=OVERLAP_STEP,
    target_height=IMG_HEIGHT,
    target_width=IMG_WIDTH,
    selected_channels=CHANNELS,
    shuffle=False,
    transform=None
)

# Just get the val dataset (is the same across models, so model name doesn't matter)
split_dataset.set_model_name("Unet")
val_dataset = split_dataset.get_val_dataset()

calibration_data = []
for i in range(1000):  # max ~1000 patches is enough
    patch_tensor, _, _ = val_dataset[i]
    patch_tensor = normalize_spectral_signature(patch_tensor).numpy().astype(np.float32)
    calibration_data.append(np.expand_dims(patch_tensor, axis=0))

model_names = ["Unet"] #"HybridModel_4"
for model_name in model_names:
    split_dataset.set_model_name(model_name)
    def build_engine(onnx_path, engine_path, calibration_data):
        with trt.Builder(TRT_LOGGER) as builder:
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            with builder.create_network(network_flags) as network:
                with trt.OnnxParser(network, TRT_LOGGER) as parser:
                    config = builder.create_builder_config()
                    # Allow tensorRT to use up to 12GB of GPU memory
                    # Use Case                   | Recommended Size
                    # Default for small models   | 1 << 20              (1 MB)
                    # Edge device                | 1 << 28              (256 MB)
                    # Moderate-size CNN/Unet     | 1 << 29 or 1 << 30   (512 MB to 1 GB)
                    # Large ViT or complex GNN   | 1 << 31 to 1 << 32   (2–4 GB)
                    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 * (1 << 30)) # 12 GB

                    # Enable INT8 and FP16
                    config.set_flag(trt.BuilderFlag.INT8)
                    config.set_flag(trt.BuilderFlag.FP16)
                    config.int8_calibrator = MyEntropyCalibrator(calibration_data)

                    # This will take longer but produce better results
                    config.builder_optimization_level = 5

                    # Parse ONNX model
                    with open(onnx_path, 'rb') as model_file:
                        if not parser.parse(model_file.read()):
                            print('Failed to parse ONNX model:')
                            for error in range(parser.num_errors):
                                print(parser.get_error(error))
                            return None

                    # Build serialized engine
                    serialized_engine = builder.build_serialized_network(network, config)
                    if serialized_engine is None:
                        print("Failed to build serialized engine.")
                        return None

                    # Save engine
                    with open(engine_path, 'wb') as f:
                        f.write(serialized_engine)
                    print("TensorRT engine built and serialized successfully!")

                    # Check engine size
                    print(f"Engine size: {os.path.getsize(engine_path) / (1024 * 1024):.2f} MB")
                    

    # Build TensorRT engine from ONNX model 
    build_engine(OUTPUT_DIR + f"{model_name}.onnx", 
                 OUTPUT_DIR + f"{model_name}.trt", 
                 calibration_data)
