import numpy as np
import os
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import ndimage
from utils import compute_multi_image_metrics, visualize_multi_image_metrics, print_metrics_summary, \
    format_metrics_as_table, save_model_metrics, normalize_spectral_signature, calculate_class_weights
from config import NB_CLASSES, PATCH_SIZE, OVERLAP_STEP, CLASS_NAMES, OUTPUT_DIR, \
    IMG_WIDTH, IMG_HEIGHT, CHANNELS, ORIG_HEIGHT, ORIG_WIDTH, TEST_PATH

# Initialize TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_trt_engine(engine_path):
    """
    Load a TensorRT engine from file
    """
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    """
    Allocate device buffers for input and output with compatibility for different TensorRT versions
    """
    inputs = []
    outputs = []
    bindings = []
    
    for binding_idx, binding in enumerate(engine):
        # Get shape and dtype
        try:
            shape = engine.get_tensor_shape(binding)
        except AttributeError:
            shape = engine.get_binding_shape(binding)
        
        # Get data type
        try:
            dtype = trt.nptype(engine.get_tensor_dtype(binding))
        except AttributeError:
            dtype = trt.nptype(engine.get_binding_dtype(binding))
        
        # Calculate size - use shape directly, no max_batch_size
        size = trt.volume(shape)
        
        # Handle dynamic batch size (first dimension is -1)
        if shape[0] == -1:
            size = size * 1  # Use batch size of 1 for inference
        
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        # Append the device buffer to bindings
        bindings.append(int(device_mem))
        
        # Check if binding is input or output
        try:
            is_input = engine.binding_is_input(binding)
        except AttributeError:
            is_input = engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT
        
        if is_input:
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    
    return inputs, outputs, bindings

def inference_with_trt(engine, context, buffers, input_data):
    """
    Run inference with TensorRT
    
    Args:
        engine: TensorRT engine
        context: TensorRT execution context
        buffers: Input/output buffers
        input_data: Input data as numpy array
    
    Returns:
        Output from the model
    """
    # Unpack buffers
    inputs, outputs, bindings = buffers
    
    # Copy input data to the buffer
    np.copyto(inputs[0]['host'], input_data.ravel())
    
    # Transfer input data to the GPU
    cuda.memcpy_htod(inputs[0]['device'], inputs[0]['host'])
    
    # Run inference
    context.execute_v2(bindings=bindings)
    
    # Transfer predictions back
    cuda.memcpy_dtoh(outputs[0]['host'], outputs[0]['device'])
    
    # Get output shape
    try:
        # Newer TensorRT API
        output_binding = list(engine)[1]  # Assuming output is the second binding
        output_shape = engine.get_tensor_shape(output_binding)
    except AttributeError:
        # Older TensorRT API
        output_shape = engine.get_binding_shape(1)
    
    # Handle dynamic batch dimension
    if output_shape[0] == -1:
        output_shape = (1,) + tuple(output_shape[1:])
    
    # Return reshaped output
    return outputs[0]['host'].reshape(output_shape)

def create_weight_mask(shape, type='gaussian'):
    """
    Create a weight mask for patch blending.
    
    Args:
        shape: Tuple of (height, width)
        type: Type of weight mask ('gaussian' or 'cosine')
    
    Returns:
        2D weight mask
    """
    h, w = shape
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    x_grid, y_grid = np.meshgrid(x, y)
    
    if type == 'gaussian':
        # Gaussian weighting - highest in center, falls off toward edges
        sigma = 0.5
        weight_mask = np.exp(-(x_grid**2 + y_grid**2) / (2*sigma**2))
        weight_mask = weight_mask / np.max(weight_mask)
    elif type == 'cosine':
        # Cosine weighting - smooth transition from center to edges
        d = np.sqrt(x_grid**2 + y_grid**2)
        d = np.clip(d, 0, 1)
        weight_mask = np.cos(d * np.pi/2)
    else:
        # Uniform weighting
        weight_mask = np.ones(shape)
    
    return weight_mask

def inference_with_weighted_blending_trt(engine_path, 
                                         model_name,
                                         image_path, 
                                         target_width, 
                                         target_height, 
                                         patch_size, 
                                         overlap_step):
    """
    Run inference with weighted blending of overlapping patches using TensorRT,
    with pixel-wise spectral normalization.
    
    Args:
        engine_path: Path to the TensorRT engine file
        model_name: Name of the model (needed for special handling)
        image_path: Path to the input image
        target_width: Width to resize input image to
        target_height: Height to resize input image to
        patch_size: Size of patches for processing
        overlap_step: Step size for overlapping patches
    """
    # Load TensorRT engine
    engine = load_trt_engine(engine_path)
    context = engine.create_execution_context()
    
    # Allocate buffers
    buffers = allocate_buffers(engine)
    
    # Load and preprocess the image
    img_arr = np.load(image_path, mmap_mode='r')
    img_arr_resized = cv2.resize(img_arr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    # Select specific channels
    selected_channels = CHANNELS
    img_arr_resized = img_arr_resized[:, :, selected_channels]
    
    # Create prediction maps for each class
    H, W, C = img_arr_resized.shape
    num_classes = NB_CLASSES
    prediction_probs = np.zeros((H, W, num_classes), dtype=np.float32)
    weights = np.zeros((H, W), dtype=np.float32)
    
    # For HybridModel_4, prepare the utilities needed for hetero graph creation
    # But we'll still use TensorRT for inference
    if model_name == "HybridModel_4":
        from utils import create_hetero_patch_graph
    
    # Process image in patches
    for start_x in range(0, H - patch_size + 1, overlap_step):
        for start_y in range(0, W - patch_size + 1, overlap_step):
            # Extract patch
            patch = img_arr_resized[start_x:start_x+patch_size, start_y:start_y+patch_size, :]
            
            # Convert to tensor for normalization
            patch_tensor = torch.from_numpy(patch).float().permute(2, 0, 1).unsqueeze(0)
            
            # Apply pixel-wise spectral normalization
            patch_tensor = normalize_spectral_signature(patch_tensor)
            
            # Special handling for HybridModel_4 - create hetero graph 
            # Note: The TensorRT engine for HybridModel_4 should have been built to accept both inputs
            if model_name == "HybridModel_4":
                hetero_graph = create_hetero_patch_graph(patch_tensor.squeeze(0))
                
                # Use hetero graph features as additional input to TensorRT
                # This assumes your TensorRT engine was built to handle these inputs
                # You'll need to extract the features from the graph in a format TensorRT can use
                # This is a simplified example; you'll need to adapt this to your specific graph format
                graph_features = extract_graph_features(hetero_graph)
                
                # Combine with image features as expected by your TensorRT model
                # The exact implementation depends on how you exported the model to TensorRT
                combined_input = prepare_combined_input(patch_tensor, graph_features)
                
                # Run inference with TensorRT
                trt_output = inference_with_trt(engine, context, buffers, combined_input)
            else:
                # Standard TensorRT inference for other models
                # Convert back to numpy for TensorRT
                patch_normalized = patch_tensor.squeeze(0).permute(1, 2, 0).numpy()
                
                # Reshape to expected input format for TensorRT (NCHW)
                patch_input = patch_normalized.transpose(2, 0, 1).reshape(1, C, patch_size, patch_size).astype(np.float32)
                
                # Run inference with TensorRT
                trt_output = inference_with_trt(engine, context, buffers, patch_input)
            
            # Convert logits to probabilities
            trt_output = trt_output - np.max(trt_output, axis=1, keepdims=True)
            probs = np.exp(trt_output)
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            probs = probs[0]  # Remove batch dimension

            # Create weight mask - gaussian weighting gives highest weight to center
            weight_mask = create_weight_mask((patch_size, patch_size), type='gaussian')
            
            # Apply weighted prediction
            for c in range(num_classes):
                prediction_probs[start_x:start_x+patch_size, start_y:start_y+patch_size, c] += probs[c] * weight_mask
            
            weights[start_x:start_x+patch_size, start_y:start_y+patch_size] += weight_mask
    
    # Normalize by weights
    for c in range(num_classes):
        prediction_probs[:, :, c] = np.divide(
            prediction_probs[:, :, c], 
            weights,
            out=np.zeros_like(prediction_probs[:, :, c]),
            where=weights>0
        )
    
    # Get the class with the highest probability
    prediction_map = np.argmax(prediction_probs, axis=2).astype(np.int64)
    
    # Clean up TensorRT resources
    del context
    del engine
    
    return prediction_map

# Helper functions for HybridModel_4
def extract_graph_features(hetero_graph):
    """
    Extract features from hetero graph that can be passed to TensorRT.
    You'll need to adapt this based on your graph structure.
    """
    # This is a placeholder - you need to implement this based on your graph structure
    # The goal is to extract features in a format TensorRT can process
    features = np.zeros((1, 64, 32, 32), dtype=np.float32)  # Example shape
    
    # Extract relevant features from your graph
    # Example: if your graph has node features you want to use
    if hasattr(hetero_graph, 'x'):
        node_features = hetero_graph.x.cpu().numpy()
        # Process features to fit your TensorRT model's expected format
    
    return features

def prepare_combined_input(image_tensor, graph_features):
    """
    Combine image features and graph features for TensorRT input.
    This depends on how you've built your TensorRT engine.
    """
    # Convert image tensor to numpy
    image_features = image_tensor.squeeze(0).permute(1, 2, 0).numpy()
    image_features = image_features.transpose(2, 0, 1).reshape(1, -1, image_features.shape[0], image_features.shape[1])
    
    # The structure here depends on how your TensorRT model was built to accept inputs
    # Option 1: If your TensorRT model takes concatenated features
    combined = np.concatenate([image_features, graph_features], axis=1)
    
    # Option 2: If your TensorRT model takes separate inputs, return a list/dict
    # combined = [image_features, graph_features]
    
    return combined

def post_process_prediction(prediction_map, min_size=100):
    """
    Post-process the prediction map to remove small regions and smooth boundaries.
    
    Args:
        prediction_map: The raw prediction map
        min_size: Minimum size of regions to keep
    
    Returns:
        Processed prediction map
    """
    processed_map = np.copy(prediction_map)
    
    # Process each class separately
    num_classes = NB_CLASSES
    for class_idx in range(num_classes):
        # Create binary mask for this class
        class_mask = (prediction_map == class_idx)
        
        # Label connected components
        labeled_mask, num_features = ndimage.label(class_mask)
        
        # Remove small components
        for label in range(1, num_features + 1):
            component_size = np.sum(labeled_mask == label)
            if component_size < min_size:
                # Find the most common neighboring class
                dilated = ndimage.binary_dilation(labeled_mask == label)
                boundary = dilated & ~(labeled_mask == label)
                if np.any(boundary):
                    neighbor_classes = prediction_map[boundary]
                    if len(neighbor_classes) > 0:
                        most_common = np.bincount(neighbor_classes).argmax()
                        processed_map[labeled_mask == label] = most_common
    
    # Apply median filter for smoother boundaries
    processed_map = ndimage.median_filter(processed_map, size=3)
    
    return processed_map

def visualize_segmentation(prediction_map, output_dir, filename, class_names):
    """
    Convert a single-channel prediction map to a color-coded visualization.
    
    Args:
        prediction_map: NumPy array of shape (H, W) containing class indices
        output_dir: Path to save the visualization image
        class_colors: Optional list of RGB colors for each class
        class_names: Optional list of class names for the legend
    """
    # Default color map if none provided
    cmap = mcolors.ListedColormap(["blue", "green", "yellow"])  # 0=Sea (Blue), 1=Land (Green), 2=Cloud (Yellow)
    bounds = [0, 1, 2, 3]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Specific colors from the uploaded image
    class_colors = [
        [255, 0, 0],      # Blue for Sea (BGR format)
        [0, 165, 255],    # Orange for Land (BGR format)
        [128, 128, 128]   # Gray for Clouds (BGR format)
    ]

    # Get the actual dimensions of the prediction map
    H, W = prediction_map.shape

    prediction_map = prediction_map.astype(np.uint8)
    # Create a 3-channel RGB image
    rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Map class indices to colors
    for class_id, color in enumerate(class_colors):
        rgb_image[prediction_map == class_id] = color    

    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        filename += '.png'  # Add PNG extension if no valid extension exists
    
    # Create complete output path
    output_path = os.path.join(output_dir, filename)
    
    try:
        success = cv2.imwrite(output_path, rgb_image)
        if not success:
            raise Exception("Failed to write image")
        print(f"Saved visualization with dimensions {H}x{W} to {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Saved visualization with matplotlib to {output_path}")
        except Exception as e2:
            print(f"Error saving image with matplotlib: {e2}")

def batch_inference_trt(engine_paths, 
                       image_paths, 
                       output_dir, 
                       target_width, 
                       target_height, 
                       patch_size, 
                       overlap_step, 
                       original_shape, 
                       class_names):
    """
    Run TensorRT inference on multiple images with multiple models, apply post-processing, and create visualizations.
    
    Args:
        engine_paths (dict): Dictionary of TensorRT engine file paths with model names as keys
        image_paths (list): List of paths to input images
        output_dir (str): Directory to save outputs
        patch_size (int): Size of patches for processing
        overlap_step (int): Step size for overlapping patches
        original_shape (tuple): The original image shape (width, height)
        class_names (list): Names of classes for visualization
    
    Returns:
        dict: Dictionary of prediction maps for each model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store results for each model
    all_results = {}
    
    # Iterate through each model
    for model_name, engine_path in engine_paths.items():
        print(f"\n===== Processing with model: {model_name} =====")
        
        # Store predictions for this model
        model_predictions = []
        
        # Create a subdirectory for model outputs
        model_output_dir = os.path.join(output_dir, f"{model_name}_trt")
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Process each image
        for idx, image_path in enumerate(image_paths):
            print(f"Processing {model_name} - image {idx+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            # Run inference with weighted blending
            start_time = cv2.getTickCount()
            raw_prediction = inference_with_weighted_blending_trt(
                engine_path, 
                model_name,
                image_path, 
                target_width, 
                target_height,
                patch_size, 
                overlap_step
            )
            end_time = cv2.getTickCount()
            inference_time = (end_time - start_time) / cv2.getTickFrequency()
            print(f"Inference time: {inference_time:.2f} seconds")
            
            # Apply post-processing
            processed_prediction = post_process_prediction(raw_prediction, min_size=100)
             
            # Resize prediction back to original dimensions
            # Using nearest neighbor interpolation to preserve class labels
            original_sized_prediction = cv2.resize(
                processed_prediction, 
                original_shape, 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Save the processed prediction
            base_filename = os.path.basename(image_path).split('.')[0]
            
            # Create standard visualization
            vis_filename = f"{base_filename}_segmentation.png"
            visualize_segmentation(
                original_sized_prediction, 
                model_output_dir, 
                vis_filename,
                class_names=class_names
            )
            
            print(f"Saved prediction and visualizations for {base_filename}")
            model_predictions.append(original_sized_prediction)
        
        # Store results for this model
        all_results[model_name] = model_predictions[0] if len(model_predictions) == 1 else model_predictions
    
    return all_results

def main():
    # Dictionary of TensorRT engine paths for each model
    engine_paths = {
        name.replace(".trt", ""): os.path.join(OUTPUT_DIR, name)
        for name in os.listdir(OUTPUT_DIR) if name.endswith(".trt")
    }

    
    # Test image paths
    test_data_files = [
        TEST_PATH + "273-20230217_CaptureDL_lakevolta_2023-02-15_0953Z/DATA/273-20230217_CaptureDL_lakevolta_2023-02-15_0953Z-radiance.npy",
        TEST_PATH + "25-20221027_CaptureDL_bangladesh_2022_10_26T04_02_54/DATA/25-20221027_CaptureDL_bangladesh_2022_10_26T04_02_54-radiance.npy",
        TEST_PATH + "26-Trondheim_2022_08_23T10_26_43/DATA/26-Trondheim_2022_08_23T10_26_43-radiance.npy"
    ]
    
    # Test label files for metrics calculation
    test_label_files = [
        TEST_PATH + "273-20230217_CaptureDL_lakevolta_2023-02-15_0953Z/GROUND-TRUTH LABELS/lakevolta_2023-02-15_0953Z_class_NPY_FORMAT.npy",
        TEST_PATH + "25-20221027_CaptureDL_bangladesh_2022_10_26T04_02_54/GROUND-TRUTH LABELS/bangladesh_2022-10-26_class_NPY_FORMAT.npy",
        TEST_PATH + "26-Trondheim_2022_08_23T10_26_43/GROUND-TRUTH LABELS/Trondheim_2022_08_23T10_26_43_class_NPY_FORMAT.npy"
    ]

    original_shape = (ORIG_WIDTH, ORIG_HEIGHT)
    
    # Calculate class weights using your function
    class_weights = calculate_class_weights(test_label_files, NB_CLASSES)
    # Convert to tensor for metrics calculation
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    else:
        class_weights_tensor = None

    # Run batch inference with TensorRT
    try:
        start_time = cv2.getTickCount()
        
        multi_model_predictions = batch_inference_trt(
            engine_paths, 
            test_data_files, 
            OUTPUT_DIR, 
            IMG_WIDTH, 
            IMG_HEIGHT,
            PATCH_SIZE, 
            OVERLAP_STEP, 
            original_shape, 
            CLASS_NAMES
        )
        
        end_time = cv2.getTickCount()
        total_time = (end_time - start_time) / cv2.getTickFrequency()
        
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per model per image: {total_time/(len(test_data_files)*len(engine_paths)):.2f} seconds")
        
        # Compute metrics for each model's predictions        
        multi_model_metrics = {}
        for model_name, predictions in multi_model_predictions.items():
            print(f"\n===== Metrics for {model_name} =====")
            model_metrics = compute_multi_image_metrics(
                predictions, 
                test_label_files, 
                num_classes=NB_CLASSES, 
                class_weights=class_weights_tensor
            )
            multi_model_metrics[model_name] = model_metrics
            
            # Visualize metrics for this model
            model_output_dir = os.path.join(OUTPUT_DIR, f"{model_name}_trt")
            visualize_multi_image_metrics(model_metrics, model_output_dir, CLASS_NAMES)
            
            # Print detailed metrics
            print(f"\n--- Metrics Table for {model_name} ---")
            table_output = format_metrics_as_table(model_metrics, CLASS_NAMES)
            print(table_output)

            print("\n--- Detailed Metrics Summary ---")
            print_metrics_summary(model_metrics, CLASS_NAMES)

            # Save metrics to files
            save_model_metrics(model_name, model_metrics, model_output_dir, CLASS_NAMES)
        
    except Exception as e:
        print(f"An error occurred during TensorRT inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()