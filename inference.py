import numpy as np
import os
import cv2
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import  create_hetero_patch_graph, visualize_multi_image_metrics, compute_multi_image_metrics, \
    print_metrics_summary, format_metrics_as_table, save_model_metrics, normalize_spectral_signature, calculate_class_weights
from config import NB_CLASSES, NB_CHANNELS_REDUCED, PATCH_SIZE, DROPOUT_RATE, \
    OVERLAP_STEP, CLASS_NAMES, OUTPUT_DIR, IMG_WIDTH, IMG_HEIGHT, \
    ORIG_HEIGHT, ORIG_WIDTH, CHANNELS, TEST_PATH, MODEL_CLASSES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference_with_weighted_blending(model, 
                                     model_name,
                                     image_path, 
                                     target_width, 
                                     target_height, 
                                     patch_size, 
                                     overlap_step, 
                                     device):
    """
    Run inference with weighted blending of overlapping patches,
    using pixel-wise spectral normalization.
    """
    # Set model to evaluation mode
    model.eval()
    
    # Load and preprocess the image
    img_arr = np.load(image_path, mmap_mode='r')
    img_arr_resized = cv2.resize(img_arr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    # Select specific channels
    selected_channels = CHANNELS
    img_arr_resized = img_arr_resized[:, :, selected_channels]
        
    # Create prediction maps for each class
    H, W, C = img_arr_resized.shape
    num_classes = NB_CLASSES if 'NB_CLASSES' in globals() else 3
    prediction_probs = np.zeros((H, W, num_classes), dtype=np.float32)
    weights = np.zeros((H, W), dtype=np.float32)
    
    # Process image in patches
    with torch.no_grad():
        for start_x in range(0, H - patch_size + 1, overlap_step):
            for start_y in range(0, W - patch_size + 1, overlap_step):
                # Extract patch
                patch = img_arr_resized[start_x:start_x+patch_size, start_y:start_y+patch_size, :]
                
                # Convert to tensor
                patch_tensor = torch.from_numpy(patch).float().permute(2, 0, 1).unsqueeze(0)
                
                # Apply pixel-wise spectral normalization
                patch_tensor = normalize_spectral_signature(patch_tensor)
                
                # Move to device
                patch_tensor = patch_tensor.to(device)
                
                if model_name == 'HybridModel_4':
                    # Construct hetero graph for this patch
                    hetero_graph = create_hetero_patch_graph(patch_tensor.squeeze(0))
                    hetero_graph = hetero_graph.to(device)
                    logits = model(patch_tensor, hetero_graph)
                else:
                    logits = model(patch_tensor)
                
                # Get prediction
                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                
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
    
    return prediction_map

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

def post_process_prediction(prediction_map, min_size=100):
    """
    Post-process the prediction map to remove small regions and smooth boundaries.
    
    Args:
        prediction_map: The raw prediction map
        min_size: Minimum size of regions to keep
    
    Returns:
        Processed prediction map
    """
    from scipy import ndimage
    
    processed_map = np.copy(prediction_map)
    
    # Process each class separately
    num_classes = NB_CLASSES if 'NB_CLASSES' in globals() else 3
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
    
def batch_inference(models, 
                    image_paths, 
                    output_dir, 
                    target_width, 
                    target_height, 
                    patch_size, 
                    overlap_step, 
                    original_shape, 
                    class_names):
    """
    Run inference on multiple images using multiple models, apply post-processing, and create visualizations.
    
    Args:
        models (dict): Dictionary of models with their names as keys
        image_paths (list): List of paths to input images
        output_dir (str): Directory to save outputs
        patch_size (int): Size of patches for processing
        overlap_step (int): Step size for overlapping patches
        original_shape (tuple): The original image shape
        class_names (list): Names of classes for visualization
    
    Returns:
        dict: Dictionary of prediction maps for each model
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Dictionary to store results for each model
    all_results = {}
    
    # Iterate through models
    for model_name, model in models.items():
        # Ensure model is in evaluation mode and on correct device
        model.eval()
        model.to(device)
        
        # Create a subdirectory for this model's outputs
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Store predictions for this model
        model_predictions = []
        
        # Process each image
        for idx, image_path in enumerate(image_paths):
            print(f"Processing {model_name} - image {idx+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            # Run inference with weighted blending
            raw_prediction = inference_with_weighted_blending(
                model, 
                model_name,
                image_path, 
                target_width, 
                target_height,
                patch_size, 
                overlap_step, 
                device
            )
            
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


def prepare_models_for_inference(model_classes, base_output_dir=OUTPUT_DIR):
    """
    Dynamically prepare models for inference based on a dictionary of model classes.
    
    Args:
        model_classes (dict): Dictionary of model classes with their names as keys
        base_output_dir (str): Base directory where model checkpoints are saved
    
    Returns:
        dict: Dictionary of loaded models with their names as keys
    """
    # Dictionary to store models
    models = {}
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dynamically load models
    for model_name, ModelClass in model_classes.items():
        # Construct checkpoint path
        checkpoint_path = os.path.join(base_output_dir, f"{model_name}_best.pth")
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint for {model_name} not found at {checkpoint_path}")
            continue
        
        # Create model instance using the specific model class
        try:
            # Initialize model with appropriate parameters
            # Adjust these parameters based on how your specific model classes are defined
            model = ModelClass(
                num_classes=NB_CLASSES, 
                patch_size=PATCH_SIZE, 
                in_channels=NB_CHANNELS_REDUCED,
                dropout_rate=DROPOUT_RATE
            )
            
            # Load model weights
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()  # Set to evaluation mode
            model.to(device)
            
            # Store in models dictionary
            models[model_name] = model
            print(f"Loaded model: {model_name} from {checkpoint_path}")
        
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
    
    if not models:
        raise ValueError("No models could be loaded. Check your model names and checkpoint paths.")
    
    return models

def main():
    original_shape = (ORIG_WIDTH, ORIG_HEIGHT)


    test_data_files = [TEST_PATH + "273-20230217_CaptureDL_lakevolta_2023-02-15_0953Z/DATA/273-20230217_CaptureDL_lakevolta_2023-02-15_0953Z-radiance.npy",
                       TEST_PATH + "25-20221027_CaptureDL_bangladesh_2022_10_26T04_02_54/DATA/25-20221027_CaptureDL_bangladesh_2022_10_26T04_02_54-radiance.npy",
                       TEST_PATH + "26-Trondheim_2022_08_23T10_26_43/DATA/26-Trondheim_2022_08_23T10_26_43-radiance.npy"]

    test_label_files = [TEST_PATH + "273-20230217_CaptureDL_lakevolta_2023-02-15_0953Z/GROUND-TRUTH LABELS/lakevolta_2023-02-15_0953Z_class_NPY_FORMAT.npy",
                        TEST_PATH + "25-20221027_CaptureDL_bangladesh_2022_10_26T04_02_54/GROUND-TRUTH LABELS/bangladesh_2022-10-26_class_NPY_FORMAT.npy",
                        TEST_PATH + "26-Trondheim_2022_08_23T10_26_43/GROUND-TRUTH LABELS/Trondheim_2022_08_23T10_26_43_class_NPY_FORMAT.npy"]

    # Calculate class weights using your function
    class_weights = calculate_class_weights(test_label_files, NB_CLASSES)
    # Convert to tensor for metrics calculation
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    else:
        class_weights_tensor = None

    # List of model names to load (corresponding to saved checkpoint files)
    model_classes = MODEL_CLASSES
    # Prepare models
    try:
        models_for_inference = prepare_models_for_inference(model_classes, OUTPUT_DIR)

        # Perform multi-model inference
        multi_model_predictions = batch_inference(
            models_for_inference, 
            test_data_files, 
            OUTPUT_DIR, 
            IMG_WIDTH, 
            IMG_HEIGHT,
            PATCH_SIZE, 
            OVERLAP_STEP, 
            original_shape, 
            CLASS_NAMES
        )

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
            model_output_dir = os.path.join(OUTPUT_DIR, model_name)
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
        print(f"An error occurred during multi-model inference: {e}")

if __name__ == "__main__":
    main()