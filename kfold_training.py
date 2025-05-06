import os
import logging
import time
import pickle
import numpy as np
import gc
import torch
from sklearn.model_selection import KFold
from utils import PatchDataDataset
from train import train_model
from config import NB_CLASSES, PATCH_SIZE, NB_CHANNELS_REDUCED,\
    DROPOUT_RATE

def train_with_kfold(split_dataset,
                    batch_size, 
                    epochs, 
                    device, 
                    outdir,
                    class_names,
                    model_classes,
                    class_weights,
                    n_folds,
                    random_state=42):
    """
    Train models using k-fold cross-validation.
    
    Args:
        split_dataset: Dataset to split for k-fold validation
        batch_size: Batch size for training
        epochs: Number of epochs to train for
        device: Device to train on
        outdir: Output directory for models and logs
        class_names: Names of classes for reporting
        model_classes: Dictionary mapping model names to model classes
        n_folds: Number of folds for cross-validation
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with results for all models across all folds
    """    
    logging.info(f"Starting {n_folds}-fold cross-validation training")
    
    # Create main output folder for k-fold results
    kfold_outdir = os.path.join(outdir, "kfold_results/")
    os.makedirs(kfold_outdir, exist_ok=True)
    
    all_models_results = {}
    
    # For each model class
    for model_name, model_class in model_classes.items():
        logging.info(f"Training model: {model_name} with {n_folds}-fold cross-validation")
        
        # Set the model name in the dataset
        split_dataset.set_model_name(model_name)
        
        # Get the original data and label files directly from the split_dataset
        all_data_files = split_dataset.data_files
        all_label_files = split_dataset.label_files
        
        # Make sure we have data files
        if not all_data_files or len(all_data_files) == 0:
            logging.error("No data files found in the dataset")
            continue
            
        logging.info(f"Total number of data files: {len(all_data_files)}")
        
        # Initialize k-fold cross-validation on the file indices
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        # Dictionary to store results for this model across all folds
        model_results = {
            'fold_histories': {},
            'fold_metrics': {},
            'average_metrics': {}
        }
        
        # Track all metrics across folds for averaging
        all_fold_val_losses = []
        all_fold_val_ious = []
        all_fold_val_dices = []
        all_fold_val_mccs = []
        all_fold_val_accs = []
        all_fold_val_precision = []
        all_fold_val_recall = []
        all_fold_val_f1 = []
        
        # Generate file indices to split
        file_indices = list(range(len(all_data_files)))
        
        # For each fold
        for fold, (train_file_indices, val_file_indices) in enumerate(kfold.split(file_indices)):
            fold_num = fold + 1
            logging.info(f"Starting fold {fold_num}/{n_folds}")
            logging.info(f"Train file indices: {len(train_file_indices)}, Val file indices: {len(val_file_indices)}")
            
            # Create fold-specific output directory
            fold_outdir = os.path.join(kfold_outdir, f"{model_name}/fold_{fold_num}/")
            os.makedirs(fold_outdir, exist_ok=True)
            
            # Get the file paths for this fold's train and validation sets
            train_data_files = [all_data_files[i] for i in train_file_indices]
            train_label_files = [all_label_files[i] for i in train_file_indices]
            val_data_files = [all_data_files[i] for i in val_file_indices]
            val_label_files = [all_label_files[i] for i in val_file_indices]
            
            logging.info(f"Train files: {len(train_data_files)}, Val files: {len(val_data_files)}")
            
            # Create new datasets using the original class           
            train_dataset = PatchDataDataset(
                data_files=train_data_files,
                label_files=train_label_files,
                patch_size=split_dataset.patch_size,
                shuffle=True,
                num_classes=split_dataset.num_classes,
                overlap_step=split_dataset.overlap_step,
                target_height=split_dataset.target_height,
                target_width=split_dataset.target_width,
                selected_channels=split_dataset.selected_channels,
                model_name=model_name, 
                transform=split_dataset.transform
            )
            
            val_dataset = PatchDataDataset(
                data_files=val_data_files,
                label_files=val_label_files,
                patch_size=split_dataset.patch_size,
                shuffle=False,
                num_classes=split_dataset.num_classes,
                overlap_step=split_dataset.overlap_step,
                target_height=split_dataset.target_height,
                target_width=split_dataset.target_width,
                selected_channels=split_dataset.selected_channels,
                model_name=model_name,  
                transform=None
            )
            
            # If using HybridModel_4, wrap datasets in HyperspectralGraphDataset
            if model_name == 'HybridModel_4':
                from models import HyperspectralGraphDataset
                train_dataset = HyperspectralGraphDataset(train_dataset)
                val_dataset = HyperspectralGraphDataset(val_dataset)

            # Create model instance
            model = model_class(
                num_classes=NB_CLASSES,
                patch_size=PATCH_SIZE,
                in_channels=NB_CHANNELS_REDUCED,
                dropout_rate=DROPOUT_RATE
            ).to(device)
            
            # Train the model on this fold using your original train_model function
            # The key difference is we pass the specific model_name
            _, history = train_model(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=batch_size,
                epochs=epochs,
                device=device,
                outdir=fold_outdir,
                model_name=model_name,  
                class_names=class_names,
                class_weights=class_weights,
                model=model
            )
            
            # Store the history for this fold
            model_results['fold_histories'][f"fold_{fold_num}"] = history
            
            # Extract final validation metrics for this fold
            final_metrics = {
                'val_loss': history['val_losses'][-1],
                'val_iou': history['val_ious'][-1],
                'val_dice': history['val_dices'][-1],
                'val_mcc': history['val_mcc'][-1],
                'val_accuracy': history['val_accuracy'][-1],
                'val_precision': history['val_precision_history'][-1],
                'val_recall': history['val_recall_history'][-1],
                'val_f1': history['val_f1_history'][-1]
            }
            
            model_results['fold_metrics'][f"fold_{fold_num}"] = final_metrics
            
            # Collect metrics for averaging
            all_fold_val_losses.append(final_metrics['val_loss'])
            all_fold_val_ious.append(final_metrics['val_iou'])
            all_fold_val_dices.append(final_metrics['val_dice'])
            all_fold_val_mccs.append(final_metrics['val_mcc'])
            all_fold_val_accs.append(final_metrics['val_accuracy'])
            all_fold_val_precision.append(final_metrics['val_precision'])
            all_fold_val_recall.append(final_metrics['val_recall'])
            all_fold_val_f1.append(final_metrics['val_f1'])
            
            # Save fold results
            with open(os.path.join(fold_outdir, f"{model_name}_fold{fold_num}_history.pkl"), "wb") as f:
                pickle.dump(history, f)
                        
            # Clean up
            del model
            torch.cuda.empty_cache()
            gc.collect()
        
        # Calculate average metrics across all folds
        model_results['average_metrics'] = {
            'avg_val_loss': np.mean(all_fold_val_losses),
            'std_val_loss': np.std(all_fold_val_losses),
            'avg_val_iou': np.mean(all_fold_val_ious),
            'std_val_iou': np.std(all_fold_val_ious),
            'avg_val_dice': np.mean(all_fold_val_dices),
            'std_val_dice': np.std(all_fold_val_dices),
            'avg_val_mcc': np.mean(all_fold_val_mccs),
            'std_val_mcc': np.std(all_fold_val_mccs),
            'avg_val_accuracy': np.mean(all_fold_val_accs),
            'std_val_accuracy': np.std(all_fold_val_accs),
        }
        
        # Calculate class-wise average metrics
        if all_fold_val_precision and len(all_fold_val_precision[0]) > 0:
            num_classes = len(all_fold_val_precision[0])
            for cls in range(num_classes):
                class_precision = [fold_precision[cls] for fold_precision in all_fold_val_precision]
                class_recall = [fold_recall[cls] for fold_recall in all_fold_val_recall]
                class_f1 = [fold_f1[cls] for fold_f1 in all_fold_val_f1]
                
                model_results['average_metrics'][f'avg_val_precision_class{cls}'] = np.mean(class_precision)
                model_results['average_metrics'][f'std_val_precision_class{cls}'] = np.std(class_precision)
                model_results['average_metrics'][f'avg_val_recall_class{cls}'] = np.mean(class_recall)
                model_results['average_metrics'][f'std_val_recall_class{cls}'] = np.std(class_recall)
                model_results['average_metrics'][f'avg_val_f1_class{cls}'] = np.mean(class_f1)
                model_results['average_metrics'][f'std_val_f1_class{cls}'] = np.std(class_f1)
        
        # Log average metrics
        logging.info(f"\nModel {model_name} - Average metrics across {n_folds} folds:")
        logging.info(f"Avg. Validation Loss: {model_results['average_metrics']['avg_val_loss']:.4f} ± {model_results['average_metrics']['std_val_loss']:.4f}")
        logging.info(f"Avg. Validation IoU: {model_results['average_metrics']['avg_val_iou']:.4f} ± {model_results['average_metrics']['std_val_iou']:.4f}")
        logging.info(f"Avg. Validation Dice: {model_results['average_metrics']['avg_val_dice']:.4f} ± {model_results['average_metrics']['std_val_dice']:.4f}")
        logging.info(f"Avg. Validation MCC: {model_results['average_metrics']['avg_val_mcc']:.4f} ± {model_results['average_metrics']['std_val_mcc']:.4f}")
        logging.info(f"Avg. Validation Accuracy: {model_results['average_metrics']['avg_val_accuracy']:.4f} ± {model_results['average_metrics']['std_val_accuracy']:.4f}")
        
        # Class-wise metrics
        for cls, name in enumerate(class_names):
            if f'avg_val_f1_class{cls}' in model_results['average_metrics']:
                logging.info(f"Class {name}:")
                logging.info(f"  Avg. Precision: {model_results['average_metrics'][f'avg_val_precision_class{cls}']:.4f} ± {model_results['average_metrics'][f'std_val_precision_class{cls}']:.4f}")
                logging.info(f"  Avg. Recall: {model_results['average_metrics'][f'avg_val_recall_class{cls}']:.4f} ± {model_results['average_metrics'][f'std_val_recall_class{cls}']:.4f}")
                logging.info(f"  Avg. F1: {model_results['average_metrics'][f'avg_val_f1_class{cls}']:.4f} ± {model_results['average_metrics'][f'std_val_f1_class{cls}']:.4f}")
        
        # Save model results
        with open(os.path.join(kfold_outdir, f"{model_name}_kfold_results.pkl"), "wb") as f:
            pickle.dump(model_results, f)
        
        # Store results for this model
        all_models_results[model_name] = model_results
    
    # Create summary plots comparing all models (based on average metrics)
    create_kfold_comparison_plots(all_models_results, class_names, kfold_outdir)
    
    # Save overall results
    with open(os.path.join(kfold_outdir, "all_models_kfold_results.pkl"), "wb") as f:
        pickle.dump(all_models_results, f)
    
    return all_models_results


def create_kfold_comparison_plots(all_models_results, class_names, outdir):
    """
    Create plots comparing metrics across different models from k-fold results,
    using the original style from create_comparison_plots.
    
    Args:
        all_models_results: Dictionary with results for all models from k-fold training
        class_names: Names of the classes
        outdir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from config import NB_CLASSES
    
    # Define a color cycle to ensure consistent colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Convert k-fold results to format similar to original training history
    # We'll use the average values across folds
    models_history = {}
    
    for model_name, model_results in all_models_results.items():
        # For each model, combine the histories from all folds
        combined_history = {}
        fold_histories = model_results['fold_histories']
        
        # Find the maximum number of epochs across all folds
        max_epochs = 0
        for fold_name, history in fold_histories.items():
            epoch_count = len(history['train_losses'])
            max_epochs = max(max_epochs, epoch_count)
        
        # Initialize arrays for each metric
        for metric in ['train_losses', 'val_losses', 'train_ious', 'val_ious', 
                     'train_dices', 'val_dices', 'train_mcc', 'val_mcc',
                     'train_accuracy', 'val_accuracy']:
            combined_history[metric] = np.zeros(max_epochs)
        
        # Initialize arrays for class-specific metrics
        for metric in ['train_precision_history', 'val_precision_history',
                     'train_recall_history', 'val_recall_history',
                     'train_f1_history', 'val_f1_history',
                     'train_dice_per_class', 'val_dice_per_class',
                     'train_iou_per_class', 'val_iou_per_class']:
            combined_history[metric] = [np.zeros(NB_CLASSES) for _ in range(max_epochs)]
        
        # Count folds for each epoch (some folds might have stopped early)
        epoch_fold_counts = np.zeros(max_epochs)
        
        # Combine metrics from all folds
        for fold_name, history in fold_histories.items():
            # For each metric, add the values from this fold
            for metric in ['train_losses', 'val_losses', 'train_ious', 'val_ious', 
                         'train_dices', 'val_dices', 'train_mcc', 'val_mcc',
                         'train_accuracy', 'val_accuracy']:
                if metric in history:
                    for e, value in enumerate(history[metric]):
                        combined_history[metric][e] += value
                        if metric == 'train_losses':  # Count once per metric
                            epoch_fold_counts[e] += 1
            
            # For class-specific metrics
            for metric in ['train_precision_history', 'val_precision_history',
                         'train_recall_history', 'val_recall_history',
                         'train_f1_history', 'val_f1_history']:
                if metric in history:
                    for e, values in enumerate(history[metric]):
                        for c, value in enumerate(values):
                            if c < len(combined_history[metric][e]):
                                combined_history[metric][e][c] += value
            
            # For per-class IoU and Dice
            for metric in ['train_dice_per_class', 'val_dice_per_class',
                         'train_iou_per_class', 'val_iou_per_class']:
                if metric in history:
                    for e, values in enumerate(history[metric]):
                        for c, value in enumerate(values):
                            # Handle tensor values
                            if hasattr(value, 'item'):
                                value = value.item()
                            if c < len(combined_history[metric][e]):
                                combined_history[metric][e][c] += value
        
        # Average the values across folds
        for e in range(max_epochs):
            if epoch_fold_counts[e] > 0:
                for metric in ['train_losses', 'val_losses', 'train_ious', 'val_ious', 
                             'train_dices', 'val_dices', 'train_mcc', 'val_mcc',
                             'train_accuracy', 'val_accuracy']:
                    combined_history[metric][e] /= epoch_fold_counts[e]
                
                for metric in ['train_precision_history', 'val_precision_history',
                             'train_recall_history', 'val_recall_history',
                             'train_f1_history', 'val_f1_history',
                             'train_dice_per_class', 'val_dice_per_class',
                             'train_iou_per_class', 'val_iou_per_class']:
                    for c in range(NB_CLASSES):
                        combined_history[metric][e][c] /= epoch_fold_counts[e]
        
        # Store combined history for this model
        models_history[model_name] = combined_history
    
    # Plot Loss, IoU, Dice comparison
    fig1, axes1 = plt.subplots(1, 5, figsize=(24, 6))
    
    # Plot Loss
    for i, (model_name, history) in enumerate(models_history.items()):
        color = colors[i % len(colors)] 
        axes1[0].plot(history['train_losses'], label=f'{model_name} Train', 
                     marker='o', linestyle='-', color=color)
        axes1[0].plot(history['val_losses'], label=f'{model_name} Val', 
                     marker='x', linestyle='--', color=color)
    
    axes1[0].set_xlabel("Epoch")
    axes1[0].set_ylabel("Loss")
    axes1[0].set_title("Loss Comparison (Avg. Across Folds)")
    axes1[0].legend()
    
    # Plot IoU
    for i, (model_name, history) in enumerate(models_history.items()):
        color = colors[i % len(colors)]  
        axes1[1].plot(history['train_ious'], label=f'{model_name} Train', 
                     marker='o', linestyle='-', color=color)
        axes1[1].plot(history['val_ious'], label=f'{model_name} Val', 
                     marker='x', linestyle='--', color=color)
    
    axes1[1].set_xlabel("Epoch")
    axes1[1].set_ylabel("IoU")
    axes1[1].set_title("IoU Comparison (Avg. Across Folds)")
    axes1[1].legend()
    
    # Plot Dice
    for i, (model_name, history) in enumerate(models_history.items()):
        color = colors[i % len(colors)]  
        train_dices = history['train_dices']
        val_dices = history['val_dices']
        axes1[2].plot(train_dices, label=f'{model_name} Train', 
                     marker='o', linestyle='-', color=color)
        axes1[2].plot(val_dices, label=f'{model_name} Val', 
                     marker='x', linestyle='--', color=color)
    
    axes1[2].set_xlabel("Epoch")
    axes1[2].set_ylabel("Dice Score")
    axes1[2].set_title("Dice Score Comparison (Avg. Across Folds)")
    axes1[2].legend()
    
    # Plot MCC
    for i, (model_name, history) in enumerate(models_history.items()):
        if 'train_mcc' in history and 'val_mcc' in history:
            color = colors[i % len(colors)]
            axes1[3].plot(history['train_mcc'], label=f'{model_name} Train', 
                         marker='o', linestyle='-', color=color)
            axes1[3].plot(history['val_mcc'], label=f'{model_name} Val', 
                         marker='x', linestyle='--', color=color)
    
    axes1[3].set_xlabel("Epoch")
    axes1[3].set_ylabel("MCC")
    axes1[3].set_title("Matthews Correlation Coefficient (Avg. Across Folds)")
    axes1[3].legend()
    
    # Plot Accuracy
    for i, (model_name, history) in enumerate(models_history.items()):
        if 'train_accuracy' in history and 'val_accuracy' in history:
            color = colors[i % len(colors)]
            axes1[4].plot(history['train_accuracy'], label=f'{model_name} Train', 
                         marker='o', linestyle='-', color=color)
            axes1[4].plot(history['val_accuracy'], label=f'{model_name} Val', 
                         marker='x', linestyle='--', color=color)
    
    axes1[4].set_xlabel("Epoch")
    axes1[4].set_ylabel("Accuracy")
    axes1[4].set_title("Accuracy Comparison (Avg. Across Folds)")
    axes1[4].legend()
    
    plt.tight_layout()
    plt.savefig(outdir + "model_comparison_metrics.png", dpi=300)
    plt.close()
    
    # Create class-specific comparison plots
    for cls in range(NB_CLASSES):
        fig, axes = plt.subplots(2, 3, figsize=(24, 10))
        
        # Plot Precision
        for i, (model_name, history) in enumerate(models_history.items()):
            color = colors[i % len(colors)] 
            train_prec = [epoch_arr[cls] for epoch_arr in history['train_precision_history']]
            val_prec = [epoch_arr[cls] for epoch_arr in history['val_precision_history']]
            axes[0, 0].plot(train_prec, label=f'{model_name} Train', 
                           marker='o', linestyle='-', color=color)
            axes[0, 0].plot(val_prec, label=f'{model_name} Val', 
                           marker='x', linestyle='--', color=color)
        
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Precision")
        axes[0, 0].set_title(class_names[cls] + " Precision (Avg. Across Folds)")
        axes[0, 0].legend()
        
        # Plot Recall
        for i, (model_name, history) in enumerate(models_history.items()):
            color = colors[i % len(colors)]
            train_rec = [epoch_arr[cls] for epoch_arr in history['train_recall_history']]
            val_rec = [epoch_arr[cls] for epoch_arr in history['val_recall_history']]
            axes[0, 1].plot(train_rec, label=f'{model_name} Train', 
                           marker='o', linestyle='-', color=color)
            axes[0, 1].plot(val_rec, label=f'{model_name} Val', 
                           marker='x', linestyle='--', color=color)
        
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Recall")
        axes[0, 1].set_title(class_names[cls] + " Recall (Avg. Across Folds)")
        axes[0, 1].legend()
        
        # Plot F1
        for i, (model_name, history) in enumerate(models_history.items()):
            color = colors[i % len(colors)] 
            train_f1 = [epoch_arr[cls] for epoch_arr in history['train_f1_history']]
            val_f1 = [epoch_arr[cls] for epoch_arr in history['val_f1_history']]
            axes[0, 2].plot(train_f1, label=f'{model_name} Train', 
                           marker='o', linestyle='-', color=color)
            axes[0, 2].plot(val_f1, label=f'{model_name} Val', 
                           marker='x', linestyle='--', color=color)
        
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("F1 Score")
        axes[0, 2].set_title(class_names[cls] + " F1 Score (Avg. Across Folds)")
        axes[0, 2].legend()
        
        # Plot class-specific Dice
        for i, (model_name, history) in enumerate(models_history.items()):
            color = colors[i % len(colors)]  
            try:
                train_dice = [epoch_arr[cls] for epoch_arr in history['train_dice_per_class']]
                val_dice = [epoch_arr[cls] for epoch_arr in history['val_dice_per_class']]
                
                epochs = range(len(train_dice))
                axes[1, 0].plot(epochs, train_dice, label=f'{model_name} Train', 
                               marker='o', linestyle='-', color=color)
                axes[1, 0].plot(epochs, val_dice, label=f'{model_name} Val', 
                               marker='x', linestyle='--', color=color)
            except (KeyError, IndexError) as e:
                print(f"Warning: Could not extract per-class Dice for {model_name}, {class_names[cls]}: {e}")

        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Dice")
        axes[1, 0].set_title(class_names[cls] + " Dice (Avg. Across Folds)")
        axes[1, 0].legend()

        # Plot class-specific IoU
        for i, (model_name, history) in enumerate(models_history.items()):
            color = colors[i % len(colors)]  
            try:
                train_iou = [epoch_arr[cls] for epoch_arr in history['train_iou_per_class']]
                val_iou = [epoch_arr[cls] for epoch_arr in history['val_iou_per_class']]
                
                epochs = range(len(train_iou))
                axes[1, 1].plot(epochs, train_iou, label=f'{model_name} Train', 
                               marker='o', linestyle='-', color=color)
                axes[1, 1].plot(epochs, val_iou, label=f'{model_name} Val', 
                               marker='x', linestyle='--', color=color)
            except (KeyError, IndexError) as e:
                print(f"Warning: Could not extract per-class IoU for {model_name}, {class_names[cls]}: {e}")

        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("IoU")
        axes[1, 1].set_title(class_names[cls] + " IoU (Avg. Across Folds)")
        axes[1, 1].legend()
        
        # Add empty plot or custom plot in the remaining panel
        axes[1, 2].set_visible(False)  # Hide the unused subplot
        
        plt.tight_layout()
        plt.savefig(outdir + "model_comparison_class_" + class_names[cls] + ".png", dpi=300)
        plt.close()
    
    #  create fold-specific plots for each model
    for model_name, model_results in all_models_results.items():
        fold_histories = model_results['fold_histories']
        
        # Create a plot showing the training curves for each fold
        plt.figure(figsize=(20, 12))
        
        # Set up 2x3 grid of plots for key metrics
        plt.subplot(2, 3, 1)
        for fold_name, history in fold_histories.items():
            plt.plot(history['val_losses'], label=f'{fold_name}')
        plt.title(f'{model_name} - Validation Loss by Fold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        for fold_name, history in fold_histories.items():
            plt.plot(history['val_ious'], label=f'{fold_name}')
        plt.title(f'{model_name} - Validation IoU by Fold')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        for fold_name, history in fold_histories.items():
            plt.plot(history['val_dices'], label=f'{fold_name}')
        plt.title(f'{model_name} - Validation Dice by Fold')
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 4)
        for fold_name, history in fold_histories.items():
            plt.plot(history['val_mcc'], label=f'{fold_name}')
        plt.title(f'{model_name} - Validation MCC by Fold')
        plt.xlabel('Epoch')
        plt.ylabel('MCC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 5)
        for fold_name, history in fold_histories.items():
            plt.plot(history['val_accuracy'], label=f'{fold_name}')
        plt.title(f'{model_name} - Validation Accuracy by Fold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(outdir + f"{model_name}_folds_comparison.png", dpi=300)
        plt.close()
