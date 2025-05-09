import os
import logging
import time
import pickle
import numpy as np
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import prune
import torch.autograd.profiler as profiler
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torchmetrics.classification import MulticlassMatthewsCorrCoef, MulticlassAccuracy
from sklearn.metrics import precision_recall_fscore_support
from utils import collate_fn, collate_fn_mixup, visualize_feature_maps, \
    collate_fn_graph, create_individual_plots, create_comparison_plots
from metrics import  dice_iou_score_multiclass_weighted, FocalLoss
from models import HybridModel_4, HyperspectralGraphDataset
from rich import print
from config import  NB_CLASSES, PATCH_SIZE, NB_CHANNELS_REDUCED,\
    PROFILE, USE_AMP, NUM_WORKERS, PREFETCH_FACTOR, WEIGHT_DECAY, \
    DROPOUT_RATE, LEARNING_RATE, SCHEDULER_PATIENCE, SCHEDULER_FACTOR, \
    APPLY_MIXUP, EARLY_STOPPING, OUTPUT_DIR

from matplotlib_config import configure_matplotlib
# Apply global settings
configure_matplotlib()


def train_multiple_models(split_dataset,
                          batch_size, 
                          epochs, 
                          device, 
                          outdir,
                          class_names,
                          class_weights,
                          model_classes):
    """
    Train multiple models and compare their performance.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        epochs: Number of epochs
        device: Device to train on
        model_classes: Dictionary mapping model names to model classes
    
    Returns:
        Dictionary containing training histories for all models
    """
    all_models_history = {}
    
    for model_name, model_class in model_classes.items():
        logging.info(f"Training model: {model_name}")
      
        # Set the model name
        split_dataset.set_model_name(model_name)

        # Access train and validation datasets
        train_patch_dataset = split_dataset.get_train_dataset()
        val_patch_dataset = split_dataset.get_val_dataset()   

        train_dataset_pyg = HyperspectralGraphDataset(patch_dataset=train_patch_dataset)
        val_dataset_pyg = HyperspectralGraphDataset(patch_dataset=val_patch_dataset)
        
        # Create the model
        model = model_class(
            num_classes=NB_CLASSES,
            patch_size=PATCH_SIZE,
            in_channels=NB_CHANNELS_REDUCED,
            dropout_rate=DROPOUT_RATE
        ).to(device)
        
        # Train the model
        _, history = train_model(
            train_dataset=train_dataset_pyg,
            val_dataset=val_dataset_pyg,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            outdir=outdir,
            model_name=model_name,
            class_names=class_names,
            class_weights=class_weights,
            model=model  # Pass the model explicitly
        )
        
        # Store the history
        all_models_history[model_name] = history
        
        # Save combined history after each model training
        with open(outdir + "all_models_history.pkl", "wb") as f:
            pickle.dump(all_models_history, f)
        
        # Clear memory
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Create comparison plots
    create_comparison_plots(all_models_history, class_names, outdir)
    
    return all_models_history

# Training code
def train_model(train_dataset, 
                val_dataset, 
                batch_size, 
                epochs, 
                device,
                outdir,
                model_name,
                class_names,
                class_weights,
                model=None):
   
    # Enable cuDNN benchmarking and deterministic algorithms
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # If encounter error with pin memory change it for hybrid4
    if model_name == 'HybridModel_4':
        my_collate_fn_train = collate_fn_graph
        my_collate_fn_val = collate_fn_graph
        #PIN_MEMORY = False
    else:
        my_collate_fn_train = collate_fn_mixup if APPLY_MIXUP else collate_fn
        my_collate_fn_val = collate_fn 
        #PIN_MEMORY = True

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True,
                              pin_memory=True,#PIN_MEMORY
                              persistent_workers=True,
                              num_workers=NUM_WORKERS,
                              drop_last=True,
                              prefetch_factor=PREFETCH_FACTOR,
                              collate_fn=my_collate_fn_train)

    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            pin_memory=True,#PIN_MEMORY
                            persistent_workers=True,
                            num_workers=NUM_WORKERS,
                            drop_last=True,
                            prefetch_factor=PREFETCH_FACTOR,
                            collate_fn=my_collate_fn_val)

    if model_name == 'HybridModel_4':
        # Dummy forward pass for initialization
        dummy_input = torch.randn(1, 
                                  NB_CHANNELS_REDUCED, 
                                  PATCH_SIZE, 
                                  PATCH_SIZE, 
                                  device=device)

        dummy_hetero_graph = HeteroData()

        # Example setup
        num_channels = NB_CHANNELS_REDUCED
        num_pixels = PATCH_SIZE * PATCH_SIZE

        # Dummy nodes per channel:
        for ch in range(num_channels):
            dummy_hetero_graph[f'ch_{ch}'].x = torch.randn(num_pixels, 1, device=device)

        # Dummy spatial edges (self-connections to avoid empty edges):
        edge_index = torch.tensor([[0], [0]], device=device)
        for ch in range(num_channels):
            dummy_hetero_graph[f'ch_{ch}', 'spatial', f'ch_{ch}'].edge_index = edge_index

        # Dummy inter-channel edges:
        for src_ch in range(num_channels):
            for tgt_ch in range(num_channels):
                if src_ch != tgt_ch:
                    dummy_hetero_graph[f'ch_{src_ch}', 'inter_channel', f'ch_{tgt_ch}'].edge_index = edge_index

        with torch.no_grad():
            model(dummy_input, dummy_hetero_graph)

    # Enable channels_last memory format
    model = model.to(memory_format=torch.channels_last)

    if model_name == 'HybridModel_4':
        optimizer = torch.optim.AdamW(
            [
                {'params': model.cnn_branch.parameters(), 'lr': LEARNING_RATE},
                {'params': model.vit_branch.parameters(), 'lr': LEARNING_RATE},
                {'params': model.gnn_branch.parameters(), 'lr': 5e-5}
            ]
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr=LEARNING_RATE, 
                                      weight_decay=WEIGHT_DECAY)

    # Convert to tensor
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    else:
        class_weights_tensor = None

    criterion = FocalLoss(alpha=0.5, 
                          gamma=2, 
                          class_weights=class_weights_tensor)
    # Use learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode='min',
                                                           factor=SCHEDULER_FACTOR,
                                                           patience=SCHEDULER_PATIENCE,
                                                           min_lr=1e-6)
    train_losses = []
    train_ious = []
    train_dices = []
    val_losses = []
    val_ious = []
    val_dices = []
    train_precision_history = []
    train_recall_history = []
    train_f1_history = []
    val_precision_history = []
    val_recall_history = []
    val_f1_history = []

    train_dice_history = []
    train_iou_history = []
    val_dice_history = []
    val_iou_history = []

    # Initialize Matthews Correlation Coefficient metric
    train_mcc_metric = MulticlassMatthewsCorrCoef(num_classes=NB_CLASSES).to(device)
    val_mcc_metric = MulticlassMatthewsCorrCoef(num_classes=NB_CLASSES).to(device)
    
    # Initialize Accuracy metric
    train_acc_metric = MulticlassAccuracy(num_classes=NB_CLASSES, average='weighted').to(device)
    val_acc_metric = MulticlassAccuracy(num_classes=NB_CLASSES, average='weighted').to(device)

    # Add history trackers for new metrics
    train_mcc_history = []
    val_mcc_history = []
    train_acc_history = []
    val_acc_history = []

    # Early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stopping_patience = EARLY_STOPPING

    # Mixed precision
    # allows FP16 computations where possible,
    # but keeps critical operations in FP32
    scaler = torch.amp.GradScaler()

    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()

    # Measure training time
    start_training_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_train_preds = []
        all_train_labels = []

        # Enable profiler only for the first epoch
        if PROFILE and epoch == 0:
            prof = profiler.profile(
                use_device='cuda',
                record_shapes=True,
                with_stack=True
            )

        for i, batch in enumerate(train_loader):
            cnn_imgs, labels, _ = batch[:3]
            hetero_graph = batch[3] if len(batch) > 3 else None
            cnn_imgs = cnn_imgs.contiguous(memory_format=torch.channels_last).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if hetero_graph is not None:
                hetero_graph = hetero_graph.to(device)
            optimizer.zero_grad()
            # Profile only the first batch in the first epoch
            if PROFILE and epoch == 0 and i == 0:
                with prof:
                    with profiler.record_function("model_training_step"):
                        if USE_AMP:
                            with torch.autocast(device_type='cuda'): 
                                if hetero_graph is not None:
                                    outputs = model(cnn_imgs, hetero_graph)
                                else:
                                    outputs = model(cnn_imgs)
                                loss = criterion(outputs, labels)
                            scaler.scale(loss).backward()   
                            # Apply gradient clipping before optimizer step
                            scaler.unscale_(optimizer)  # Unscale gradients before clipping
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            if hetero_graph is not None:
                                outputs = model(cnn_imgs, hetero_graph)
                            else:
                                outputs = model(cnn_imgs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            # Apply gradient clipping before optimizer step
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
            else:
                if USE_AMP:
                    with torch.autocast(device_type="cuda"):
                        if hetero_graph is not None:
                            outputs = model(cnn_imgs, hetero_graph)
                        else:
                            outputs = model(cnn_imgs)
                        loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    # Apply gradient clipping before optimizer step
                    scaler.unscale_(optimizer)  # Unscale gradients before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if hetero_graph is not None:
                        outputs = model(cnn_imgs, hetero_graph)
                    else:
                        outputs = model(cnn_imgs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    # Apply gradient clipping before optimizer step
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            # Upsample outputs if needed to match the label dimensions
            #if outputs.shape[2:] != labels.shape[1:]:
            outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)
           
            total_loss += loss.item() * cnn_imgs.size(0)
            preds = outputs.argmax(dim=1)

            # Calculate metrics for training batch
            train_mcc_metric.update(preds, labels)
            train_acc_metric.update(preds, labels)

            correct += (preds == labels).sum().item()
            total += labels.numel() 

            all_train_preds.append(preds.cpu().numpy().flatten())
            all_train_labels.append(labels.cpu().numpy().flatten())

            # Log GPU memory usage per batch
            #logging.info(f"Batch {i}: GPU Memory Usage - Current: {torch.cuda.memory_allocated()/1e9:.2f}GB, Peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

        # After each epoch
        memory_stats = {
            'current': torch.cuda.memory_allocated() - initial_memory,
            'peak': torch.cuda.max_memory_allocated() - initial_memory
        }
        logging.info(f"GPU Memory Usage - Current: {memory_stats['current']/1e9:.2f}GB, Peak: {memory_stats['peak']/1e9:.2f}GB")

        all_train_preds = np.concatenate(all_train_preds)
        all_train_labels = np.concatenate(all_train_labels)

        train_dice, train_iou, train_dice_per_class, train_iou_per_class = dice_iou_score_multiclass_weighted(
            all_train_preds, 
            all_train_labels, 
            NB_CLASSES, 
            class_weights_tensor
        )
                
        # Compute per-class precision, recall, and F1-score:
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            all_train_labels, all_train_preds, labels=range(NB_CLASSES), zero_division=0
        )
            
        # Calculate epoch metrics for training
        train_mcc = train_mcc_metric.compute()
        train_mcc_history.append(train_mcc.item())
        train_mcc_metric.reset()
        
        train_acc = train_acc_metric.compute()
        train_acc_history.append(train_acc.item())
        train_acc_metric.reset()
        
        train_precision_history.append(train_precision)
        train_recall_history.append(train_recall)
        train_f1_history.append(train_f1)

        train_loss = total_loss / len(train_dataset)
        train_losses.append(train_loss)
        train_ious.append(train_iou)
        train_dices.append(train_dice)
        train_dice_history.append(train_dice_per_class)
        train_iou_history.append(train_iou_per_class)

        # Print profiler results only once (after first epoch)
        if PROFILE and epoch == 0:
            logging.info(prof.key_averages().table(sort_by="cuda_time_total"))
        
        # Validation:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                cnn_imgs, labels, _ = batch[:3]
                hetero_graph = batch[3] if len(batch) > 3 else None
                cnn_imgs = cnn_imgs.contiguous(memory_format=torch.channels_last).to(device, non_blocking=True)
                labels = labels.to(device)
                if hetero_graph is not None:
                    hetero_graph = hetero_graph.to(device)
                if USE_AMP:
                    with torch.autocast(device_type="cuda"):
                        if hetero_graph is not None:
                            outputs = model(cnn_imgs, hetero_graph)
                        else:
                            outputs = model(cnn_imgs)
                        loss = criterion(outputs, labels)
                else:
                    if hetero_graph is not None:
                        outputs = model(cnn_imgs, hetero_graph)
                    else:
                        outputs = model(cnn_imgs)
                    loss = criterion(outputs, labels)

                # Upsample outputs if needed to match the label dimensions
                #if outputs.shape[2:] != labels.shape[1:]:
                outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)

                total_loss += loss.item() * cnn_imgs.size(0)
                preds = outputs.argmax(dim=1)

                # Calculate metrics for validation batch
                val_mcc_metric.update(preds, labels)
                val_acc_metric.update(preds, labels)

                correct += (preds == labels).sum().item()
                total += labels.numel() 

                all_val_preds.append(preds.cpu().numpy().flatten())
                all_val_labels.append(labels.cpu().numpy().flatten())

        all_val_preds = np.concatenate(all_val_preds)
        all_val_labels = np.concatenate(all_val_labels)

        val_dice, val_iou, val_dice_per_class, val_iou_per_class = dice_iou_score_multiclass_weighted(
            all_val_preds, 
            all_val_labels, 
            NB_CLASSES, 
            class_weights_tensor
        )
                
        # Compute per-class precision, recall, and F1-score:
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            all_val_labels, all_val_preds, labels=range(NB_CLASSES), zero_division=0
        )

        # Calculate epoch metrics for validation
        val_mcc = val_mcc_metric.compute()
        val_mcc_history.append(val_mcc.item())
        val_mcc_metric.reset()
        
        val_acc = val_acc_metric.compute()
        val_acc_history.append(val_acc.item())
        val_acc_metric.reset()

        val_precision_history.append(val_precision)
        val_recall_history.append(val_recall)
        val_f1_history.append(val_f1)

        val_loss = total_loss / len(val_dataset)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        val_dices.append(val_dice)
        val_dice_history.append(val_dice_per_class)
        val_iou_history.append(val_iou_per_class)


        logging.info(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, " +
            f"Train Dice: {train_dice:.4f}, Train MCC: {train_mcc:.4f}, Train Acc: {train_acc:.4f} | " +
            f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}, " +
            f"Val MCC: {val_mcc:.4f}, Val Acc: {val_acc:.4f}"
        )

        for cls in range(NB_CLASSES):
            logging.info(
                f"Train F1 ({class_names[cls]}): {train_f1[cls]:.4f}, "
                f"Train Precision ({class_names[cls]}): {train_precision[cls]:.4f}, "
                f"Train Recall ({class_names[cls]}): {train_recall[cls]:.4f}, "
                f"Train IoU ({class_names[cls]}): {train_iou_per_class[cls].item():.4f}, "
                f"Train Dice ({class_names[cls]}): {train_dice_per_class[cls].item():.4f} | "
                f"Val F1 ({class_names[cls]}): {val_f1[cls]:.4f}, "
                f"Val Precision ({class_names[cls]}): {val_precision[cls]:.4f}, "
                f"Val Recall ({class_names[cls]}): {val_recall[cls]:.4f}, "
                f"Val IoU ({class_names[cls]}): {val_iou_per_class[cls].item():.4f}, "
                f"Val Dice ({class_names[cls]}): {val_dice_per_class[cls].item():.4f}"
            )
        
        # Step scheduler based on validation loss
        scheduler.step(val_loss)
        #scheduler.step()
        logging.info(f"Epoch {epoch+1}: Learning rate -> {scheduler.get_last_lr()}")

        # Early stopping 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0       
            # Global pruning
            # Traverse model and collect all Conv2d and Linear 
            # Prune only once!
            if epoch == 0:
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        # Ensure the tensor is contiguous before pruning (good practice)
                        module.weight.data = module.weight.data.contiguous()
                        # Apply structured pruning: prune 20% of output filters (dim=0)
                        prune.ln_structured(
                            module,
                            name='weight',
                            amount=0.2,
                            n=2,  # L2 norm
                            dim=0  # Prune entire output filters
                        )

                    elif isinstance(module, torch.nn.Linear):
                        module.weight.data = module.weight.data.contiguous()
                        # Apply structured pruning: prune 20% of neurons (input connections)
                        prune.ln_structured(
                            module,
                            name='weight',
                            amount=0.2,
                            n=2,  # L2 norm
                            dim=1  # Prune entire neurons (columns)
                        )

                named_module_dict = dict(model.named_modules())
                for name, module in named_module_dict.items():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                        if hasattr(module, 'weight'):
                            tensor = module.weight
                            sparsity = 100.0 * float(torch.sum(tensor == 0)) / float(tensor.nelement())
                            #print(f"Sparsity in {name}.weight: {sparsity:.2f}%")


                # Show global sparsity
                total_zeros = 0
                total_elements = 0

                for name, module in named_module_dict.items():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                        if hasattr(module, 'weight'):
                            tensor = module.weight
                            total_zeros += torch.sum(tensor == 0).item()
                            total_elements += tensor.nelement()

                global_sparsity = 100.0 * total_zeros / total_elements

                print(f"Global sparsity: {global_sparsity:.2f}%")


                # Make pruning permantly
                # replace the reparameterized weight with the pruned one directly
                for name, module in named_module_dict.items():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                        if hasattr(module, 'weight_orig'):
                            prune.remove(module, 'weight')

            # Save best model
            torch.save(model.state_dict(), OUTPUT_DIR + f"{model_name}_best.pth")
        else:
            early_stop_counter += 1  
        
        if early_stop_counter >= early_stopping_patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break

    # Total training time
    total_training_time = time.time() - start_training_time
    logging.info(f"Total Training Time: {(total_training_time / 3600.0):.2f} hr")

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_ious': train_ious,
        'val_ious': val_ious,
        'train_dices': train_dices,
        'train_dice_per_class': train_dice_history,
        'train_iou_per_class': train_iou_history,
        'val_dices': val_dices,
        'val_dice_per_class': val_dice_history,
        'val_iou_per_class': val_iou_history,
        'train_precision_history': train_precision_history,
        'val_precision_history': val_precision_history,
        'train_recall_history': train_recall_history,
        'val_recall_history': val_recall_history,
        'train_f1_history': train_f1_history,
        'val_f1_history': val_f1_history,
        'train_mcc': train_mcc_history,
        'val_mcc': val_mcc_history,
        'train_accuracy': train_acc_history,
        'val_accuracy': val_acc_history
    }

    with open(OUTPUT_DIR + f"{model_name}_history.pkl", "wb") as f:
        pickle.dump(history, f)

    create_individual_plots(history, model_name, outdir)
    
    return model, history
