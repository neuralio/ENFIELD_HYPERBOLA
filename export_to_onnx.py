import os
import torch.ao.quantization as tq
import torch
from torch.quantization.quantize_fx import prepare_fx
from utils import SplitPatchDataDataset, normalize_spectral_signature
from models import HybridModel_3, HybridModel_4
from config import NB_CLASSES, NB_CHANNELS_REDUCED, PATCH_SIZE, DROPOUT_RATE, OUTPUT_DIR, \
    BASE_PATH, VAL_SPLIT, IMG_HEIGHT, IMG_WIDTH, CHANNELS, OVERLAP_STEP

def fuse_model(model, model_name):
    if model_name == "Unet":
        """Fuses Conv + ReLU layers inside your EnhancedUNet."""
        fused_layers = [
            # Downsample path
            ['cnn_branch.down1.conv1.0', 'cnn_branch.down1.conv1.2'],
            ['cnn_branch.down2.conv1.0', 'cnn_branch.down2.conv1.2'],
            ['cnn_branch.down3.conv1.0', 'cnn_branch.down3.conv1.2'],
            ['cnn_branch.bridge.conv1.0', 'cnn_branch.bridge.conv1.2'],
            # Upsample path
            ['cnn_branch.up1.0', 'cnn_branch.up1.2'],
            ['cnn_branch.up1.4', 'cnn_branch.up1.6'],
            ['cnn_branch.up2.0', 'cnn_branch.up2.2'],
            ['cnn_branch.up2.4', 'cnn_branch.up2.6'],
            ['cnn_branch.up3.0', 'cnn_branch.up3.2'],
            ['cnn_branch.up3.4', 'cnn_branch.up3.6'],
        ]
    elif model_name == "HybridModel_4":
        fused_layers = [
            ['cnn_branch.down1.conv1.0', 'cnn_branch.down1.conv1.2'],
            ['cnn_branch.down2.conv1.0', 'cnn_branch.down2.conv1.2'],
            ['cnn_branch.down3.conv1.0', 'cnn_branch.down3.conv1.2'],
            ['cnn_branch.bridge.conv1.0', 'cnn_branch.bridge.conv1.2'],
            ['cnn_branch.up1.0', 'cnn_branch.up1.2'],
            ['cnn_branch.up1.4', 'cnn_branch.up1.6'],
            ['cnn_branch.up2.0', 'cnn_branch.up2.2'],
            ['cnn_branch.up2.4', 'cnn_branch.up2.6'],
            ['cnn_branch.up3.0', 'cnn_branch.up3.2'],
            ['cnn_branch.up3.4', 'cnn_branch.up3.6'],
            # Add ViT decoder fusions
            ['vit_branch.segmentation_head.decoder.0', 'vit_branch.segmentation_head.decoder.1'],
        ]
    torch.quantization.fuse_modules(model, fused_layers, inplace=True)

def load_train_dataset(model_name):
    """Loads one real validation patch for calibration."""
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
        transform=None
    )
    split_dataset.set_model_name(model_name)
    train_dataset = split_dataset.get_train_dataset()
    return train_dataset

def load_validation_dataset(model_name):
    """Loads one real validation patch for calibration."""
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
    split_dataset.set_model_name(model_name)
    val_dataset = split_dataset.get_val_dataset()
    return val_dataset

def export_quantized_onnx(model_name):
    checkpoint_path = os.path.join(OUTPUT_DIR, f"{model_name}_best.pth")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_name == "HybridModel_4":
        model = HybridModel_4(
            num_classes=NB_CLASSES,
            patch_size=PATCH_SIZE,
            in_channels=NB_CHANNELS_REDUCED,
            dropout_rate=DROPOUT_RATE
        )
    elif model_name == "Unet":
        model = HybridModel_3(
            num_classes=NB_CLASSES,
            patch_size=PATCH_SIZE,
            in_channels=NB_CHANNELS_REDUCED,
            dropout_rate=DROPOUT_RATE
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).train()

    # Fuse model
    fuse_model(model, model_name)

    # Load calibration patches
    train_dataset = load_train_dataset(model_name)
    val_dataset = load_validation_dataset(model_name)
    
    # Create custom QConfig without reduce_range
    activation_observer = tq.observer.HistogramObserver.with_args(
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric
    )
    
    weight_observer = tq.observer.MinMaxObserver.with_args(
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric
    )
    
    custom_qconfig = tq.QConfig(
        activation=activation_observer,
        weight=weight_observer
    )
    
    qconfig_mapping = tq.QConfigMapping().set_global(custom_qconfig)
    qconfig_mapping = qconfig_mapping.set_object_type(torch.nn.ConvTranspose2d, None)  # do NOT quantize ConvTranspose2d

    # Prepare model for QAT
    dummy_input = torch.randn(1, NB_CHANNELS_REDUCED, PATCH_SIZE, PATCH_SIZE).to(device)
    model_prepared = tq.quantize_fx.prepare_qat_fx(model, qconfig_mapping, example_inputs=dummy_input)

    # Fine-tune model a few epochs with normal training loop
    print("Training with Quantization Aware Training for few epochs...")

    optimizer = torch.optim.Adam(model_prepared.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    batch_size = 16
    num_epochs = 30

    for epoch in range(num_epochs):
        model_prepared.train()
        running_loss = 0.0
        total_samples = 0

        for idx in range(0, len(train_dataset), batch_size):
            inputs = []
            targets = []

            for b in range(batch_size):
                if idx + b >= len(train_dataset):
                    break
                patch_tensor, patch_label, _ = train_dataset[idx + b]
                patch_tensor = normalize_spectral_signature(patch_tensor)
                inputs.append(patch_tensor.unsqueeze(0))
                targets.append(patch_label.unsqueeze(0))

            if not inputs:
                continue

            inputs = torch.cat(inputs, dim=0).float().to(device)  
            targets = torch.cat(targets, dim=0).long().to(device)

            optimizer.zero_grad()
            outputs = model_prepared(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        avg_loss = running_loss / total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.6f}")

        # Validation Step
        model_prepared.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for idx in range(0, len(val_dataset)):
                patch_tensor, patch_label, _ = val_dataset[idx]
                patch_tensor = normalize_spectral_signature(patch_tensor)
                patch_tensor = patch_tensor.unsqueeze(0).float().to(device)
                patch_label = patch_label.unsqueeze(0).long().to(device)

                output = model_prepared(patch_tensor)
                preds = output.argmax(dim=1)  

                correct += (preds == patch_label).sum().item()
                total += patch_label.numel()

        val_accuracy = 100.0 * correct / total
        print(f"Validation Accuracy after Epoch [{epoch+1}/{num_epochs}]: {val_accuracy:.2f}%")


    # Convert to quantized model
    model_prepared.cpu()
    model_quantized = tq.quantize_fx.convert_fx(model_prepared)
    model_quantized.eval()
    dummy_input = dummy_input.cpu()

    torch.onnx.export(
        model_quantized,
        dummy_input,
        OUTPUT_DIR + f"{model_name}.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
    )

    print("Exported quantized ONNX model successfully.")

if __name__ == "__main__":
    for name in ["Unet"]: #HybridModel_4
        export_quantized_onnx(model_name=name)
