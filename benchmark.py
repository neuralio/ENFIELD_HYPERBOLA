import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import psutil
import os
import pandas as pd
from inference import inference_with_weighted_blending
from edge_inference import inference_with_weighted_blending_trt
from models import HybridModel_3
from config import OUTPUT_DIR, PATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CHANNELS, \
    NB_CLASSES, OVERLAP_STEP, TEST_PATH
from matplotlib_config import configure_matplotlib
# Apply global settings
configure_matplotlib()

def measure_memory_cpu():
    # In MB
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def measure_memory_gpu():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024  # In MB
    else:
        return 0.0

def measure_peak_memory_gpu():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        return 0.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
model_path = os.path.join(OUTPUT_DIR, "Unet_best.pth")
engine_path = os.path.join(OUTPUT_DIR, "Unet.trt")

# Test images
test_image_paths = [
    TEST_PATH + "273-20230217_CaptureDL_lakevolta_2023-02-15_0953Z/DATA/273-20230217_CaptureDL_lakevolta_2023-02-15_0953Z-radiance.npy",
    TEST_PATH + "25-20221027_CaptureDL_bangladesh_2022_10_26T04_02_54/DATA/25-20221027_CaptureDL_bangladesh_2022_10_26T04_02_54-radiance.npy",
    TEST_PATH + "26-Trondheim_2022_08_23T10_26_43/DATA/26-Trondheim_2022_08_23T10_26_43-radiance.npy"
]

# Load model for normal inference
model = HybridModel_3(
    num_classes=NB_CLASSES,
    patch_size=PATCH_SIZE,
    in_channels=len(CHANNELS),
    dropout_rate=0.3
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Benchmark results
results = []

# --- Benchmark 1: Non-Quantized (Normal PyTorch) ---
print("\n[Non-Quantized Inference]")

start_mem_cpu = measure_memory_cpu()
start_mem_gpu = measure_memory_gpu()

start_time = time.perf_counter()

for img_path in test_image_paths:
    prediction = inference_with_weighted_blending(
        model=model,
        model_name="Unet",
        image_path=img_path,
        target_width=IMG_WIDTH,
        target_height=IMG_HEIGHT,
        patch_size=PATCH_SIZE,
        overlap_step=OVERLAP_STEP,
        device=device,
    )

if torch.cuda.is_available():
    torch.cuda.synchronize()

end_time = time.perf_counter()
end_mem_cpu = measure_memory_cpu()
end_mem_gpu = measure_memory_gpu()

elapsed_time_normal = end_time - start_time
mem_used_normal_cpu = end_mem_cpu - start_mem_cpu
mem_used_normal_gpu = measure_peak_memory_gpu()

print(f"Inference Time (Normal): {elapsed_time_normal:.4f} seconds")
print(f"Memory Used (Normal, CPU): {mem_used_normal_cpu:.2f} MB")
print(f"Memory Used (Normal, GPU peak): {mem_used_normal_gpu:.2f} MB")

results.append(("Normal Inference", elapsed_time_normal, mem_used_normal_cpu, mem_used_normal_gpu))


# --- Benchmark 2: Quantized (TensorRT) ---
print("\n[Quantized TensorRT Inference]")

start_mem_cpu = measure_memory_cpu()
start_mem_gpu = measure_memory_gpu()

start_time = time.perf_counter()

for img_path in test_image_paths:
    prediction = inference_with_weighted_blending_trt(
        engine_path=engine_path,
        model_name="Unet",
        image_path=img_path,
        target_width=IMG_WIDTH,
        target_height=IMG_HEIGHT,
        patch_size=PATCH_SIZE,
        overlap_step=OVERLAP_STEP,
    )

end_time = time.perf_counter()
end_mem_cpu = measure_memory_cpu()
end_mem_gpu = measure_memory_gpu()

elapsed_time_trt = end_time - start_time
mem_used_trt_cpu = end_mem_cpu - start_mem_cpu
mem_used_trt_gpu = measure_peak_memory_gpu()

print(f"Inference Time (TensorRT): {elapsed_time_trt:.4f} seconds")
print(f"Memory Used (TensorRT, CPU): {mem_used_trt_cpu:.2f} MB")
print(f"Memory Used (TensorRT, GPU peak): {mem_used_trt_gpu:.2f} MB")

results.append(("TensorRT Quantized", elapsed_time_trt, mem_used_trt_cpu, mem_used_trt_gpu))

# --- Final Comparison ---
print("\n===== Benchmark Results =====")
for name, elapsed, mem_cpu, mem_gpu in results:
    print(f"{name}: Time = {elapsed:.4f}s, CPU Mem = {mem_cpu:.2f}MB, GPU Mem Peak = {mem_gpu:.2f}MB")

# --- Create a Plot ---
labels = [r[0] for r in results]
times = [r[1] for r in results]
cpu_mems = [r[2] for r in results]
gpu_mems = [r[3] for r in results]

x = np.arange(len(labels))

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].bar(x, times, color='skyblue')
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels, rotation=45)
axs[0].set_title('Inference Time (s)')
axs[0].grid(False)

axs[1].bar(x, cpu_mems, color='salmon')
axs[1].set_xticks(x)
axs[1].set_xticklabels(labels, rotation=45)
axs[1].set_title('CPU Memory Usage (MB)')
axs[1].grid(False)

axs[2].bar(x, gpu_mems, color='lightgreen')
axs[2].set_xticks(x)
axs[2].set_xticklabels(labels, rotation=45)
axs[2].set_title('GPU Peak Memory Usage (MB)')
axs[2].grid(False)

plt.tight_layout()
plot_path = OUTPUT_DIR + "benchmark_plot.png"
plt.savefig(plot_path)
plt.show()

print(f"\nSaved benchmark plot to {plot_path}")

# --- Create CSV ---
csv_path = OUTPUT_DIR + "benchmark_results.csv"
df = pd.DataFrame({
    "Inference Type": labels,
    "Inference Time (s)": times,
    "CPU Memory Usage (MB)": cpu_mems,
    "GPU Peak Memory Usage (MB)": gpu_mems
})
df.to_csv(csv_path, index=False)
print(f"Saved benchmark results to {csv_path}")
