import time
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import psutil

# Configuration
DATA_ROOT = "/Volumes/X9Pro/DOTA/images/train"  # Remplacez par le chemin réel si différent
IMG_SIZE = 512
BATCH_SIZE = 16
NUM_IMAGES_TO_TEST = 500

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def test_disk_io_random_read(image_paths):
    print_header("1. DISK IO BENCHMARK (Random Read)")
    print(f"Reading {len(image_paths)} images from disk...")
    
    start_time = time.time()
    total_bytes = 0
    
    for p in tqdm(image_paths, leave=False):
        img_bytes = p.read_bytes()
        total_bytes += len(img_bytes)
        
    duration = time.time() - start_time
    mb_per_sec = (total_bytes / 1024 / 1024) / duration
    
    print(f"-> Throughput: {mb_per_sec:.2f} MB/s")
    print(f"-> Latency: {(duration/len(image_paths))*1000:.2f} ms/image")
    print(f"-> Total Time: {duration:.2f} s")

def test_image_decoding(image_paths):
    print_header("2. CPU IMAGE DECODING (cv2.imread)")
    print(f"Decoding {len(image_paths)} images...")
    
    start_time = time.time()
    
    for p in tqdm(image_paths, leave=False):
        img = cv2.imread(str(p))
        if img is not None:
            # Simulate resize which is part of loading
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
    duration = time.time() - start_time
    fps = len(image_paths) / duration
    
    print(f"-> Speed: {fps:.2f} images/sec")
    print(f"-> Total Time: {duration:.2f} s")

def worker_load(path):
    img = cv2.imread(str(path))
    if img is not None:
        return cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return None

def test_multiprocessing_overhead(image_paths, workers=[0, 4, 8]):
    print_header("3. MULTIPROCESSING SCALING")
    
    for w in workers:
        print(f"\n--- Testing with {w} workers ---")
        start_time = time.time()
        
        if w == 0:
            for p in tqdm(image_paths, leave=False):
                worker_load(p)
        else:
            with multiprocessing.Pool(w) as pool:
                list(tqdm(pool.imap(worker_load, image_paths), total=len(image_paths), leave=False))
                
        duration = time.time() - start_time
        fps = len(image_paths) / duration
        print(f"-> Speed: {fps:.2f} images/sec")

def test_mps_tensor_transfer():
    print_header("4. RAM -> GPU (MPS) TRANSFER SPEED")
    
    if not torch.backends.mps.is_available():
        print("MPS not available, skipping.")
        return

    device = torch.device("mps")
    # Simulate a batch of float32 images
    tensor_size = (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    data = torch.randn(tensor_size, dtype=torch.float32)
    mb_size = (data.element_size() * data.nelement()) / 1024 / 1024
    
    print(f"Transferring Batch of {mb_size:.2f} MB...")
    
    # Warmup
    data.to(device)
    
    start_time = time.time()
    iters = 50
    for _ in range(iters):
        _ = data.to(device, non_blocking=True)
        torch.mps.synchronize()
        
    duration = time.time() - start_time
    avg_time = duration / iters
    bandwidth = mb_size / avg_time
    
    print(f"-> Bandwidth: {bandwidth:.2f} MB/s")
    print(f"-> Latency per Batch: {avg_time*1000:.2f} ms")

def test_compute_capability():
    print_header("5. COMPUTE SPEED (FLOPS check)")
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
        
    runs = 100
    size = 2048
    
    for dev in devices:
        print(f"\n--- Device: {dev.upper()} ---")
        device = torch.device(dev)
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warmup
        torch.matmul(a, b)
        if dev == 'mps': torch.mps.synchronize()
        
        start = time.time()
        for _ in range(runs):
            c = torch.matmul(a, b)
        if dev == 'mps': torch.mps.synchronize()
        
        duration = time.time() - start
        avg_time = duration / runs
        # TFLOPS = 2 * n^3 / time / 1e12
        tflops = (2 * size**3) / avg_time / 1e12
        
        print(f"-> Matrix Mul ({size}x{size}): {avg_time*1000:.2f} ms")
        print(f"-> Est. Performance: {tflops:.3f} TFLOPS")

if __name__ == "__main__":
    # 1. Find images
    p = Path(DATA_ROOT)
    if not p.exists():
        print(f"Error: Path {DATA_ROOT} not found. Please edit script.")
        exit(1)
        
    all_images = list(p.glob("*.png")) + list(p.glob("*.jpg"))
    if not all_images:
        print("No images found.")
        exit(1)
        
    # Take separate subset to avoid OS caching bias if possible (or just shuffle)
    np.random.shuffle(all_images)
    test_images = all_images[:NUM_IMAGES_TO_TEST]
    
    print(f"Benchmark starting on {len(test_images)} images...")
    print(f"System: {psutil.cpu_count()} CPU cores, {psutil.virtual_memory().total / 1e9:.1f} GB RAM")
    
    test_disk_io_random_read(test_images)
    test_image_decoding(test_images)
    test_multiprocessing_overhead(test_images, workers=[0, 4, 8])
    test_mps_tensor_transfer()
    test_compute_capability()
