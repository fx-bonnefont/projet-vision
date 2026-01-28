import csv
import time
import os
import torch
import psutil
from datetime import datetime
from pathlib import Path

class SystemLogger:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.log_file = self.output_dir / f"training_log_{timestamp}.csv"
        
        # Define headers
        self.headers = [
            'timestamp', 'epoch', 'batch', 'mode', 
            'loss', 'grad_norm', 'pos_pixel_ratio', 
            'lr', 'ram_gb', 'vram_gb'
        ]
        
        # Create file with headers
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            
        print(f"SystemLogger initialized. Logging to: {self.log_file}")
        
    def log_batch(self, 
                  epoch: int, 
                  batch: int, 
                  mode: str, 
                  loss: float, 
                  grad_norm: float = 0.0, 
                  pos_pixel_ratio: float = 0.0,
                  lr: float = 0.0):
        
        # System metrics
        ram_gb = psutil.virtual_memory().used / 1e9
        
        vram_gb = 0.0
        if torch.backends.mps.is_available():
            # MPS doesn't expose memory API easily in python like cuda
            # We can only track process RSS which includes VRAM roughly
            vram_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
        elif torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / 1e9
            
        row = [
            datetime.now().isoformat(),
            epoch,
            batch,
            mode,
            f"{loss:.6f}",
            f"{grad_norm:.4f}",
            f"{pos_pixel_ratio:.4f}",
            f"{lr:.6f}",
            f"{ram_gb:.2f}",
            f"{vram_gb:.2f}"
        ]
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
