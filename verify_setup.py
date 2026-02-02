#!/usr/bin/env python3
"""
Pre-deployment verification script.
Run this before submitting to the cluster to catch common issues.
"""
import os
import sys
import subprocess

def print_header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print('='*60)

def check_pass(msg):
    print(f"  [OK] {msg}")

def check_fail(msg):
    print(f"  [FAIL] {msg}")
    return False

def check_warn(msg):
    print(f"  [WARN] {msg}")

# ========================
# Checks
# ========================

all_passed = True

print_header("SegDino Deployment Verification")

# 1. Python version
print("\n[1] Python Version")
version = sys.version_info
if version.major == 3 and version.minor >= 10:
    check_pass(f"Python {version.major}.{version.minor}.{version.micro}")
else:
    all_passed = check_fail(f"Python {version.major}.{version.minor} (need 3.10+)")

# 2. Required files
print("\n[2] Required Files")
required_files = [
    'train.py',
    'model.py',
    'dataset.py',
    'loss.py',
    'train_ddp.sh',
    'test_single_gpu.sh',
    '.env'
]

for f in required_files:
    if os.path.exists(f):
        check_pass(f)
    else:
        all_passed = check_fail(f"{f} missing")

# 3. Scripts are executable
print("\n[3] Script Permissions")
for script in ['train_ddp.sh', 'test_single_gpu.sh']:
    if os.path.exists(script) and os.access(script, os.X_OK):
        check_pass(f"{script} is executable")
    else:
        check_warn(f"{script} not executable (run: chmod +x {script})")

# 4. PyTorch
print("\n[4] PyTorch Installation")
try:
    import torch
    check_pass(f"PyTorch {torch.__version__}")

    if torch.cuda.is_available():
        check_pass(f"CUDA {torch.version.cuda} available")
        check_pass(f"{torch.cuda.device_count()} GPU(s) detected")
        for i in range(torch.cuda.device_count()):
            check_pass(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        check_warn("CUDA not available (OK for CPU testing)")
except ImportError:
    all_passed = check_fail("PyTorch not installed (pip install torch torchvision)")

# 5. Other dependencies
print("\n[5] Dependencies")
deps = {
    'transformers': 'Transformers',
    'cv2': 'OpenCV',
    'PIL': 'Pillow',
    'tqdm': 'tqdm',
    'dotenv': 'python-dotenv'
}

for module, name in deps.items():
    try:
        __import__(module)
        check_pass(name)
    except ImportError:
        all_passed = check_fail(f"{name} not installed")

# 6. HuggingFace token
print("\n[6] HuggingFace Configuration")
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        content = f.read()
        if 'HF_TOKEN' in content and 'your_token_here' not in content:
            check_pass(".env file configured with HF_TOKEN")
        else:
            check_warn(".env exists but HF_TOKEN not set or is placeholder")
else:
    check_warn(".env file not found (create with HF_TOKEN=your_token)")

# 7. Directories
print("\n[7] Directory Structure")
dirs = ['runs', 'logs']
for d in dirs:
    if os.path.exists(d):
        check_pass(f"{d}/ exists")
    else:
        check_warn(f"{d}/ missing (will be created automatically)")

# 8. Model aliases check
print("\n[8] Model Configuration")
try:
    from model import DINOV3_MODELS
    expected_models = ['vit-small', 'vit-base', 'vit-large', 'vit-large-sat']
    if all(m in DINOV3_MODELS for m in expected_models):
        check_pass(f"{len(DINOV3_MODELS)} model variants available")
        for name in expected_models:
            check_pass(f"  - {name}: {DINOV3_MODELS[name].split('/')[-1]}")
    else:
        all_passed = check_fail("Model aliases not properly configured")
except Exception as e:
    all_passed = check_fail(f"Error loading model config: {e}")

# 9. SLURM script configuration
print("\n[9] SLURM Script Configuration")
if os.path.exists('train_ddp.sh'):
    with open('train_ddp.sh', 'r') as f:
        content = f.read()
        if 'DATA_DIR=' in content:
            # Extract DATA_DIR value
            for line in content.split('\n'):
                if 'DATA_DIR=' in line and not line.strip().startswith('#'):
                    check_pass(f"DATA_DIR configured: {line.strip()}")
                    if '/path/to/your/DOTA_PLANES_TILED' in line:
                        check_warn("DATA_DIR still has placeholder value - update before cluster submission!")
                    break
        else:
            check_warn("DATA_DIR not found in train_ddp.sh")

# ========================
# Summary
# ========================

print_header("Verification Summary")

if all_passed:
    print("\n  [OK] All checks passed!")
    print("  → Ready for deployment to cluster")
    print("\n  Next steps:")
    print("    1. Test locally: ./test_single_gpu.sh")
    print("    2. Update DATA_DIR in train_ddp.sh")
    print("    3. Submit to cluster: sbatch train_ddp.sh")
    sys.exit(0)
else:
    print("\n  [FAIL] Some checks failed")
    print("  → Fix issues above before deployment")
    print("\n  Common fixes:")
    print("    - pip install torch torchvision transformers opencv-python-headless tqdm python-dotenv pillow")
    print("    - chmod +x train_ddp.sh test_single_gpu.sh")
    print("    - Create .env with HF_TOKEN=your_token")
    sys.exit(1)
