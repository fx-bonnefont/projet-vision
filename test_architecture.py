
import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from model import build_model, get_image_size

def test_architecture():
    print("="*60)
    print("ARCHiTECTURE SANITY CHECK")
    print("="*60)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    try:
        # 1. Build Model
        print("\n[1/3] Building Model (DINOv3 ViT-S/16)...")
        model = build_model('dinov3_vits16', device=device)
        print("  ✅ Model built successfully.")
        
        # 2. Prepare Dummy Input
        img_size = get_image_size('dinov3_vits16') # 512
        print(f"\n[2/3] Creating dummy input [1, 3, {img_size}, {img_size}]...")
        x = torch.randn(1, 3, img_size, img_size).to(device)
        
        # 3. Forward Pass
        print("\n[3/3] Running forward pass...")
        with torch.no_grad():
            out = model(x)
        
        print(f"  ✅ Output Shape: {out.shape}")
        
        if out.shape == (1, 1, img_size, img_size):
            print("\n🎉 SUCCESS! The architecture handles the flow correctly.")
        else:
            print(f"\n⚠️ WARNING: Output shape mismatch. Expected (1, 1, {img_size}, {img_size})")

    except Exception as e:
        print(f"\n❌ FAIL: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_architecture()
