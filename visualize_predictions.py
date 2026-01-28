import torch
import cv2
import numpy as np
import os
import argparse
import random
from model import load_model, get_image_size
from dataset import SegmentationDataset

def visualize(model_path, data_dir, num_images=5, output_dir='visualizations', device=None):
    """
    Load model and run inference on random images, saving visualizations using OpenCV.
    Uses SegmentationDataset to correctly parse DOTA/YOLO text labels into masks.
    """
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Setup paths
    img_dir = os.path.join(data_dir, 'images', 'test')
    lbl_dir = os.path.join(data_dir, 'labels', 'test')
    
    if not os.path.exists(img_dir):
        img_dir = os.path.join(data_dir, 'images', 'train')
        lbl_dir = os.path.join(data_dir, 'labels', 'train')
        print("Test set not found, using Train set for visualization.")

    # Initialize model
    try:
        model = load_model(weights_path=model_path, device=device)
        model.eval()
        backbone_name = model.backbone.config.name
        img_size = get_image_size(backbone_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize Dataset (to handle text parsing)
    dataset = SegmentationDataset(
        image_dir=img_dir,
        label_dir=lbl_dir,
        img_size=img_size,
        cache_data=False 
    )
    
    if len(dataset) == 0:
        print("No images found!")
        return

    # Create output directory and clear old files
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Pick random indices
    indices = random.sample(range(len(dataset)), min(len(dataset), num_images))
    print(f"Processing {len(indices)} images...")
    
    # Normalization constants for inference
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for idx in indices:
            path_obj = dataset.image_files[idx]
            name_no_ext = path_obj.stem
            
            # 1. Load Original Image (for dimensions and debug view)
            # We bypass dataset._load_raw_data resizing to get original dimensions
            original_safe = cv2.imread(str(path_obj))
            if original_safe is None: continue
            orig_h, orig_w = original_safe.shape[:2]
            
            # 2. Check for Debug Image (Optimized/Annotated)
            # DOTA debug images have prefix 'visu_'
            # Reconstruct debug path: data_dir/debug/(train|test)/visu_filename
            subset = path_obj.parent.name # 'train' or 'test'
            debug_fname = f"visu_{path_obj.name}"
            debug_path = os.path.join(data_dir, 'debug', subset, debug_fname)
            
            if os.path.exists(debug_path):
                input_display = cv2.imread(debug_path)
            else:
                print(f"Debug image not found at {debug_path}, using original.")
                input_display = original_safe

            # 3. Prepare Input for Model
            # Resize to model input size (e.g. 512x512)
            img_resized = cv2.resize(original_safe, (img_size, img_size))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb.astype(np.float32) / 255.0
            img_norm = (img_norm - IMAGENET_MEAN) / IMAGENET_STD
            img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float().unsqueeze(0).to(device)
            
            # 4. Inference
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # 5. Process Output
            # Threshold > 0.5
            pred_mask_512 = (probs > 0.5).astype(np.uint8) * 255
            
            # Resize prediction BACK to original resolution
            pred_mask_orig = cv2.resize(pred_mask_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # 6. Load/Generate Ground Truth at Original Resolution
            # We use the dataset internal method to parse txt, but we pass orig sizes
            label_path = dataset.label_dir / f"{name_no_ext}.txt"
            gt_mask_orig = dataset._create_mask(label_path, orig_h, orig_w)
            gt_mask_orig = (gt_mask_orig * 255).astype(np.uint8)

            # 7. Save 3 Separate Images
            # Convert masks to BGR for saving
            pred_display = cv2.cvtColor(pred_mask_orig, cv2.COLOR_GRAY2BGR)
            gt_display = cv2.cvtColor(gt_mask_orig, cv2.COLOR_GRAY2BGR)
            
            out_base = os.path.join(output_dir, f"vis_{name_no_ext}")
            
            cv2.imwrite(f"{out_base}_1_input.png", input_display)
            cv2.imwrite(f"{out_base}_2_gt.png", gt_display)
            cv2.imwrite(f"{out_base}_3_pred.png", pred_display)
            
            print(f"Saved set for {name_no_ext}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to .pth model file')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--num', type=int, default=5, help='Number of images to visualize')
    args = parser.parse_args()
    
    visualize(args.model, args.data, args.num)
