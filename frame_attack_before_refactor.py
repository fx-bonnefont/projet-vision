import os
import numpy as np
import cv2
import random
import argparse

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import torchvision.transforms.functional as TF
import mlflow

import kornia.morphology as K
from transformers import loss

from dataset import CachedFeaturesDataset, PreTiledDataset, mask_to_centers, DOTA_MEAN, DOTA_STD
from inference import draw_centers, load_checkpoint, predict_sliding_window, find_peaks, compute_metrics_at_threshold, match_centers, get_gaussian_weight_map
from loss import CenterLoss
from train import generate_run_id, segdino_collate 
from codenames import generate_codename
from extract_features import extract_and_save_features, get_cache_dir
from inference import find_peaks, match_centers
from loss import CenterLoss
from model import DecoderOnly


RANDOM_STATE = 1619
TILE_SIZE = 512
STRIDE = 384
THRESHOLD = 0.3
MATCH_RADIUS = 20.0
BATCH_SIZE = 16
DATA_DIR = "segdata/DOTA/DOTA_PLANES_TILED"
ATTACK_DIR = "data/attack"
NUM_WORKERS = 8
VALIDATION_RATIO = 0.3


class PatchAttack():
    def __init__(self, param, attack = None):
        self.device = "mps" if torch.backends.mps.is_available() else  "cuda" if torch.cuda.is_available() else "cpu"
        self.n_epoch = param.e
        self.learning_rate = param.l
        self.batch_size = param.b
        self.batch_repetion = param.batch_repetition
        self.early_stop = param.early_stop
        self.interesting_images = ["P0023_obj_27.png"]
        self.instantiate_dataset()
        self.id_load = param.attack_id
        self.skip_validation = param.skip_validation # defalut True
        
        self.mlflow_client = mlflow.tracking.MlflowClient()
        
        self.model = load_checkpoint(param.c, self.device)
        self.model.eval()
        for p in self.model.parameters():
                p.requires_grad_(False)
        
        self.kernel = torch.ones((param.thickness, param.thickness), device=self.device)
    
        self.cefunction = - CenterLoss().to(self.device)
        with torch.no_grad:
            if attack is None:
                thickness = param.thickness
                self.attack = attack = torch.nn.Parameter(torch.rand([1,3,thickness,400]))
            else: #restart from a previous attack
                self.attack = self.load_attack(self.id_load)
                _, self.thickness, self.L = self.attack.shape
                
            self.optimizer = torch.optim.AdamW([self.attack], lr=self.learning_rate)
            self.norm_min = TF.normalize(torch.zeros_like(attack), DOTA_MEAN, DOTA_STD)
            self.norm_max = TF.normalize(torch.ones_like(attack), DOTA_MEAN, DOTA_STD)




    def validate_attack(self, attack, validation_loader):
        model = self.model
        model.eval()
        self.validation_loss = 0.0
        max_prediction = [] 
        max_prediction_attacked = []
        
        with torch.inference_mode():
            for imgs_t, _, metas, masks_t in tqdm(validation_loader, desc="validation", leave=False):
                imgs_t = imgs_t.to(self.device, non_blocking=True)
                masks_t = masks_t.to(self.device, non_blocking=True)
                pred_t = torch.sigmoid(model(imgs_t))
                imgs_attacked_t = self.apply_attack(imgs_t, metas, self.thickness, attack)
                pred_attacked_t = torch.sigmoid(model(imgs_attacked_t))
                loss = -self.cefunction(pred_t, pred_attacked_t)
                self.validation_loss += loss.item()
            max_prediction.append(pred_t.max().item())
            max_prediction_attacked.append(pred_attacked_t.max().item())
            
        max_prediction = np.array(max_prediction)
        max_prediction_attacked = np.array(max_prediction_attacked)
                    
        return self.validation_loss, max_prediction, max_prediction_attacked

    

    def instantiate_dataset(self):
        
        # Prepare datasets
        dataset = PreTiledDataset(DATA_DIR,"train",return_empty=True)
        dataset_test = PreTiledDataset(DATA_DIR,"test", return_empty=True)
        
        n_val = int(len(dataset) * VALIDATION_RATIO)
        n_train = len(dataset) - n_val
        dataset_train, dataset_validation = random_split(dataset, [n_train, n_val], generator=g)
            
            
        self.train_loader = DataLoader(
            dataset_train,
            batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
            drop_last=True, collate_fn=segdino_collate, persistent_workers=True
        )

        self.validation_loader = DataLoader(
            dataset_validation,
            batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
            drop_last=True, collate_fn=segdino_collate, persistent_workers=True
        )
        
        self.test_loader = DataLoader(dataset_test,
            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
            collate_fn=segdino_collate, persistent_workers=True
        )
        

    def save_attack(self):
        torch.save(self.attack.detach().cpu(), self.attack_path)
        mlflow.log_artifact(self.attack_path)
        return True
    
    def load_attack(self, run_id):
        artifact = [f.path for f in self.mlflow_client.list_artifacts(run_id)]
        path = self.mlflow_client.download_artifacts(run_id, artifact[0])
        attack = torch.load(path, map_location=self.device).to(self.device)
        return attack
    
    
        return F.interpolate(strip, size=(t, needed_w), mode=mode, align_corners=False)

    # Helper: resize a horizontal strip [B,3,t,L] to [B,3,t,needed_w]
    def resize_h(self, strip, needed_w):
        return F.interpolate(strip, size=(t, needed_w), mode=mode, align_corners=False)

    # Helper: resize a vertical strip derived from same attack to [B,3,needed_h,t]
    # Helper: resize a vertical strip derived from same attack to [B,3,needed_h,t]
    def resize_v(strip, needed_h):
        # strip: [B,3,t,L] -> [B,3,L,t] then interpolate to (needed_h, t)
        strip_v = strip.permute(0, 1, 3, 2)
        out = F.interpolate(strip_v, size=(needed_h, t), mode=mode, align_corners=False)
        return out  # [B,3,needed_h,t]
        
        
    def put_attack_square_border(self, img,  cx, cy, r):
        """
        img:    [B, 3, H, W]
        attack: [1 or B, 3, thickness, L]   (e.g. L=400). Used for all 4 sides via interpolation.
        (cx, cy): center in (row, col) coordinates
        r: half-side of the square (side = 2r)
        thickness: border thickness in pixels
        """
        C, H, W = img.shape
        t = self.thickness
        attack_left = self.attack[:, :, :, :TILE_SIZE]
        attack_top = self.attack[:, :, :, TILE_SIZE:TILE_SIZE*2]
        attack_right = self.attack[:, :, :, TILE_SIZE*2:TILE_SIZE*3]
        attack_bottom = self.attack[:, :, :, TILE_SIZE*3:]
        
        # Square interior (clamped)
        x0 = max(cx - r, 0)
        x1 = min(cx + r, H)
        y0 = max(cy - r, 0)
        y1 = min(cy + r, W)

        # TOP border (outside): rows [x0 - t, x0), cols [y0, y1)
        xt0, xt1 = max(x0 - t, 0), x0
        if xt1 > xt0 and y1 > y0:
            needed_h = xt1 - xt0
            needed_w = y1 - y0
            top = self.resize_h(attack_top, needed_w)          # [B,3,t,needed_w]
            top = top[:, :, :needed_h, :]            # crop if clamped near image top
            img[:, xt0:xt1, y0:y1] = top

        # BOTTOM border (outside): rows [x1, x1 + t), cols [y0, y1)
        xb0, xb1 = x1, min(x1 + t, H)
        if xb1 > xb0 and y1 > y0:
            needed_h = xb1 - xb0
            needed_w = y1 - y0
            bottom = self.resize_h(attack_bottom, needed_w)
            bottom = bottom[:, :, :needed_h, :]
            img[:, xb0:xb1, y0:y1] = bottom

        # LEFT border (outside): rows [x0, x1), cols [y0 - t, y0)
        yl0, yl1 = max(y0 - t, 0), y0
        if yl1 > yl0 and x1 > x0:
            needed_h = x1 - x0
            needed_w = yl1 - yl0  # this is thickness after clamping
            left = self.resize_v(attack_left, needed_h)        # [B,3,needed_h,t]
            left = left[:, :, :, :needed_w]          # crop if clamped near image left
            img[:, x0:x1, yl0:yl1] = left

        # RIGHT border (outside): rows [x0, x1), cols [y1, y1 + t)
        yr0, yr1 = y1, min(y1 + t, W)
        if yr1 > yr0 and x1 > x0:
            needed_h = x1 - x0
            needed_w = yr1 - yr0
            right = self.resize_v(attack_right, needed_h)
            right = right[:, :, :, :needed_w]
            img[:, x0:x1, yr0:yr1] = right

            return img
    
    
    def apply_attack(self, imgs_t, metas, thickness, attack):
        attacked_imgs_t = imgs_t.clone()
        for i in range(imgs_t.shape[0]):
            meta = metas[i]
            img = attacked_imgs_t[i]
            for j in range(meta["num_objects"]):
                cx, cy = meta["centers"][j]
                area = meta["areas"][j]
                edge_size = np.floor(np.sqrt(area))
                r = int(np.floor(edge_size/2))
                img = self.put_attack_square_border(img, attack, cx, cy, r, thickness, mode="bilinear")
            attacked_imgs_t[i,:,:,:] = img   
        return attacked_imgs_t
    
    def train_attack(self):# Init the attack
        with mlflow.start_run() as run:
            stop = False
            self.run_id = run.info.run_id
            self.attack_path = f"runs/{self.run_id}_attack.pt"

            mlflow.log_param("n_epoch", self.n_epoch)
            mlflow.log_param("learning_rate", self.learning_rate)
            
            # epoch loop
            global_step = 0
            for epoch in tqdm(range(1, self.n_epoch+1), desc=self.run):
                # loop throught the batch
                training_loss = 0.0
                batch_count = 0
                
                for imgs_t, _, metas, masks_t in tqdm(self.train_loader, desc="train", leave=False):
                    B, C, H, W = imgs_t.shape
                    imgs_t = imgs_t.to(self.device, non_blocking=True)
                    masks_t = masks_t.to(self.device, non_blocking=True)
                    with torch.no_grad():
                        pred_t = torch.sigmoid(self.model(imgs_t))
                        
                    for _ in range(self.batch_repetion):
                        self.optimizer.zero_grad(set_to_none=True)
                        imgs_attacked_t = self.apply_attack(imgs_t, metas, self.thickness, self.attack)
                        pred_attacked_t = torch.sigmoid(self.model(imgs_attacked_t))
                        
                        loss = - self.cefunction(pred_t, pred_attacked_t)
                        loss.backward()
                        self.optimizer.step()
                                        
                        with torch.no_grad():
                            self.attack.clamp_(self.norm_min, self.norm_max)  # keeps it in image range, no new graph
                        
                        training_loss += loss.item()
                        batch_count += 1
                        global_step += 1
                        mlflow.log_metric("training_loss", loss.item(), step=global_step)
                    
                    if global_step>self.early_stop:
                        stop = True
                        break
                    
                    batch_count += 1
                    
                    tqdm.write(f"BATCH {batch_count:3d} | Average training Loss: {training_loss/self.batch_repetion:.4f}")
                    mlflow.log_metric("Batch Average Training_loss", training_loss/self.batch_repetion, step=global_step)
                    if batch_count % 10 == 0:
                        self.save_attack()
                
                if stop:
                    break
                
                if self.skip_validation is False:
                    validation_loss, max_prediction, max_prediction_attacked = self.validate_attack(self.attack)
                    mlflow.log_metric("validation_loss", validation_loss, step=global_step) 
                    mlflow.log_metric("max_prediction_unattacked", max_prediction.mean(), step=global_step)
                    mlflow.log_metric("max_prediction_attacked", max_prediction_attacked.mean(), step=global_step)
                    tqdm.write(f"EPOCH {epoch:3d} | train: {training_loss:.4f} | validation_loss: {validation_loss} | Average Max prediction original vs attacked: {max_prediction.mean()} vs {max_prediction_attacked.mean()}")

                
            # save
            self.save_attack()
            
        return self.attack, self.run_id
    
    def x_to_torch(self, x):
        x_t = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0
        x_t = TF.normalize(x_t, DOTA_MEAN, DOTA_STD)
        x_t = x_t.unsqueeze(0).to(self.device)
        return x_t

    def numpyfy_attack_results(self, x_attacked, y_attacked):
        mean = torch.tensor(DOTA_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(DOTA_STD, device=self.device).view(1, 3, 1, 1)

        x_attacked_un = (x_attacked * std + mean)
        x_min, x_max = (x_attacked_un*(1-patch_t)).min(), (x_attacked_un*(1-patch_t)).max()
        x_attacked_un = (x_attacked_un + x_min)/(x_min+ x_max)*255
        img_attacked = x_attacked_un.cpu().detach().numpy().squeeze().transpose(1,2,0)
        prediction_attacked = y_attacked.cpu().detach().numpy().squeeze()
        
        return img_attacked, prediction_attacked

    def show_attack_results(self, img, meta, attack):
        weight_map_t = get_gaussian_weight_map(TILE_SIZE, self.device)
        img_t = self.x_to_torch(img)
        
        self.model.eval()
        with torch.no_grad():
            probality_map_unattacked_t = torch.sigmoid(self.model(img_t))*weight_map_t
            img_attacked_t = self.apply_attack(img_t, meta, self.thickness, attack)
            probality_map_attacked_t = torch.sigmoid(self.model(img_attacked_t))*weight_map_t
                    
            loss = - self.cefunction(probality_map_unattacked_t, probality_map_attacked_t)
        
        loss_test = loss.item()
        
        img_unattacked, probality_map_unattacked = self.numpyfy_attack_results(img_t, probality_map_unattacked_t)
        img_attacked, probality_map_attacked = self.numpyfy_attack_results(img_attacked_t, probality_map_attacked_t)
        p_max_unattacked = probality_map_unattacked.max()
        p_max_attacked = probality_map_attacked.max()
        
        centers_unattacked = find_peaks(probality_map_unattacked, threshold=THRESHOLD)
        img_unattacked_cross = draw_centers(img_unattacked, centers_unattacked, (0, 165, 255), radius=radius, thickness=thickness)

        centers_attacked = find_peaks(probality_map_attacked, threshold=THRESHOLD)
        img_attacked_cross = draw_centers(img_attacked, centers_attacked, (0, 165, 255), radius=radius, thickness=thickness)

        ## Normalization
        # Make x and vis_pred float [0,1] too (if they are uint8 or 0..255 float)
        img_unattacked_cross = img_unattacked_cross.astype(np.float32) / 255.0
        img_attacked_cross = img_attacked_cross.astype(np.float32) / 255.0

        # Normalize prob_map to [0,1]
        probality_map_unattacked = probality_map_unattacked.astype(np.float32)
        pmin0, pmax0 = probality_map_unattacked.min(), probality_map_unattacked.max()
        probality_map_unattacked = (probality_map_unattacked - pmin0) / (pmax0 - pmin0 + 1e-8)
        # Colormap returns float in [0,1]
        probality_map_unattacked = plt.get_cmap("magma")(probality_map_unattacked)[..., :3]
        
        # Normalize prob_map to [0,1]
        probality_map_attacked = probality_map_attacked.astype(np.float32)
        pmin, pmax = probality_map_attacked.min(), probality_map_attacked.max()
        probality_map_attacked = (probality_map_attacked - pmin) / (pmax - pmin + 1e-8)
        # Colormap returns float in [0,1]
        probality_map_attacked = plt.get_cmap("magma")(probality_map_attacked)[..., :3]
        
        fig, ax = plt.subplots(2,2, figsize=(12,12))

        plt.title(f"Comparison of Detection with and Without Attack | Loss ={loss_test:.2f}")
        ax[0,0].imshow(img_unattacked_cross)
        ax[0,0].set_title(f"Original Image and Prediction | Max = {p_max_unattacked:.2f}")
        ax[0,1].imshow(probality_map_unattacked)
        ax[0,1].set_title("Original Prob")
        ax[1,0].imshow(img_attacked_cross)
        ax[1,0].set_title(f"Attacked Image and Prediction | Max = {p_max_attacked:.2f}")
        ax[1,1].imshow(probality_map_attacked)
        ax[1,1].set_title("Attacked  Prob")

        for ax in fig.axes:
            ax.axis("off")
        plt.show()
        
    def plot_interesting_images(self, run_id = None):
        for img_name in self.interesting_images:
            img_path = os.path.join(DATA_DIR, "test/images", img_name)
            mask_path = os.path.join(DATA_DIR, "test/masks", img_name)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            centers, areas = mask_to_centers(mask, return_areas=True)
            meta = [{
                "centers": centers,
                "areas": areas,
                "num_objects": len(centers)
            }]
            
            if run_id is not None:
                self.show_attack_results(img, mask, meta, self.attack)
            else:
                attack = self.load_attack(run_id)
                self.show_attack_results(img, mask, meta, attack)
                
        


def main():
    parser = argparse.ArgumentParser(description="Evaluate an adversarial patch around a plane on SegDino")
    parser.add_argument("-c", required=True, help="checkpoint path")
    parser.add_argument("--patch", required=True, help="path to saved patch .pt")
    parser.add_argument("-b", type=int, default=BATCH_SIZE, help="batch size")
    parser.add_argument("--thickness", type=int, default=24, help="patch size")
    parser.add_argument("-l", type=int, default=0.5, help="learning rate")
    parser.add_argument("-e", type=int, default=2, help="number of epochs")
    args = parser.parse_args()
    
    print("Initializing attack...")
    patch_attack = PatchAttack(args)
    
    print("Training the attack...")
    _, run_id = patch_attack.train_attack()
    print(f"Attack trained and saved with run_id: {run_id}")
    
    print("visualizing attack...")
    #patch_attack.visualize()
        