# =============================================================================
# WEEK 2 — DATA PIPELINE
# Forest Persistence Segmentation Capstone
# Prerequisite: Week 1 complete, models/best_model.pth exists
# =============================================================================


# =============================================================================
# CELL 1 — INSTALL WEEK 2 LIBRARIES
# =============================================================================
# !pip install dvc -q


# =============================================================================
# CELL 2 — AUTOMATED TILING PIPELINE
# Chops any large satellite image into 256x256 training tiles
# =============================================================================
import os
import numpy as np
from PIL import Image
import shutil

TILE_SIZE  = 256
OVERLAP    = 32       # tiles overlap so the model sees forest edges better
MIN_FOREST = 0.05     # skip tiles with less than 5% forest pixels

os.makedirs("data/images", exist_ok=True)
os.makedirs("data/masks",  exist_ok=True)

def is_valid_tile(mask_arr, min_forest=MIN_FOREST):
    """Return True only if tile has enough forest to be useful for training."""
    return (mask_arr > 127).sum() / mask_arr.size >= min_forest

def tile_image_and_mask(img_path, mask_path):
    """Chop one image+mask pair into overlapping 256x256 tiles."""
    img  = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    W, H = img.size
    step = TILE_SIZE - OVERLAP
    tiles = []
    for y in range(0, H - TILE_SIZE + 1, step):
        for x in range(0, W - TILE_SIZE + 1, step):
            box       = (x, y, x + TILE_SIZE, y + TILE_SIZE)
            tile_img  = img.crop(box)
            tile_mask = mask.crop(box)
            if is_valid_tile(np.array(tile_mask)):
                tiles.append((tile_img, tile_mask))
    return tiles

def process_dataset(source_dir):
    """Walk source directory, find image+mask pairs, tile them."""
    count = 0
    for root, dirs, files in os.walk(source_dir):
        for fname in sorted(files):
            if fname.endswith(".png") and "mask" not in fname.lower():
                img_path  = os.path.join(root, fname)
                mask_path = os.path.join(root, fname.replace(".png", "_mask.png"))
                if not os.path.exists(mask_path):
                    continue
                for tile_img, tile_mask in tile_image_and_mask(img_path, mask_path):
                    name = f"tile_{count:05d}.png"
                    tile_img.save(f"data/images/{name}")
                    tile_mask.save(f"data/masks/{name}")
                    count += 1
    return count

# Try processing real data first; if none found keep synthetic tiles from Week 1
total = process_dataset("raw_data")
if total == 0:
    print("No raw data found — keeping synthetic tiles from Week 1.")
    total = len(os.listdir("data/images"))

print(f"Total usable tiles: {total}")


# =============================================================================
# CELL 3 — TRAIN / VAL / TEST SPLIT (70 / 15 / 15)
# =============================================================================
import random

random.seed(42)

for split in ["train", "val", "test"]:
    os.makedirs(f"data/{split}/images", exist_ok=True)
    os.makedirs(f"data/{split}/masks",  exist_ok=True)

all_tiles = sorted(os.listdir("data/images"))
random.shuffle(all_tiles)

n         = len(all_tiles)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

splits = {
    "train": all_tiles[:train_end],
    "val":   all_tiles[train_end:val_end],
    "test":  all_tiles[val_end:]
}

for split, files in splits.items():
    for fname in files:
        shutil.copy(f"data/images/{fname}", f"data/{split}/images/{fname}")
        shutil.copy(f"data/masks/{fname}",  f"data/{split}/masks/{fname}")

print(f"Train: {len(splits['train'])} tiles")
print(f"Val:   {len(splits['val'])} tiles")
print(f"Test:  {len(splits['test'])} tiles")


# =============================================================================
# CELL 4 — VERSION DATA WITH DVC
# Tracks your dataset like Git tracks code — fully reproducible
# =============================================================================
os.makedirs(".dvc", exist_ok=True)

# Initialise git + dvc (safe to run even if already initialised)
os.system("git init")
os.system("dvc init")
os.system("dvc add data/train data/val data/test")
os.system("git add .dvc .gitignore")
os.system('git commit -m "Week 2: versioned train/val/test split"')

print("Data versioned with DVC!")
print("Anyone can reproduce this exact dataset split using: dvc pull")


# =============================================================================
# CELL 5 — VERIFY SPLITS WITH A BATCH VISUALISATION
# =============================================================================
import cv2
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Re-define dataset here so Week 2 is self-contained
def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

class ForestDataset(Dataset):
    def __init__(self, image_dir, mask_dir, is_train=True):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.images    = sorted(os.listdir(image_dir))
        self.transform = get_transforms(is_train)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir,  self.images[idx])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(mask_path, 0)
        mask  = (mask > 127).astype(np.float32)
        aug   = self.transform(image=image, mask=mask)
        return aug["image"], aug["mask"].unsqueeze(0)

train_loader = DataLoader(
    ForestDataset("data/train/images", "data/train/masks", is_train=True),
    batch_size=4, shuffle=True
)

imgs, masks = next(iter(train_loader))
print(f"Batch image shape: {imgs.shape}")
print(f"Batch mask shape:  {masks.shape}")

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i in range(4):
    img_np  = (imgs[i].permute(1,2,0).numpy() * std + mean).clip(0, 1)
    mask_np = masks[i].squeeze().numpy()
    axes[0, i].imshow(img_np);                 axes[0, i].set_title(f"Image {i}"); axes[0, i].axis("off")
    axes[1, i].imshow(mask_np, cmap="Greens"); axes[1, i].set_title(f"Mask {i}");  axes[1, i].axis("off")

plt.suptitle("Training batch — top: satellite image, bottom: forest mask")
plt.tight_layout()
plt.savefig("outputs/week2_batch_check.png")
plt.show()
print("Week 2 complete! Data pipeline ready.")
