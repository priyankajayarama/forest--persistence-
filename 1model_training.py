# =============================================================================
# WEEK 1 — MODEL IMPROVEMENTS & TRAINING
# Forest Persistence Segmentation Capstone
# Cells: Run each block separately in Colab
# =============================================================================


# =============================================================================
# CELL 1 — INSTALL LIBRARIES (restart runtime after this finishes)
# =============================================================================
# !pip install segmentation-models-pytorch mlflow albumentations \
#              opencv-python-headless pillow -q


# =============================================================================
# CELL 2 — CREATE SYNTHETIC TRAINING DATA
# =============================================================================
import os
import numpy as np
from PIL import Image

os.makedirs("data/images", exist_ok=True)
os.makedirs("data/masks",  exist_ok=True)

def make_forest_tile(idx, size=256):
    np.random.seed(idx)
    r = np.clip(np.random.randint(30,  100, (size, size)), 0, 255).astype(np.uint8)
    g = np.clip(np.random.randint(60,  160, (size, size)), 0, 255).astype(np.uint8)
    b = np.clip(np.random.randint(20,   80, (size, size)), 0, 255).astype(np.uint8)
    image = np.stack([r, g, b], axis=2).astype(np.int32)

    mask = np.zeros((size, size), dtype=np.uint8)
    for _ in range(np.random.randint(3, 8)):
        cx  = np.random.randint(30, size - 30)
        cy  = np.random.randint(30, size - 30)
        rad = np.random.randint(20, 70)
        yy, xx = np.ogrid[:size, :size]
        circle = (xx - cx)**2 + (yy - cy)**2 < rad**2
        mask[circle] = 255
        image[:, :, 1] = np.where(circle, np.clip(image[:, :, 1] + 60, 0, 255), image[:, :, 1])
        image[:, :, 0] = np.where(circle, np.clip(image[:, :, 0] - 20, 0, 255), image[:, :, 0])

    return image.astype(np.uint8), mask

NUM_TILES = 120
for i in range(NUM_TILES):
    img, mask = make_forest_tile(i)
    name = f"tile_{i:04d}.png"
    Image.fromarray(img).save(f"data/images/{name}")
    Image.fromarray(mask).save(f"data/masks/{name}")

print(f"Created {NUM_TILES} tiles")
print(f"Images: {len(os.listdir('data/images'))}")
print(f"Masks:  {len(os.listdir('data/masks'))}")


# =============================================================================
# CELL 3 — MODEL DEFINITION (U-Net + ResNet34 + Combined Loss)
# =============================================================================
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def get_model():
    """U-Net with pretrained ResNet34 encoder."""
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )

class CombinedLoss(nn.Module):
    """BCE + Dice loss — penalises missed forest pixels harder than plain BCE."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target, smooth=1.0):
        pred      = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        true_flat = target.view(-1)
        intersection = (pred_flat * true_flat).sum()
        return 1 - (2 * intersection + smooth) / (
            pred_flat.sum() + true_flat.sum() + smooth
        )

    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice_loss(pred, target)

print("Model and loss defined.")


# =============================================================================
# CELL 4 — DATASET WITH AUGMENTATION
# =============================================================================
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
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

print("Dataset defined.")


# =============================================================================
# CELL 5 — TRAIN THE MODEL
# =============================================================================
import mlflow
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {DEVICE}")

train_dataset = ForestDataset("data/train/images", "data/train/masks", is_train=True)
val_dataset   = ForestDataset("data/val/images",   "data/val/masks",   is_train=False)
train_loader  = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=4, shuffle=False)

model     = get_model().to(DEVICE)
criterion = CombinedLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

NUM_EPOCHS    = 50
best_val_loss = float("inf")
train_losses, val_losses = [], []

mlflow.set_experiment("forest_segmentation")

with mlflow.start_run():
    mlflow.log_params({"encoder": "resnet34", "loss": "bce+dice",
                       "epochs": NUM_EPOCHS, "batch_size": 4, "lr": 1e-4})

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                val_loss += criterion(model(imgs), masks).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"  -> Best model saved (val={best_val_loss:.4f})")

    mlflow.log_artifact("models/best_model.pth")

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train")
plt.plot(val_losses,   label="Val")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Training progress — forest segmentation")
plt.legend(); plt.grid(True)
plt.savefig("outputs/loss_curve.png")
plt.show()
print(f"Training complete. Best val loss: {best_val_loss:.4f}")


# =============================================================================
# CELL 6 — EVALUATE ON TEST SET (IoU, Precision, Recall)
# =============================================================================
test_dataset = ForestDataset("data/test/images", "data/test/masks", is_train=False)
test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False)

model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
model.eval()

def calculate_metrics(pred_mask, true_mask, threshold=0.5):
    pred_bin = (pred_mask >= threshold).astype(np.uint8)
    true_bin = (true_mask >= threshold).astype(np.uint8)
    tp    = (pred_bin & true_bin).sum()
    fp    = (pred_bin & ~true_bin.astype(bool)).sum()
    fn    = (~pred_bin.astype(bool) & true_bin).sum()
    union = (pred_bin | true_bin).sum()
    iou       = tp / (union + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    return iou, precision, recall

all_iou, all_prec, all_rec = [], [], []
with torch.no_grad():
    for imgs, masks in test_loader:
        imgs = imgs.to(DEVICE)
        preds = torch.sigmoid(model(imgs)).cpu().numpy()
        masks = masks.numpy()
        for pred, mask in zip(preds, masks):
            iou, prec, rec = calculate_metrics(pred[0], mask[0])
            all_iou.append(iou)
            all_prec.append(prec)
            all_rec.append(rec)

print("Test Results:")
print(f"  IoU:       {np.mean(all_iou):.4f}")
print(f"  Precision: {np.mean(all_prec):.4f}")
print(f"  Recall:    {np.mean(all_rec):.4f}")


# =============================================================================
# CELL 7 — VISUALISE PREDICTIONS
# =============================================================================
imgs, masks = next(iter(test_loader))
model.eval()
with torch.no_grad():
    preds = torch.sigmoid(model(imgs.to(DEVICE))).cpu()

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

for i in range(4):
    img_np  = (imgs[i].permute(1,2,0).numpy() * std + mean).clip(0, 1)
    mask_np = masks[i].squeeze().numpy()
    pred_np = preds[i].squeeze().numpy()
    axes[0, i].imshow(img_np);                 axes[0, i].set_title("Satellite image"); axes[0, i].axis("off")
    axes[1, i].imshow(mask_np, cmap="Greens"); axes[1, i].set_title("True mask");       axes[1, i].axis("off")
    axes[2, i].imshow(pred_np, cmap="Greens"); axes[2, i].set_title("Predicted mask");  axes[2, i].axis("off")

plt.suptitle("Model predictions on test set", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/predictions.png")
plt.show()
print("Week 1 complete! Model trained and evaluated.")
print("Saved: models/best_model.pth")
print("Saved: outputs/loss_curve.png")
print("Saved: outputs/predictions.png")
