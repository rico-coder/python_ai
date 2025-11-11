# train_catdog.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import OxfordIIITPet
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# -----------------------
# Settings
# -----------------------
DATA_ROOT = "data"
BATCH_SIZE = 32
IMG_SIZE = 128
EPOCHS = 8
LR = 1e-3
NUM_WORKERS = 0  # Windows safe; set to 2-4 if using Linux/Mac
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------
# Transforms
# -----------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# -----------------------
# Training code in main()
# -----------------------
def main():
    # -----------------------
    # Download Oxford-IIIT Pet dataset (binary cat vs dog)
    # -----------------------
    print("Downloading Oxford-IIIT Pet dataset (if not present)...")
    full_dataset = OxfordIIITPet(root=DATA_ROOT, download=True,
                                 target_types="binary-category",
                                 transform=train_transform)

    # -----------------------
    # Train/Validation split
    # -----------------------
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # Apply different transform for validation
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("Train size:", len(train_ds), "Val size:", len(val_ds))

    # -----------------------
    # Model: Pretrained ResNet18
    # -----------------------
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # single logit for binary classification
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # -----------------------
    # Training loop
    # -----------------------
    best_val_acc = 0.0
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(1, EPOCHS+1):
        # Training
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val Acc {val_acc*100:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_resnet18_catdog.pth"))
            print("Saved best model.")

    # -----------------------
    # Plot training
    # -----------------------
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.legend(); plt.title("Loss")
    plt.subplot(1,2,2)
    plt.plot([v*100 for v in val_accs], label='val_acc_percent')
    plt.legend(); plt.title("Validation Accuracy (%)")
    plt.show()

    # -----------------------
    # Helper: predict single image
    # -----------------------
    def predict_single_image(img_path):
        model.eval()
        img = Image.open(img_path).convert("RGB")
        prep = val_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(prep)
            prob = torch.sigmoid(out).item()
            label = "dog" if prob > 0.5 else "cat"
        print(f"Prediction: {label} (prob_dog={prob:.3f})")
        plt.imshow(np.array(img)); plt.title(f"{label} {prob:.2f}"); plt.axis('off'); plt.show()

    # Beispiel: uncomment und teste
    # predict_single_image("my_pics/my_dog.jpg")
    print("Training done! You can now use predict_single_image() for testing your own images.")

# -----------------------
# Main guard (Windows safe)
# -----------------------
if __name__ == "__main__":
    main()
