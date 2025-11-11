import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------
# Settings
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "saved_models/best_resnet18_catdog.pth"  # dein trainiertes Modell
IMG_SIZE = 128

# -----------------------
# Transform für Bilder
# -----------------------
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# -----------------------
# Modell laden
# -----------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 1)  # binary output
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# -----------------------
# Funktion für Vorhersage
# -----------------------
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = val_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()
        label = "dog" if prob > 0.5 else "cat"
    print(f"{os.path.basename(img_path)} --> Prediction: {label} (prob_dog={prob:.2f})")
    plt.imshow(np.array(img))
    plt.title(f"{label} {prob:.2f}")
    plt.axis('off')
    plt.show()

