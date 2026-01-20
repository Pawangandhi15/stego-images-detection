# =========================================================
# Stego Images Dataset (marcozuppelli/stegoimagesdataset)
# FULL LOAD + EDA + Preprocessing (Colab Ready)
# =========================================================

# -----------------------------
# 0. Install Dependencies
# -----------------------------
!pip install -q kagglehub opencv-python seaborn scikit-learn matplotlib pandas tqdm

# -----------------------------
# 1. Import Libraries
# -----------------------------
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8,5)

# -----------------------------
# 2. Download Dataset (NO API)
# -----------------------------
DATA_DIR = kagglehub.dataset_download(
    "marcozuppelli/stegoimagesdataset"
)

print("Dataset downloaded at:", DATA_DIR)
print("Top-level contents:", os.listdir(DATA_DIR))

# -----------------------------
# 3. Inspect Folder Contents
# -----------------------------
all_files = []
for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            all_files.append(os.path.join(root, f))

print(f"Found {len(all_files)} images total")
print("Sample paths:", all_files[:5])

# -----------------------------
# 4. Load Images & Infer Labels
# -----------------------------
def infer_label_from_filename(fname):
    """
    Guess label based on filename:
    - If 'stego' present in file name → stego class = 1
    - Otherwise → cover class = 0
    (Update logic if your dataset uses a different scheme)
    """
    base = os.path.basename(fname).lower()
    return 1 if "stego" in base else 0

images, labels = [], []
IMG_SIZE = (256, 256)

for img_path in tqdm(all_files):
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        images.append(img)
        labels.append(infer_label_from_filename(img_path))

images = np.array(images)
labels = np.array(labels)
print("Loaded images:", images.shape)
print("Label counts:", np.bincount(labels))

# -----------------------------
# 5. Visualize Samples
# -----------------------------
def show_samples(images, labels, n=5):
    plt.figure(figsize=(12,4))
    for i in range(n):
        idx = np.random.randint(len(images))
        plt.subplot(1, n, i+1)
        plt.imshow(images[idx])
        plt.title("Stego" if labels[idx] == 1 else "Cover")
        plt.axis("off")
    plt.show()

show_samples(images, labels)

# -----------------------------
# 6. Pixel Intensity Distribution
# -----------------------------
pixels = images.reshape(-1, 3)

for i, c in enumerate(["Red", "Green", "Blue"]):
    sns.kdeplot(pixels[:, i], label=c)

plt.title("Pixel Intensity Distribution")
plt.legend()
plt.show()

# -----------------------------
# 7. Feature Extraction
# -----------------------------
def extract_features(img):
    feats = []
    for c in range(3):
        channel = img[:, :, c].flatten()
        feats.extend([
            channel.mean(),
            channel.std(),
            channel.min(),
            channel.max()
        ])
    return feats

feature_names = [
    "R_mean","R_std","R_min","R_max",
    "G_mean","G_std","G_min","G_max",
    "B_mean","B_std","B_min","B_max"
]

X = np.array([extract_features(img) for img in tqdm(images)])
y = labels

df = pd.DataFrame(X, columns=feature_names)
df["label"] = y

print(f"Feature DataFrame shape: {df.shape}")
display(df.head())

# -----------------------------
# 8. Correlation Heatmap
# -----------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# -----------------------------
# 9. Normalize / Standardize
# -----------------------------
X_features = df.drop("label", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# -----------------------------
# 10. Train / Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -----------------------------
# 11. Class Weights
# -----------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
print("Class weights:", dict(enumerate(class_weights)))

# -----------------------------
# 12. PCA Visualization
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", alpha=0.6)
plt.title("PCA Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# -----------------------------
# 13. Save Data
# -----------------------------
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("✅ Preprocessing complete!")
