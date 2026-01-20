# =============================
# 16. CNN Model for Steganalysis
# =============================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

class StegoCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.classifier(x))


# =============================
# 17. CNN Dataset Preparation
# =============================
X_img = images / 255.0
X_img = np.transpose(X_img, (0,3,1,2))  # NHWC → NCHW

X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
    X_img, labels, test_size=0.2, stratify=labels, random_state=42
)

train_ds = TensorDataset(
    torch.tensor(X_train_img, dtype=torch.float32),
    torch.tensor(y_train_img, dtype=torch.float32)
)

test_ds = TensorDataset(
    torch.tensor(X_test_img, dtype=torch.float32),
    torch.tensor(y_test_img, dtype=torch.float32)
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)



# =============================
# 19. CNN Evaluation
# =============================
model.eval()
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        outputs = model(x_batch).squeeze().cpu().numpy()

        y_prob.extend(outputs)
        y_pred.extend((outputs > 0.5).astype(int))
        y_true.extend(y_batch.numpy())

print(classification_report(y_true, y_pred))
print("CNN AUC:", roc_auc_score(y_true, y_prob))




