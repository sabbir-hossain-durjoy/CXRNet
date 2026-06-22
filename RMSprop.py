from multiprocessing import freeze_support
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

# -----------------------------
# 1️⃣ SE BLOCK
# -----------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean(dim=(2, 3))
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# -----------------------------
# 2️⃣ CBAM (FIXED)
# -----------------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = x.mean(dim=(2, 3))
        max_pool = torch.amax(x, dim=(2, 3))
        ca = self.mlp(avg_pool) + self.mlp(max_pool)
        ca = torch.sigmoid(ca).view(x.size(0), x.size(1), 1, 1)
        x = x * ca

        avg_out = x.mean(dim=1, keepdim=True)
        max_out = torch.amax(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.sigmoid(self.conv(sa))
        x = x * sa
        return x


# -----------------------------
# Helper: Unfreeze last N params
# -----------------------------
def unfreeze_last_n_params(model: nn.Module, n: int):
    for p in model.parameters():
        p.requires_grad = False
    params = list(model.parameters())
    for p in params[-n:]:
        p.requires_grad = True


# -----------------------------
# 3️⃣ FUSION MODEL
# -----------------------------
class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.densenet = models.densenet121(pretrained=True)
        self.densenet.classifier = nn.Identity()
        unfreeze_last_n_params(self.densenet, n=50)

        self.se_d = SEBlock(1024)
        self.head_d = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512),
            nn.Dropout(0.3), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256),
            nn.Dropout(0.3), nn.ReLU()
        )

        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet.classifier = nn.Identity()
        for p in self.efficientnet.parameters():
            p.requires_grad = False

        self.se_e = SEBlock(1280)
        self.head_e = nn.Sequential(
            nn.Linear(1280, 512), nn.BatchNorm1d(512),
            nn.Dropout(0.3), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256),
            nn.Dropout(0.3), nn.ReLU()
        )

        fusion_dim = 256 + 256
        self.cbam = CBAM(fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        f_d = self.densenet(x)
        f_d = f_d.unsqueeze(-1).unsqueeze(-1)
        f_d = self.se_d(f_d).squeeze(-1).squeeze(-1)
        f_d = self.head_d(f_d)

        f_e = self.efficientnet(x)
        f_e = f_e.unsqueeze(-1).unsqueeze(-1)
        f_e = self.se_e(f_e).squeeze(-1).squeeze(-1)
        f_e = self.head_e(f_e)

        f = torch.cat([f_d, f_e], dim=1)
        f = f.unsqueeze(-1).unsqueeze(-1)
        f = self.cbam(f).squeeze(-1).squeeze(-1)
        out = self.classifier(f)
        return out


# -----------------------------
# 4️⃣ MAIN FUNCTION
# -----------------------------
def main():
    input_path = r"D:\Research\Pneumonia\mama"
    output_dir = r"D:\Pre-Defense\Pneumonia\new\analysis\RMSprop_optimizer"
    os.makedirs(output_dir, exist_ok=True)
    best_model_file = os.path.join(output_dir, "best_model.pth")

    EPOCHS = 50
    BATCH_SIZE = 32
    LR = 1e-4
    IMG_SIZE = (224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(input_path, transform=transform)
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes

    total = len(full_dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    train_labels = [full_dataset.samples[i][1] for i in train_set.indices]
    class_weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = FusionModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # ✅ RMSprop optimizer instead of Adam
    optimizer = optim.RMSprop(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        alpha=0.9,
        eps=1e-8,
        weight_decay=1e-5,
        momentum=0.9
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        model.train()
        running_loss, running_corrects = 0.0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * X.size(0)
            running_corrects += (preds == y).sum().item()

        train_loss = running_loss / len(train_set)
        train_acc = running_corrects / len(train_set)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                preds = outputs.argmax(dim=1)
                val_loss += loss.item() * X.size(0)
                val_corrects += (preds == y).sum().item()

        val_loss /= len(val_set)
        val_acc = val_corrects / len(val_set)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Acc: {train_acc:.4f}, Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}, Loss: {val_loss:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_file)
            print(f"✅ Saved best model with Val Acc: {best_acc:.4f}")

    pd.DataFrame(history).to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    model.load_state_dict(torch.load(best_model_file, map_location=device))
    model.eval()

    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    total_correct = np.trace(cm)
    total_samples = np.sum(cm)
    accuracy = total_correct / total_samples

    report_text = [
        "==== Classification Report ====\n",
        report,
        "\n==== Confusion Matrix ====\n",
        np.array2string(cm, separator=', '),
        f"\n\nOverall Accuracy: {accuracy:.4f}\n"
    ]

    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    report_text.append("\n==== Per-Class Accuracy ====\n")
    for idx, cls in enumerate(class_names):
        report_text.append(f"{cls}: {per_class_acc[idx]:.4f}\n")

    perf_file = os.path.join(output_dir, "performance_metrics.txt")
    with open(perf_file, "w") as f:
        f.writelines(report_text)

    print(f"\n✅ Performance metrics saved to: {perf_file}")

    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    y_scores = np.array(y_scores)
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    freeze_support()
    main()
