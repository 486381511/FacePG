# -------------------- 导入依赖库 --------------------
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

# 中文输出兼容
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# -------------------- 基本配置参数 --------------------
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR  = "./TrainDatasets"
BATCH_SIZE   = 32
EPOCHS       = 200
LR           = 1e-3
EMBEDDING_SZ = 512
MARGIN_M     = 0.50
SCALE_S      = 64.0
PRETRAINED_PATH = "weights/resnet50-11ad3fa6.pth"

# -------------------- 自定义数据集类 --------------------
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            cls_dir = os.path.join(root_dir, cls)
            for file in os.listdir(cls_dir):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, file), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, label

# -------------------- ArcMarginProduct 层 --------------------
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=SCALE_S, m=MARGIN_M):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.th = torch.cos(torch.tensor(3.14159265 - m))
        self.mm = torch.sin(torch.tensor(3.14159265 - m)) * m

    def forward(self, embeddings, labels):
        embeddings = nn.functional.normalize(embeddings)
        weights = nn.functional.normalize(self.weight)

        cosine = nn.functional.linear(embeddings, weights)
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine, device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

# -------------------- ResNet50 特征提取网络 --------------------
class ResNetEmbedding(nn.Module):
    def __init__(self, embedding_size=EMBEDDING_SZ):
        super().__init__()
        resnet = resnet50(weights=None)

        if not os.path.exists(PRETRAINED_PATH):
            raise FileNotFoundError(f"未找到预训练模型: {PRETRAINED_PATH}")
        state_dict = torch.load(PRETRAINED_PATH, map_location=device)
        resnet.load_state_dict(state_dict)

        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.bn(x)
        x = nn.functional.normalize(x)
        return x

# -------------------- 主训练流程 --------------------
def main():
    transform = transforms.Compose([
        transforms.Resize((112, 112), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    dataset = FaceDataset(DATASET_DIR, transform)
    num_classes = len(dataset.class_to_idx)
    if num_classes < 2:
        raise ValueError("至少需要两个类别")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    backbone = ResNetEmbedding().to(device)
    margin_fc = ArcMarginProduct(EMBEDDING_SZ, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(backbone.parameters()) + list(margin_fc.parameters()),
        lr=LR,
        weight_decay=5e-4
    )

    best_loss = float('inf')
    best_model_path = "weights/arcface_model_best.pth"

    print(f"Start training on {device} with {num_classes} identities…")

    for epoch in range(1, EPOCHS + 1):
        backbone.train()
        margin_fc.train()
        total_loss = 0

        for imgs, labels in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            embeddings = backbone(imgs)
            logits = margin_fc(embeddings, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch}/{EPOCHS}]  Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "backbone_state": backbone.state_dict(),
                "margin_state": margin_fc.state_dict(),
                "class_to_idx": dataset.class_to_idx
            }, best_model_path)
            print(f"新最佳模型已保存，Loss: {best_loss:.4f}")

    torch.save({
        "backbone_state": backbone.state_dict(),
        "margin_state": margin_fc.state_dict(),
        "class_to_idx": dataset.class_to_idx
    }, "weights/arcface_model.pth")
    print("模型训练完成，已保存为 arcface_model.pth")

# -------------------- 程序入口 --------------------
if __name__ == "__main__":
    main()
