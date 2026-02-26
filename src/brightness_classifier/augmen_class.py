# augmen_class.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# route settings
train_dir = r"E:\aihub_lowlight\balanced_scene_split\brightness_stage_split\train"
val_dir   = r"E:\aihub_lowlight\balanced_scene_split\brightness_stage_split\val"

# Transform
tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Dataset / Loader
train_dataset = ImageFolder(train_dir, transform=tf)
val_dataset   = ImageFolder(val_dir, transform=tf)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("클래스 인덱스:", train_dataset.class_to_idx)
print("Train 데이터 개수:", len(train_dataset))
print("Val 데이터 개수:", len(val_dataset))

# 모델 구성
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 3)  
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 학습
for epoch in range(10):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch+1}] done.")

# 저장
save_path = r"E:\aihub_lowlight\brightness_stage_classifier.pt"
torch.save(model.state_dict(), save_path)
print("모델 저장 완료:", save_path)


