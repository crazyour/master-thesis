import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm  # 导入 tqdm

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载训练和验证数据集
train_dataset = datasets.ImageFolder(root='/home/luanma12/recognition_10/data/train', transform=transform)
val_dataset = datasets.ImageFolder(root='/home/luanma12/recognition_10/data/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 学习率列表
learning_rates = [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.003, 0.005, 0.007, 
                  0.0002, 0.0004, 0.0006, 0.0008, 0.002, 0.004, 0.006, 0.008]

# 训练和验证模型
num_epochs = 10

# 设置模型索引
model_index = 1

for lr in learning_rates:
    # 加载新的 ResNet50 模型
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # 修改为十分类
    model = model.to(device)
    
    # 使用不同的学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nTraining with learning rate: {lr}")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", dynamic_ncols=True, leave=False)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.set_postfix(avg_loss=f"{running_loss / (train_bar.n + 1):.4f}")  # 平均损失
        
        avg_train_loss = running_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", dynamic_ncols=True, leave=False)
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_bar.set_postfix(avg_loss=f"{val_loss / (val_bar.n + 1):.4f}")  # 平均损失
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        
        # 打印当前 epoch 的结果
        print(f'Epoch [{epoch+1}/{num_epochs}] | '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | '
              f'Val Accuracy: {accuracy:.4f}')
    
    # 保存模型
    torch.save(model.state_dict(), f'/home/luanma12/recognition_10/evolutionary/model/rl_model/model_{model_index}.pth')
    print(f"Model saved as 'model_{model_index}.pth'")
    
    # 更新索引
    model_index += 1
