import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.models as models
from PIL import UnidentifiedImageError

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
def evaluate(model_path):
    # 数据预处理，与训练时保持一致
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 自定义数据集类来处理损坏的图像文件
    class VerifiedImageFolder(datasets.ImageFolder):
        def __getitem__(self, index):
            path, target = self.samples[index]
            try:
                img = self.loader(path)
                if self.transform is not None:
                    img = self.transform(img)
                return img, target, path  # 添加路径返回
            except (UnidentifiedImageError, OSError):
                print(f"Skipping corrupted image: {path}")
                return None  # 返回 None 以跳过此项

    # 加载测试数据集
    test_dataset = VerifiedImageFolder(root='/home/luanma12/recognition_10/data/train', transform=transform)

    # 创建自定义 DataLoader 以过滤掉 None
    def collate_fn(batch):
        # 过滤掉 None 条目
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 加载训练好的模型
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # 二分类输出
    model = model.to(device)

    # 加载模型参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置模型为评估模式

    # 定义损失函数（可选）
    criterion = nn.CrossEntropyLoss()

    # 测试模型性能
    correct = 0
    total = 0
    test_loss = 0.0
    incorrect_samples = []  # 记录预测错误的样本

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 记录预测错误的图片和路径
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    incorrect_samples.append((paths[i], predicted[i].item(), labels[i].item()))

    # 打印预测错误的图片路径及其分类
    print("Incorrect Predictions:")
    for path, pred, true_label in incorrect_samples:
        print(f"Path: {path}, Predicted: {pred}, True: {true_label}")

    # 计算测试集的准确率和平均损失
    accuracy = correct / total
    return accuracy

model_path="/home/luanma12/recognition_10/evolutionary/model/rl_model/model_0.pth"
print(evaluate(model_path))
