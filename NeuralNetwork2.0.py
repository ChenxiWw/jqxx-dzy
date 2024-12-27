import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import os

# 设置随机种子以确保实验的可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置设备为GPU（如果可用）或CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载CSV格式的CIFAR-10数据
def load_data(train_path, test_path):
    print("Loading training data...")
    train_data = pd.read_csv(train_path)
    X = train_data.iloc[:, :-1].values  # 训练特征数据
    y = train_data.iloc[:, -1].values  # 训练标签数据
    print("Loading test data...")
    test_data = pd.read_csv(test_path)
    X_test = test_data.values  # 测试特征数据
    print(f"Training data shape: {X.shape}")
    print(f"Test data shape: {X_test.shape}")
    return X, y, X_test

# 评估模型的性能指标，并返回混淆矩阵数据
def evaluate_model_metrics(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='Evaluating metrics'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    # 计算精确率、召回率、F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted')
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    # 打印详细的分类报告
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds))
    return precision, recall, f1, conf_matrix

# 加载已有模型或训练新模型
def load_or_train_model(model, train_loader, val_loader, device,
                        model_path='nnmodel.pth', force_train=None):
    if os.path.exists(model_path) and force_train is not True:
        if force_train is None:
            response = input("发现已训练的模型，是否使用？(y/n): ").lower()
        else:
            response = 'y'
        if response == 'y':
            print(f"加载模型从 {model_path}")
            model.load_state_dict(torch.load(model_path))
            return model, [], [], [], []  # 返回空列表以继续后续逻辑

    if force_train is False:
        raise FileNotFoundError("未找到已训练的模型文件！")

    print("开始训练新模型...")
    # 初始化训练组件
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    trainer = Trainer(model, criterion, optimizer, scheduler, device)

    # 训练循环
    num_epochs = 50
    best_val_acc = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")

    return model, train_losses, train_accs, val_losses, val_accs

# 自定义数据集类，用于CIFAR-10数据集
class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        data = data.reshape(-1, 3, 32, 32)
        self.data = torch.FloatTensor(data) / 255.0
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[idx]
        return img

# 改进的卷积神经网络模型
class ImprovedConvNet(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ImprovedConvNet, self).__init__()
        # 初始化卷积层和全连接层
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 10)
        )
        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.classifier(x)
        return x

# 训练器类，封装训练和评估过程
class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return running_loss / len(train_loader), 100. * correct / total

    def evaluate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Evaluating'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return running_loss / len(val_loader), 100. * correct / total

# 绘制混淆矩阵热力图
def plot_confusion_matrix(conf_matrix):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix,
                annot=True,  # 显示数值
                fmt='d',  # 整数格式
                cmap='Blues',  # 使用蓝色调色板
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix for CIFAR-10 Classification')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

# 绘制训练过程的损失和准确率曲线
def plot_training_curves(train_losses, train_accs, val_losses, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    plt.show()

# 主函数，执行数据加载、模型训练和评估
def main():
    set_seed(42)  # 设置随机种子
    # 数据增强设置
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    val_transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 加载数据
    print("Loading data...")
    X, y, X_test = load_data('train.csv', 'test.csv')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # 创建数据集和加载器
    train_dataset = CIFAR10Dataset(X_train, y_train, transform=train_transform)
    val_dataset = CIFAR10Dataset(X_val, y_val, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False,
                            num_workers=4, pin_memory=True)
    # 初始化模型
    model = ImprovedConvNet().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    # 加载或训练模型
    model, train_losses, train_accs, val_losses, val_accs = load_or_train_model(
        model, train_loader, val_loader, device)
    # 如果是新训练的模型，绘制训练曲线
    if train_losses is not None:
        plot_training_curves(train_losses, train_accs, val_losses, val_accs)
        # 评估模型性能
        print("\n计算模型性能指标...")
        precision, recall, f1, conf_matrix = evaluate_model_metrics(model, val_loader, device)
        print("\n总体性能指标:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        # 绘制混淆矩阵
        print("\n绘制混淆矩阵...")
        plot_confusion_matrix(conf_matrix)

if __name__ == '__main__':
    mp.freeze_support()  # 处理多进程支持问题
    main()  # 执行主函数