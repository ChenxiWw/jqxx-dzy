import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from tqdm import tqdm

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels=None):
        """
        参数:
            data: numpy数组，形状为(N, 3072)
            labels: numpy数组，形状为(N,)，如果是测试集则为None
        """
        # 将数据从(N, 3072)重构为(N, 3, 32, 32)的形状
        # 3072 = 3 x 32 x 32，其中3是通道数(RGB)，32x32是图像尺寸
        data = data.reshape(-1, 3, 32, 32)
        # 归一化像素值到0-1范围
        self.data = torch.FloatTensor(data) / 255.0

        if labels is not None:
            self.labels = torch.LongTensor(labels)
        else:
            self.labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]


# ConvNet模型定义保持不变
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def load_data(train_path, test_path):

    print("Loading training data...")
    train_data = pd.read_csv(train_path)
    X = train_data.iloc[:, :-1].values  # 所有特征列
    y = train_data.iloc[:, -1].values  # 最后一列是标签

    print("Loading test data...")
    test_data = pd.read_csv(test_path)
    X_test = test_data.values  # 测试数据没有标签

    print(f"Training data shape: {X.shape}")
    print(f"Test data shape: {X_test.shape}")

    return X, y, X_test


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """
    训练模型并记录训练过程
    """
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print('--------------------')

    return train_losses, train_accs, val_losses, val_accs


def plot_training_curves(train_losses, train_accs, val_losses, val_accs):
    """
    绘制训练过程的损失和准确率曲线
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 损失曲线
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # 准确率曲线
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def main():
    # 加载数据
    print("Loading data...")
    X, y, X_test = load_data('train.csv', 'test.csv')

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 创建数据加载器
    train_dataset = CIFAR10Dataset(X_train, y_train)
    val_dataset = CIFAR10Dataset(X_val, y_val)
    test_dataset = CIFAR10Dataset(X_test)

    # 设置批量大小
    batch_size = 128

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # 初始化模型
    model = ConvNet().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    print("Starting training...")
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=20
    )

    # 绘制训练曲线
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)

    # 在验证集上进行预测并计算混淆矩阵
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, range(10))

    # 打印分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))


if __name__ == '__main__':
    mp.freeze_support()
    main()