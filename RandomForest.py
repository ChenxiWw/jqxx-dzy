import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import parallel_backend
import time
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class CIFAR10Classifier:
    def __init__(self, n_estimators=500, max_depth=1000):
        """
        初始化分类器
        参数:
            n_estimators: 随机森林中树的数量
            max_depth: 树的最大深度
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,  # 使用所有可用的CPU核心
            random_state=42
        )
        self.history = {'acc': [], 'val_acc': []}

    def load_data(self, train_path, test_path=None, sample_size=None):
        """
        加载数据集
        参数:
            train_path: 训练数据路径
            test_path: 测试数据路径
            sample_size: 采样大小，用于快速测试
        """
        # 加载训练数据
        train_data = pd.read_csv(train_path)

        if sample_size:
            train_data = train_data.sample(n=sample_size, random_state=42)

        # 分离特征和标签
        X = train_data.iloc[:, :-1].values / 255.0  # 归一化像素值
        y = train_data.iloc[:, -1].values

        # 分割训练集和验证集
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        if test_path:
            test_data = pd.read_csv(test_path)
            self.X_test = test_data.values / 255.0

    def train(self, batch_size=1000):
        """
        训练模型并记录准确率变化
        参数:
            batch_size: 每批训练数据的大小
        """
        print("开始训练模型...")
        start_time = time.time()


        with parallel_backend('threading'):
            # 批量训练并记录准确率
            n_batches = len(self.X_train) // batch_size
            for i in tqdm(range(n_batches)):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size

                # 获取当前批次的数据
                X_batch = self.X_train[start_idx:end_idx]
                y_batch = self.y_train[start_idx:end_idx]

                # 训练当前批次
                self.model.fit(X_batch, y_batch)

                # 记录训练和验证准确率
                train_pred = self.model.predict(X_batch)
                val_pred = self.model.predict(self.X_val)

                self.history['acc'].append(accuracy_score(y_batch, train_pred))
                self.history['val_acc'].append(accuracy_score(self.y_val, val_pred))

        training_time = time.time() - start_time
        print(f"训练完成！用时: {training_time:.2f} 秒")

    def evaluate(self):
        """
        评估模型性能，计算各项指标
        """
        print("\n模型评估结果：")
        y_pred = self.model.predict(self.X_val)

        # 计算各项指标
        accuracy = accuracy_score(self.y_val, y_pred)
        precision = precision_score(self.y_val, y_pred, average='weighted')
        recall = recall_score(self.y_val, y_pred, average='weighted')

        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")

        return accuracy, precision, recall

    def plot_training_history(self):
        """
        绘制训练过程中准确率的变化曲线
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['acc'], label='训练准确率')
        plt.plot(self.history['val_acc'], label='验证准确率')
        plt.title('模型训练过程中的准确率变化')
        plt.xlabel('批次')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_confusion_matrix(self, class_names):
        """
        绘制混淆矩阵
        参数:
            class_names: 类别名称列表
        """
        y_pred = self.model.predict(self.X_val)
        cm = confusion_matrix(self.y_val, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('混淆矩阵')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.show()



def main():
    # 类别名称
    class_names = ['飞机', '汽车', '鸟', '猫', '鹿',
                   '狗', '青蛙', '马', '船', '卡车']

    # 初始化分类器
    classifier = CIFAR10Classifier(n_estimators=500, max_depth=100)

    # 加载数据（示例使用10000个样本进行快速测试）
    classifier.load_data('train.csv', sample_size=50000)

    # 训练模型
    classifier.train(batch_size=1000)

    # 评估模型
    accuracy, precision, recall = classifier.evaluate()

    # 绘制训练历史
    classifier.plot_training_history()

    # 绘制混淆矩阵
    classifier.plot_confusion_matrix(class_names)


if __name__ == "__main__":
    main()