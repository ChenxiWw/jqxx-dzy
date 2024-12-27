import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 1. 数据加载和预处理
def load_and_preprocess_data():
    print("正在加载数据...")
    data = pd.read_csv('train.csv')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    print("正在进行标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# 2. 降维处理
def dimension_reduction(X, n_components=100):
    print(f"正在进行PCA降维到{n_components}维...")
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    X_reduced = pca.fit_transform(X)
    explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
    print(f"降维后保留的方差比例: {explained_variance_ratio:.4f}")
    return X_reduced

# 3. K-means模型训练
def train_kmeans(X, n_clusters=10, batch_size=1024):
    print(f"正在训练MiniBatchKMeans模型 (n_clusters={n_clusters}, batch_size={batch_size})...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                            batch_size=batch_size,
                            random_state=42)
    start_time = time.time()
    kmeans.fit(X)
    end_time = time.time()
    print(f"训练用时: {end_time - start_time:.2f}秒")
    return kmeans

# 4. 聚类结果映射到真实标签
def map_clusters_to_labels(kmeans, X, y):
    print("正在将聚类结果映射到真实标签...")
    cluster_labels = kmeans.predict(X)
    mapping = {}
    for i in range(kmeans.n_clusters):
        mask = (cluster_labels == i)
        if np.sum(mask) > 0:
            mapping[i] = np.bincount(y[mask]).argmax()

    predicted_labels = np.array([mapping[label] for label in cluster_labels])
    return predicted_labels

# 5. 评估指标计算
def evaluate_model(y_true, y_pred):
    print("\n模型评估结果：")
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')

    print(f"准确率: {accuracy:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"精确率: {precision:.4f}")
    print("\n分类报告:")
    print(classification_report(y_true, y_pred))
    return accuracy, recall, precision

# 6. 可视化函数
def plot_confusion_matrix(y_true, y_pred):
    print("正在绘制混淆矩阵...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()

def plot_metrics(accuracy, recall, precision):
    print("正在绘制评估指标图...")
    metrics = ['Accuracy', 'Recall', 'Precision']
    values = [accuracy, recall, precision]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values)
    plt.title('模型评估指标')
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    plt.show()

# 主函数
def main():
    try:
        # 设置参数
        n_components = 100  # PCA降维维度
        n_clusters = 10     # 聚类数量
        batch_size = 4096   # 批处理大小

        # 加载数据
        X, y = load_and_preprocess_data()

        # 划分训练集和测试集
        print("正在划分训练集和测试集...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 降维
        X_train_reduced = dimension_reduction(X_train, n_components)
        X_test_reduced = dimension_reduction(X_test, n_components)

        # 训练模型
        kmeans = train_kmeans(X_train_reduced, n_clusters, batch_size)

        # 预测并评估
        print("正在进行预测...")
        y_pred = map_clusters_to_labels(kmeans, X_test_reduced, y_test)

        # 输出评估结果
        accuracy, recall, precision = evaluate_model(y_test, y_pred)

        # 绘制混淆矩阵和评估指标图
        plot_confusion_matrix(y_test, y_pred)
        plot_metrics(accuracy, recall, precision)

    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()