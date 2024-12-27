import torch
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import *
import time


def load_and_preprocess_data(batch_size=1000):
    data = pd.read_csv('train.csv')
    # 转换为numpy节省内存
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 分批返回数据
    for i in range(0, len(X_scaled), batch_size):
        batch_X = torch.tensor(X_scaled[i:i + batch_size], dtype=torch.float32)
        batch_y = torch.tensor(y[i:i + batch_size], dtype=torch.long)
        yield batch_X.cuda(), batch_y.cuda()


def pca_torch(X, n_components):
    """PyTorch实现的PCA"""
    # 计算协方差矩阵
    X_centered = X - X.mean(dim=0)
    cov = torch.mm(X_centered.t(), X_centered) / (X.shape[0] - 1)

    # 特征值分解
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
def find_optimal_eps(X, k=5):
    """使用k-距离图来找到最优的eps值"""
    # 计算每个点到其他点的距离
    distances = torch.cdist(X[:1000], X[:1000])  # 取样本来计算
    # 获取每个点的第k个最近邻距离
    k_distances = torch.sort(distances)[0][:, k]
    # 对距离进行排序
    sorted_distances = torch.sort(k_distances)[0]

    # 绘制k-距离图
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(sorted_distances)), sorted_distances.cpu().numpy())
    plt.xlabel('样本点')
    plt.ylabel(f'{k}-距离')
    plt.title('k-距离图')
    plt.show()

    # 返回距离的拐点作为建议的eps值
    return sorted_distances[len(sorted_distances)//2].item()
    # 选择前n_components个特征向量
    idx = torch.argsort(eigenvalues, descending=True)[:n_components]
    components = eigenvectors[:, idx]

    # 投影数据
    X_reduced = torch.mm(X_centered, components)

    return X_reduced


def torch_dbscan_optimized(X, eps, min_samples, batch_size=1000):
    n_samples = X.shape[0]
    labels = torch.full((n_samples,), -1, device=X.device)
    core_points = torch.zeros(n_samples, dtype=torch.bool, device=X.device)

    # 分批计算核心点
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_X = X[i:batch_end]

        neighbor_count = torch.zeros(batch_end - i, device=X.device)
        for j in range(0, n_samples, batch_size):
            sub_end = min(j + batch_size, n_samples)
            dist = torch.cdist(batch_X, X[j:sub_end])
            # 显式转换为bool
            neighbor_count += torch.sum((dist <= eps).bool(), dim=1)

        core_points[i:batch_end] = neighbor_count >= min_samples

    cluster_id = 0

    # 分批处理聚类
    for i in range(n_samples):
        if labels[i] != -1 or not core_points[i]:
            continue

        stack = [i]
        labels[i] = cluster_id

        while stack:
            current = stack.pop()

            for j in range(0, n_samples, batch_size):
                end = min(j + batch_size, n_samples)
                dist = torch.cdist(X[current:current + 1], X[j:end])
                neighbors = torch.where((dist[0] <= eps).bool())[0] + j

                mask = labels[neighbors] == -1
                labels[neighbors[mask]] = cluster_id

                stack.extend(neighbors[mask][core_points[neighbors[mask]]].tolist())

        cluster_id += 1

    return labels


def visualize_clusters_2d(X, labels, title="聚类结果可视化"):
    """使用PCA将数据降至2维并可视化聚类结果"""
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
    else:
        X_2d = X

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('第一主成分')
    plt.ylabel('第二主成分')
    plt.show()


def visualize_cluster_distribution(labels):
    """可视化各聚类的样本分布"""
    unique_labels = np.unique(labels)
    counts = [np.sum(labels == label) for label in unique_labels]

    plt.figure(figsize=(12, 6))
    plt.bar(unique_labels, counts)
    plt.title('聚类样本分布')
    plt.xlabel('聚类标签')
    plt.ylabel('样本数量')
    plt.show()


def map_clusters_to_labels(cluster_labels, X, y):
    """将聚类结果映射到真实标签"""
    mapping = {}
    for i in np.unique(cluster_labels):
        mask = (cluster_labels == i)
        if np.sum(mask) > 0:
            mapping[i] = np.bincount(y[mask]).argmax()
    predicted_labels = np.array([mapping[label] for label in cluster_labels])
    return predicted_labels


def evaluate_model(y_true, y_pred):
    """评估模型性能"""
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')

    print(f"\n模型评估结果：")
    print(f"准确率: {accuracy:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"精确率: {precision:.4f}")
    print("\n分类报告:")
    print(classification_report(y_true, y_pred))
    return accuracy, recall, precision


def plot_confusion_matrix(y_true, y_pred):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    try:
        # 分批处理数据
        batch_size = 1000
        data_loader = load_and_preprocess_data(batch_size)

        # 收集所有批次
        X_batches = []
        y_batches = []
        for X_batch, y_batch in data_loader:
            X_batches.append(X_batch)
            y_batches.append(y_batch)

        X = torch.cat(X_batches)
        y = torch.cat(y_batches)

        # PCA降维
        n_components = min(500, X.shape[1])
        print(f"执行PCA降维到{n_components}维...")
        X_reduced = pca_torch(X, n_components)

        # DBSCAN聚类
        print("执行DBSCAN聚类...")
        start_time = time.time()
        labels = torch_dbscan_optimized(X_reduced, eps=0.5, min_samples=5, batch_size=batch_size)
        end_time = time.time()
        print(f"聚类完成，用时: {end_time - start_time:.2f}秒")

        # 将数据移到CPU进行可视化和评估
        X_cpu = X_reduced.cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        y_cpu = y.cpu().numpy()

        # 可视化聚类结果
        print("生成可视化结果...")
        visualize_clusters_2d(X_cpu, labels_cpu, "DBSCAN聚类结果")
        visualize_cluster_distribution(labels_cpu)

        # 将聚类结果映射到真实标签
        print("映射聚类结果到真实标签...")
        predicted_labels = map_clusters_to_labels(labels_cpu, X_cpu, y_cpu)

        # 评估模型性能
        print("评估模型性能...")
        evaluate_model(y_cpu, predicted_labels)

        # 绘制混淆矩阵
        print("生成混淆矩阵...")
        plot_confusion_matrix(y_cpu, predicted_labels)

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()