import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from tqdm import tqdm
import time
import os
import joblib  # 用于保存和加载模型

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_preprocess_data(train_path, test_path, chunk_size=1000):
    print("正在加载数据...")
    # 分块读取和处理训练数据
    chunks = []
    for chunk in pd.read_csv(train_path, chunksize=chunk_size):
        X_chunk = chunk.iloc[:, :-1].values
        y_chunk = chunk.iloc[:, -1].values
        chunks.append((X_chunk, y_chunk))
    X_train = np.vstack([c[0] for c in chunks])
    y_train = np.concatenate([c[1] for c in chunks])

    # 同样分块处理测试数据
    test_chunks = []
    for chunk in pd.read_csv(test_path, chunksize=chunk_size):
        test_chunks.append(chunk.values)
    X_test = np.vstack(test_chunks)

    # 使用增量式标准化
    scaler = StandardScaler()
    for i in range(0, len(X_train), chunk_size):
        chunk = X_train[i:i + chunk_size]
        if i == 0:
            scaler.partial_fit(chunk)
        else:
            scaler.partial_fit(chunk)

    # 分块标准化
    for i in range(0, len(X_train), chunk_size):
        X_train[i:i + chunk_size] = scaler.transform(X_train[i:i + chunk_size])
    for i in range(0, len(X_test), chunk_size):
        X_test[i:i + chunk_size] = scaler.transform(X_test[i:i + chunk_size])

    return X_train, y_train, X_test, scaler


def train_optimized_svm(X_train, y_train, subsample_size=5000):
    print("开始训练模型...")
    start_time = time.time()

    # 随机抽样以减少训练时间
    indices = np.random.choice(len(X_train), subsample_size, replace=False)
    X_train_sub = X_train[indices]
    y_train_sub = y_train[indices]

    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_sub, y_train_sub, test_size=0.35, random_state=2
    )

    # 使用预计算核矩阵以提高速度
    svm = SVC(kernel='rbf', C=50.0, probability=True, cache_size=2000)
    svm.fit(X_train_final, y_train_final)

    training_time = time.time() - start_time
    print(f"模型训练完成，用时 {training_time:.2f} 秒")

    train_acc = svm.score(X_train_final, y_train_final)
    val_acc = svm.score(X_val, y_val)

    history = {
        'train_acc': [train_acc],
        'val_acc': [val_acc]
    }

    return svm, history, X_val, y_val


def evaluate_model(model, X_val, y_val):
    print("\n正在评估指标...")
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='macro')
    recall = recall_score(y_val, y_pred, average='macro')

    print("\n模型评估指标:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")

    return accuracy, precision, recall, y_pred


def predict_test_set(model, X_test, chunk_size=1000, output_path='predictions.csv'):
    print("\n开始对测试集进行预测...")
    predictions = []
    for i in range(0, len(X_test), chunk_size):
        X_chunk = X_test[i:i + chunk_size]
        y_pred_chunk = model.predict(X_chunk)
        predictions.extend(y_pred_chunk)

    predictions_df = pd.DataFrame({'prediction': predictions})
    predictions_df.to_csv(output_path, index=False)
    print(f"预测结果已保存至 '{output_path}'")


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_model(model, path='SVMmodel.pth'):
    """保存模型到指定路径"""
    print(f"正在保存模型到 {path}...")
    joblib.dump(model, path)
    print("模型保存完成")


def load_model(path='SVMmodel.pth'):
    """从指定路径加载模型"""
    print(f"正在加载模型从 {path}...")
    model = joblib.load(path)
    print("模型加载完成")
    return model


def main():
    train_path = 'train.csv'
    test_path = 'test.csv'
    model_path = 'SVMmodel.pth'

    # 检查是否存在已训练的模型
    if os.path.exists(model_path):
        while True:
            choice = input("发现已训练的模型，是否使用已有模型？(y/n): ").lower()
            if choice in ['y', 'n']:
                break
            print("请输入 'y' 或 'n'")

        if choice == 'y':
            # 加载已有模型
            model = load_model(model_path)
            # 仍然需要加载数据用于评估
            X_train, y_train, X_test, scaler = load_and_preprocess_data(train_path, test_path)
            # 使用一小部分数据进行评估
            indices = np.random.choice(len(X_train), 5000, replace=False)
            X_val = X_train[indices]
            y_val = y_train[indices]
        else:
            # 重新训练模型
            X_train, y_train, X_test, scaler = load_and_preprocess_data(train_path, test_path)
            model, history, X_val, y_val = train_optimized_svm(X_train, y_train)
            # 保存新训练的模型
            save_model(model, model_path)
    else:
        # 首次训练模型
        print("未发现已训练的模型，开始训练新模型...")
        X_train, y_train, X_test, scaler = load_and_preprocess_data(train_path, test_path)
        model, history, X_val, y_val = train_optimized_svm(X_train, y_train)
        # 保存新训练的模型
        save_model(model, model_path)

    # 评估模型
    accuracy, precision, recall, y_pred = evaluate_model(model, X_val, y_val)

    # 绘制混淆矩阵
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    plot_confusion_matrix(y_val, y_pred, class_names)

    # 预测测试集
    predict_test_set(model, X_test)


if __name__ == "__main__":
    main()