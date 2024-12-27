import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_model(model_path):
    """
    加载保存的SVM模型

    参数:
    model_path (str): 模型文件的路径

    返回:
    object: 加载的SVM模型
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model





def predict_single_image(image_data, model, scaler):
    """
    对单张图片进行预测

    参数:
    image_data (array): 图片的特征数据
    model: 加载的SVM模型
    scaler: 用于特征标准化的scaler对象

    返回:
    int: 预测的类别
    array: 预测的概率分布
    """
    # 确保输入数据形状正确
    if image_data.ndim == 1:
        image_data = image_data.reshape(1, -1)

    # 应用相同的特征标准化
    scaled_data = scaler.transform(image_data)

    # 进行预测
    prediction = model.predict(scaled_data)[0]
    probabilities = model.predict_proba(scaled_data)[0]

    return prediction, probabilities


def main():
    # 类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 加载模型
    model = load_model('svm_model.pkl')

    # 加载需要进行预测的数据
    # 这里假设我们有一个新的测试样本
    test_data = pd.read_csv('test.csv')  # 替换为你的测试数据文件

    # 加载训练数据的scaler
    # 注意：这个scaler应该和训练时使用的是同一个
    scaler = StandardScaler()
    train_data = pd.read_csv('train.csv')  # 替换为你的训练数据文件
    scaler.fit(train_data.iloc[:, :-1])

    # 对测试样本进行预测
    for idx, row in test_data.iterrows():
        prediction, probabilities = predict_single_image(row.values, model, scaler)

        print(f"\n预测结果:")
        print(f"预测类别: {class_names[prediction]}")
        print("\n各类别概率:")
        for class_name, prob in zip(class_names, probabilities):
            print(f"{class_name}: {prob:.4f}")


if __name__ == "__main__":
    main()